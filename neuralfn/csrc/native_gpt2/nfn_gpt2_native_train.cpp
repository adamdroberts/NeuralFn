#include "token_shards.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
#error "nfn_gpt_native_train currently targets POSIX execvp environments."
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

constexpr std::int64_t kAttentionForwardValueReuse = 64;
constexpr std::int64_t kAttentionBackwardDimReuse = 64;
constexpr std::int64_t kDefaultStoredMlpBlocks = 12;
constexpr std::int64_t kDefaultStoredPackedAttentionBlocks = 12;
constexpr int kDefaultLmHeadRowChunkSize = 8192;

struct Config {
    std::string model_family = "gpt";
    std::string dataset_alias = "roneneldan__TinyStories__TinyStoriesV2-GPT4";
    std::string target;
    std::string output_dir;
    std::string backend = "tile-cuda";
    std::string tile_ops_lib;
    std::string activation = "gelu";
    std::string template_name = "gpt";
    std::string graph_file;
    std::string json_out_path;
    int moa_interval = 50;
    int eval_every_steps = 250;
    int sample_every_steps = 20000;
    int generate_tokens = 144;
    int checkpoint_every_steps = 200;
    int batch_size = 64;
    int seq_len = 1024;
    int train_batch_tokens = 524288;
    int warmup_steps = 60;
    int max_steps = 20000;
    int num_layers = 12;
    int eval_batches = 1;
    int eval_batch_size = 0;
    int lm_head_row_chunk_size = kDefaultLmHeadRowChunkSize;
    double learning_rate = 0.0006;
    double final_lr_fraction = 0.0;
    double weight_decay = 0.1;
    bool allow_train_as_val = false;
    bool dry_run = false;
    bool print_command = false;
    bool print_plan = false;
    bool check_tile_ops = false;
    bool smoke_tile_ops = false;
    bool smoke_optimizer_step = false;
    bool smoke_lm_step = false;
    bool smoke_attention_step = false;
    bool smoke_mlp_step = false;
    bool smoke_norm_residual_step = false;
    bool smoke_transformer_block_step = false;
    bool smoke_transformer_lm_step = false;
    bool smoke_embedding_lm_step = false;
    bool train_embedding_lm = false;
    bool train_transformer_lm = true;
    bool startup_only = false;
    bool checkpoint_metadata_smoke = false;
    bool write_checkpoint = true;
    bool template_explicit = false;
    bool seq_len_explicit = false;
    std::string cuda_runtime_lib;
};

struct BufferPlan {
    std::string name;
    std::vector<std::int64_t> shape;
    std::int64_t offset = 0;
    std::int64_t count = 0;
    bool weight_decay = true;
};

struct StagePlan {
    std::string name;
    std::string phase;
    std::string kernel_abi;
    std::int64_t elements = 0;
};

struct ValidationLossRecord {
    std::int64_t step = 0;
    std::int64_t batches = 0;
    std::int64_t tokens = 0;
    double loss_sum = 0.0;
    double loss_mean = 0.0;
};

std::string env_or_empty(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

class ScopedStdoutRedirect {
  public:
    ScopedStdoutRedirect() = default;

    ScopedStdoutRedirect(const ScopedStdoutRedirect&) = delete;
    ScopedStdoutRedirect& operator=(const ScopedStdoutRedirect&) = delete;

    ~ScopedStdoutRedirect() {
        if (previous_ != nullptr) {
            std::cout.rdbuf(previous_);
        }
    }

    bool open(const std::string& path, std::string* error) {
        if (path.empty()) {
            return true;
        }
        const fs::path out_path(path);
        const fs::path parent = out_path.parent_path();
        if (!parent.empty()) {
            std::error_code ec;
            fs::create_directories(parent, ec);
            if (ec) {
                if (error != nullptr) {
                    *error = "failed to create JSON output directory " + parent.string() + ": " + ec.message();
                }
                return false;
            }
        }
        file_.open(out_path, std::ios::out | std::ios::trunc);
        if (!file_) {
            if (error != nullptr) {
                *error = "failed to open JSON output file " + out_path.string() + ": " + std::strerror(errno);
            }
            return false;
        }
        previous_ = std::cout.rdbuf(file_.rdbuf());
        return true;
    }

  private:
    std::ofstream file_;
    std::streambuf* previous_ = nullptr;
};

std::string env_or_empty_any(std::initializer_list<const char*> names) {
    for (const char* name : names) {
        std::string value = env_or_empty(name);
        if (!value.empty()) {
            return value;
        }
    }
    return {};
}

bool packed_qkv_attention_default_enabled() {
    const std::string value =
        env_or_empty_any({"NFN_NATIVE_GPT_PACKED_QKV_ATTENTION", "NFN_NATIVE_GPT2_PACKED_QKV_ATTENTION"});
    return value.empty() ||
           value == "1" ||
           value == "true" ||
           value == "TRUE" ||
           value == "on" ||
           value == "ON";
}

bool env_flag_enabled(const std::string& value) {
    return value == "1" ||
           value == "true" ||
           value == "TRUE" ||
           value == "on" ||
           value == "ON";
}

bool env_flag_enabled_or_default(const std::string& value, bool default_value) {
    if (value.empty()) {
        return default_value;
    }
    return env_flag_enabled(value);
}

std::int64_t env_nonnegative_i64_or(std::initializer_list<const char*> names, std::int64_t fallback) {
    const std::string value = env_or_empty_any(names);
    if (value.empty()) {
        return fallback;
    }
    char* end = nullptr;
    errno = 0;
    const long long parsed = std::strtoll(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || parsed < 0) {
        return fallback;
    }
    return static_cast<std::int64_t>(parsed);
}

bool fuse_attention_residual_ln2_default_enabled() {
    const std::string value = env_or_empty_any(
        {"NFN_NATIVE_GPT_FUSE_ATTENTION_RESIDUAL_LN2", "NFN_NATIVE_GPT2_FUSE_ATTENTION_RESIDUAL_LN2"});
    return value.empty() ||
           value == "1" ||
           value == "true" ||
           value == "TRUE" ||
           value == "on" ||
           value == "ON";
}

fs::path home_dir() {
    std::string home = env_or_empty("HOME");
    if (home.empty()) {
        return fs::current_path();
    }
    return fs::path(home);
}

std::string default_target() {
    std::string generic_env_target = env_or_empty("NFN_NATIVE_GPT_TRAIN_BIN");
    if (!generic_env_target.empty()) {
        return generic_env_target;
    }
    std::string env_target = env_or_empty("NFN_NATIVE_GPT2_TRAIN_BIN");
    if (!env_target.empty()) {
        return env_target;
    }
    fs::path known = "/mnt/disk2/dev/open-source/llm.kittens/train_gpt2cu";
    if (fs::exists(known)) {
        return known.string();
    }
    return "train_gpt2cu";
}

std::string default_output_dir() {
    std::string env_output = env_or_empty("NATIVE_CUDA_OUTPUT_DIR");
    if (!env_output.empty()) {
        return env_output;
    }
    return (home_dir() / "NeuralFn" / "artifacts" / "gpt").string();
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
}

void print_invocation_command(int argc, char** argv) {
    std::vector<std::string> command;
    command.reserve(static_cast<std::size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        command.emplace_back(argv[i]);
    }
    print_command(command);
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "Native no-Python dense GPT trainer entrypoint for cached uint16 NeuralFn datasets.\n"
        << "Requires fineweb_train_*.bin and fineweb_val_*.bin under the dataset directory.\n\n"
        << "Dataset options:\n"
        << "  --dataset-alias NAME_OR_PATH      Dataset alias under ~/.cache/nfn/datasets or absolute path\n"
        << "  --tinystories                     Shortcut for roneneldan__TinyStories__TinyStoriesV2-GPT4\n"
        << "  --allow-train-val-fallback        Reuse train shard when no validation shard exists\n\n"
        << "Template/graph options:\n"
        << "  --model-family gpt|gpt2|gpt3    Dense GPT selector; all canonicalize to model_family=gpt, while gpt3 can default to 2048 context\n"
        << "  --template-name NAME              GPT template preset or alias to select; default gpt resolves to the dense GPT native implementation\n"
        << "  --graph-file PATH                 Custom NeuralFn graph JSON to select; reports missing native graph trainer until implemented\n\n"
        << "Launch options:\n"
        << "  --backend llm-kittens|tile-cuda   Backend; defaults to tile-cuda and the NeuralFn-owned trainer path\n"
        << "  --target PATH                     llm-kittens train_gpt2cu path; unused by the default tile-cuda backend\n"
        << "  --tile-ops-lib PATH               libnfn_native_train_tile_ops.so path for --backend tile-cuda checks\n"
        << "  --check-tile-ops                  Verify raw NeuralFn Tile trainer ABI symbols and exit\n"
        << "  --smoke-tile-ops                  Launch nfn_native_tile_fill_float32 through CUDA runtime and verify copyback\n"
        << "  --smoke-optimizer-step            Run one AdamW update over the dense GPT registered parameter layout\n"
        << "  --smoke-lm-step                   Run a tiny tied-embedding dense GPT LM step through raw Tile kernels\n"
        << "  --smoke-attention-step            Run a tiny dense GPT attention stage through raw Tile kernels\n"
        << "  --smoke-mlp-step                  Run a tiny dense GPT MLP stage through raw Tile kernels\n"
        << "  --smoke-norm-residual-step        Run a tiny dense GPT LayerNorm/residual/backward stage through raw Tile kernels\n"
        << "  --smoke-transformer-block-step    Run a tiny dense GPT transformer block through raw Tile kernels\n"
        << "  --smoke-transformer-lm-step       Sample cached tokens and run embeddings, one transformer block, final norm, tied LM head, CE, backward, and AdamW\n"
        << "  --smoke-embedding-lm-step         Sample cached tokens and run dense GPT embedding/final-norm/LM-head kernels\n"
        << "  --train-embedding-lm              Train dense GPT embedding/final-norm/LM-head path over cached shards with Tile kernels\n"
        << "  --train-transformer-lm            Run the dense GPT transformer/LM training loop with validation JSON (default)\n"
        << "  --startup-only                    Run full Tile-CUDA transformer setup and exit before optimizer steps\n"
        << "  --no-train-transformer-lm         Disable the default transformer-LM loop for plan/check/debug commands\n"
        << "  --no-checkpoint                   Skip final trained checkpoint export for speed/preflight runs\n"
        << "  --checkpoint-metadata-smoke       Write a sparse native dense GPT checkpoint-format artifact and DONE marker without CUDA/Torch\n"
        << "  --cuda-runtime-lib PATH           libcudart path for Tile-CUDA smokes/training; defaults to NFN_CUDA_RUNTIME_LIB/libcudart.so\n"
        << "  --json-out PATH                   Write native JSON output to PATH instead of stdout\n"
        << "  --profile-json PATH               Alias for --json-out; useful with NFN_NATIVE_GPT_STAGE_TIMING=1\n"
        << "  --print-plan                      Print native backend JSON plan and exit\n"
        << "  --output-dir PATH                 Native output directory\n"
        << "  --dry-run                         Print/resolve without exec\n"
        << "  --print-command                   Print the backend command without training; tile-cuda exits before CUDA/shard setup\n\n"
        << "Training options mirror train_gpt.py names, including --eval-every-steps, --eval-batches, --eval-batch-size, --batch-size, --train-seq-len,\n"
        << "  --train-batch-tokens, --learning-rate, --final-lr-fraction, --weight-decay, --warmup-steps, and --max-steps.\n"
        << "  --lm-head-row-chunk-size N        Tied LM-head full-vocab row chunk size for the Tile-CUDA transformer loop; default "
        << kDefaultLmHeadRowChunkSize << ".\n"
        << "Dataset default: roneneldan__TinyStories__TinyStoriesV2-GPT4.\n"
        << "SM120 defaults match llm.kittens/train-sm120.sh: -v 250 -b 64 -t 1024 -d 524288 -l 0.0006 -q 0.0 -c 0.1 -u 60 -x 20000.\n";
}

std::string require_value(int argc, char** argv, int* index, const std::string& flag) {
    if (*index + 1 >= argc) {
        std::cerr << flag << " requires a value\n";
        std::exit(2);
    }
    *index += 1;
    return argv[*index];
}

int parse_int(const std::string& value, const std::string& flag) {
    try {
        std::size_t consumed = 0;
        int parsed = std::stoi(value, &consumed);
        if (consumed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return parsed;
    } catch (const std::exception&) {
        std::cerr << flag << " expects an integer, got " << value << "\n";
        std::exit(2);
    }
}

std::string number_string(double value) {
    std::ostringstream out;
    out << value;
    return out.str();
}

std::int64_t shape_count(const std::vector<std::int64_t>& shape) {
    std::int64_t total = 1;
    for (const std::int64_t dim : shape) {
        total *= dim;
    }
    return total;
}

std::string json_escape(const std::string& value) {
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

std::string normalize_template_name(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (char ch : value) {
        if (ch == '-') {
            out.push_back('_');
        } else {
            out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
    }
    return out.empty() ? "gpt" : out;
}

std::string resolved_native_template_name(const std::string& value) {
    const std::string normalized = normalize_template_name(value);
    return normalized == "gpt" ? "gpt2" : normalized;
}

std::string normalize_model_family(const std::string& value) {
    std::string normalized;
    normalized.reserve(value.size());
    for (char ch : value) {
        char lowered = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        normalized.push_back(lowered == '_' ? '-' : lowered);
    }
    if (normalized.empty()) {
        return "gpt";
    }
    if (normalized == "gpt" || normalized == "gpt2" || normalized == "gpt3") {
        return normalized;
    }
    throw std::runtime_error("model family must be one of: gpt, gpt2, gpt3");
}

std::string canonical_dense_gpt_model_family(const std::string& model_selector) {
    const std::string normalized = normalize_model_family(model_selector);
    if (normalized == "gpt" || normalized == "gpt2" || normalized == "gpt3") {
        return "gpt";
    }
    return normalized;
}

bool is_default_gpt_template(const Config& cfg) {
    return resolved_native_template_name(cfg.template_name) == "gpt2";
}

const std::vector<std::string>& shipped_gpt_template_presets() {
    static const std::vector<std::string> presets = {
        "nanogpt",
        "nanogpt_megakernel",
        "gpt2",
        "gpt2_megakernel",
        "gpt2_moa",
        "llama",
        "modern_norms_llama",
        "mixllama",
        "moe",
        "llama_fast",
        "llama_fast_megakernel",
        "mixllama_fast",
        "mixllama_fast_megakernel",
        "jamba",
        "ternary_b158",
        "fp8_llama",
        "mxfp4_llama",
        "deepseek_v3",
        "deepseek_v4",
        "gemma3",
        "diff_transformer",
        "longctx_sparse_llama",
        "qwen3_longctx",
        "auxfree_moe_jepa_evo",
        "diff_semantic_moe_jepa_evo",
        "dyt_geglu_semantic_dense_jepa_evo",
        "llama_megakernel",
        "kv_pca_llama",
        "seq2seq",
        "diffusion",
        "ttt_llama",
        "llm_jepa",
        "dense_jepa_evo",
        "moe_jepa_evo",
        "jepa_semantic_hybrid",
        "jepa_semantic_hybrid_megakernel",
        "semantic_router_moe",
        "semantic_router_moe_megakernel",
        "semantic_moe_jepa_evo",
        "semantic_dense_jepa_evo",
        "hnet_lm",
        "universal_llama",
        "nanogpt_modern",
        "gpt2_modern",
        "llama_modern",
        "moe_modern",
        "jamba_modern",
        "ternary_b158_modern",
        "seq2seq_modern",
        "diffusion_modern",
        "ttt_llama_modern",
        "llm_jepa_modern",
        "dense_jepa_evo_modern",
        "moe_jepa_evo_modern",
        "hnet_lm_modern",
        "universal_llama_modern",
        "kv_pca_llama_modern",
        "jepa_semantic_hybrid_modern",
        "semantic_router_moe_modern",
        "semantic_dense_jepa_evo_modern",
        "semantic_moe_jepa_evo_modern",
    };
    return presets;
}

bool selected_template_is_shipped(const Config& cfg) {
    const std::string name = normalize_template_name(cfg.template_name);
    if (name == "gpt") {
        return true;
    }
    const std::vector<std::string>& presets = shipped_gpt_template_presets();
    return std::find(presets.begin(), presets.end(), name) != presets.end();
}

bool selected_template_is_native_dense_gpt_compatible(const Config& cfg) {
    const std::string name = resolved_native_template_name(cfg.template_name);
    return name == "gpt2" || name == "gpt2_megakernel" || name == "gpt2_moa";
}

bool selected_graph_is_native_runnable(const Config& cfg) {
    return cfg.graph_file.empty() && selected_template_is_native_dense_gpt_compatible(cfg);
}

std::string selected_graph_support_status(const Config& cfg) {
    if (!cfg.graph_file.empty()) {
        return "custom-graph-native-trainer-missing";
    }
    if (!selected_template_is_shipped(cfg)) {
        return "unknown-template";
    }
    return selected_template_is_native_dense_gpt_compatible(cfg) ? "native-transformer-lm"
                                                                : "template-native-trainer-missing";
}

std::string selected_architecture_source(const Config& cfg) {
    return cfg.graph_file.empty() ? "template" : "custom_graph";
}

std::string dense_gpt_architecture_contract(const Config& cfg) {
    if (!cfg.graph_file.empty()) {
        return "custom-graph-file";
    }
    return "gpt-template-preset";
}

std::string model_family_context_policy(const Config& cfg) {
    (void)cfg;
    return "dense-gpt-selectors-canonicalize-to-gpt-template-or-graph-selects-architecture";
}

std::int64_t native_gpt2_parameter_count(
    std::int64_t max_seq_len,
    std::int64_t padded_vocab_size,
    std::int64_t num_layers,
    std::int64_t channels) {
    const std::int64_t c = channels;
    const std::int64_t l = num_layers;
    return
        padded_vocab_size * c +
        max_seq_len * c +
        l * c +
        l * c +
        l * 3 * c * c +
        l * 3 * c +
        l * c * c +
        l * c +
        l * c +
        l * c +
        l * 4 * c * c +
        l * 4 * c +
        l * c * 4 * c +
        l * c +
        c +
        c;
}

bool write_sparse_native_gpt2_checkpoint(
    const fs::path& checkpoint_path,
    std::int64_t max_seq_len,
    std::int64_t vocab_size,
    std::int64_t num_layers,
    std::int64_t num_heads,
    std::int64_t channels,
    std::int64_t padded_vocab_size,
    std::string* error) {
    constexpr std::int32_t kMagic = 20240326;
    constexpr std::int32_t kVersion = 5;
    constexpr std::int64_t kHeaderInts = 256;
    constexpr std::int64_t kHeaderBytes = kHeaderInts * 4;
    constexpr std::int64_t kBytesPerParam = 2;

    if (max_seq_len <= 0 || vocab_size <= 0 || num_layers <= 0 || num_heads <= 0 || channels <= 0 || padded_vocab_size <= 0) {
        if (error != nullptr) {
            *error = "checkpoint shape values must be positive";
        }
        return false;
    }

    std::vector<std::int32_t> header(static_cast<std::size_t>(kHeaderInts), 0);
    header[0] = kMagic;
    header[1] = kVersion;
    header[2] = static_cast<std::int32_t>(max_seq_len);
    header[3] = static_cast<std::int32_t>(vocab_size);
    header[4] = static_cast<std::int32_t>(num_layers);
    header[5] = static_cast<std::int32_t>(num_heads);
    header[6] = static_cast<std::int32_t>(channels);
    header[7] = static_cast<std::int32_t>(padded_vocab_size);

    const std::int64_t parameter_count = native_gpt2_parameter_count(
        max_seq_len, padded_vocab_size, num_layers, channels);
    const std::int64_t file_bytes = kHeaderBytes + parameter_count * kBytesPerParam;

    try {
        fs::create_directories(checkpoint_path.parent_path());
    } catch (const std::exception& exc) {
        if (error != nullptr) {
            *error = std::string("failed to create checkpoint directory: ") + exc.what();
        }
        return false;
    }

    std::ofstream out(checkpoint_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        if (error != nullptr) {
            *error = "failed to open checkpoint for writing: " + checkpoint_path.string();
        }
        return false;
    }
    out.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(kHeaderBytes));
    if (!out) {
        if (error != nullptr) {
            *error = "failed to write checkpoint header: " + checkpoint_path.string();
        }
        return false;
    }
    if (file_bytes > 0) {
        out.seekp(static_cast<std::streamoff>(file_bytes - 1));
        const char zero = '\0';
        out.write(&zero, 1);
    }
    out.close();
    if (!out) {
        if (error != nullptr) {
            *error = "failed to finish checkpoint file: " + checkpoint_path.string();
        }
        return false;
    }
    return true;
}

double parse_double(const std::string& value, const std::string& flag) {
    try {
        std::size_t consumed = 0;
        double parsed = std::stod(value, &consumed);
        if (consumed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return parsed;
    } catch (const std::exception&) {
        std::cerr << flag << " expects a number, got " << value << "\n";
        std::exit(2);
    }
}

std::string lower_activation(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    if (value == "sd_prelu" || value == "sdprelu") {
        value = "sd-prelu";
    }
    return value;
}

bool valid_activation(const std::string& value) {
    static const std::vector<std::string> allowed = {
        "gelu", "relu", "silu", "relu2", "prelu", "sd-prelu", "swiglu", "geglu", "ensemble", "moa",
    };
    for (const std::string& item : allowed) {
        if (value == item) {
            return true;
        }
    }
    return false;
}

void apply_template_activation_defaults(Config& cfg) {
    if (resolved_native_template_name(cfg.template_name) == "gpt2_moa" && lower_activation(cfg.activation) == "gelu") {
        cfg.activation = "moa";
    }
}

bool valid_backend(const std::string& value) {
    return value == "llm-kittens" || value == "tile-cuda";
}

std::string normalize_backend(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

template <typename Fn>
Fn load_symbol(void* handle, const char* name) {
    dlerror();
    void* symbol = dlsym(handle, name);
    if (symbol == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<Fn>(symbol);
}

std::string dl_last_error(const char* fallback) {
    const char* error = dlerror();
    return error == nullptr ? std::string(fallback) : std::string(error);
}

std::vector<std::string> cuda_runtime_candidates(const Config& cfg) {
    if (!cfg.cuda_runtime_lib.empty()) {
        return {cfg.cuda_runtime_lib};
    }
    std::string env_value = env_or_empty("NFN_CUDA_RUNTIME_LIB");
    if (!env_value.empty()) {
        return {env_value};
    }
    return {"libcudart.so", "libcudart.so.13", "libcudart.so.12"};
}

std::string cuda_version_string(int encoded_version) {
    if (encoded_version <= 0) {
        return "unknown";
    }
    const int major = encoded_version / 1000;
    const int minor = (encoded_version % 1000) / 10;
    return std::to_string(major) + "." + std::to_string(minor);
}

std::vector<std::string> required_tile_symbols() {
    return {
        "nfn_native_tile_fill_float32",
        "nfn_native_tile_fill_many_float32",
        "nfn_native_tile_fill_many_values_float32",
        "nfn_native_tile_fill_many_values_bf16_bits_float32",
        "nfn_native_tile_init_gpt2_token_weight_float32",
        "nfn_native_tile_init_gpt2_token_weight_fast_float32",
        "nfn_native_tile_init_gpt2_token_weight_with_bf16_shadow_float32",
        "nfn_native_tile_init_gpt2_token_weight_fast_with_bf16_shadow_float32",
        "nfn_native_tile_uint16_to_int64",
        "nfn_native_tile_float32_to_bf16_bits",
        "nfn_native_tile_bf16_bits_to_float32",
        "nfn_native_tile_bf16_bits_add_bias_inplace_float32",
        "nfn_native_tile_store_mlp_activations_bf16_float32",
        "nfn_native_tile_restore_mlp_activations_bf16_float32",
        "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_float32",
        "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_float32",
        "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_float32",
        "nfn_native_tile_linear_backward_input_dgelu_bf16_bits_float32",
        "nfn_native_tile_linear_backward_input_dgelu_weight_bf16_bits_float32",
        "nfn_native_tile_linear_backward_input_dgelu_weight_bf16_bits_only_float32",
        "nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_float32",
        "nfn_native_tile_linear_weight_bf16_gelu_bf16_float32",
        "nfn_native_tile_linear_bf16_input_weight_bf16_gelu_bf16_float32",
        "nfn_native_tile_gelu_backward_inplace_bf16_bits_float32",
        "nfn_native_tile_float32_to_bf16_bits_many",
        "nfn_native_tile_gradient_accumulate_float32",
        "nfn_native_tile_sum_partials_float32",
        "nfn_native_tile_sumsq_partials_float32",
        "nfn_native_tile_sumsq_partials_many_float32",
        "nfn_native_tile_sumsq_partials_many_bf16_bits_float32",
        "nfn_native_tile_global_norm_clip_scale_float32",
        "nfn_native_tile_scale_inplace_by_device_float32",
        "nfn_native_tile_adamw_step_float32",
        "nfn_native_tile_adamw_step_with_device_scale_float32",
        "nfn_native_tile_adamw_step_many_with_device_scale_float32",
        "nfn_native_tile_adamw_step_many_with_device_scale_bf16_shadow_float32",
        "nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_float32",
        "nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32",
        "nfn_native_tile_token_embedding_float32",
        "nfn_native_tile_token_embedding_backward_weight_float32",
        "nfn_native_tile_absolute_position_embedding_float32",
        "nfn_native_tile_absolute_position_embedding_backward_float32",
        "nfn_native_tile_absolute_position_embedding_backward_accumulate_float32",
        "nfn_native_tile_layer_norm_float32",
        "nfn_native_tile_layer_norm_apply_stats_bf16_out_float32",
        "nfn_native_tile_layer_norm_backward_input_float32",
        "nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_float32",
        "nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_bf16_bits_float32",
        "nfn_native_tile_layer_norm_backward_affine_float32",
        "nfn_native_tile_layer_norm_backward_affine_accumulate_float32",
        "nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_bf16_bits_float32",
        "nfn_native_tile_layer_norm_backward_affine_residual_add_accumulate_with_stats_float32",
        "nfn_native_tile_layer_norm_backward_affine_residual_add_accumulate_with_stats_bf16_bits_float32",
        "nfn_native_tile_linear_float32",
        "nfn_native_tile_linear_bf16_float32",
        "nfn_native_tile_linear_weight_bf16_float32",
        "nfn_native_tile_linear_weight_bf16_output_float32",
        "nfn_native_tile_linear_bf16_input_float_weight_bf16_output_float32",
        "nfn_native_tile_linear_bf16_input_weight_bf16_float32",
        "nfn_native_tile_linear_backward_input_float32",
        "nfn_native_tile_linear_backward_input_weight_bf16_float32",
        "nfn_native_tile_linear_backward_input_weight_bf16_to_bf16_bits_float32",
        "nfn_native_tile_linear_backward_weight_float32",
        "nfn_native_tile_linear_backward_weight_accumulate_float32",
        "nfn_native_tile_linear_backward_weight_accumulate_float32_bf16_bits",
        "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32",
        "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta",
        "nfn_native_tile_linear_backward_weight_accumulate_bf16_float32",
        "nfn_native_tile_linear_backward_bias_float32",
        "nfn_native_tile_linear_backward_bias_accumulate_float32",
        "nfn_native_tile_linear_backward_weight_bias_accumulate_float32_bf16_bits",
        "nfn_native_tile_linear_backward_weight_bias_accumulate_float32_bf16_bits_beta",
        "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_float32_beta",
        "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32",
        "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32_beta",
        "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_to_bf16_bits_float32",
        "nfn_native_tile_split_qkv_float32",
        "nfn_native_tile_split_qkv_to_heads_float32",
        "nfn_native_tile_split_qkv_to_heads_add_bias_float32",
        "nfn_native_tile_merge_qkv_float32",
        "nfn_native_tile_merge_heads_to_qkv_float32",
        "nfn_native_tile_reshape_heads_float32",
        "nfn_native_tile_merge_heads_float32",
        "nfn_native_tile_scaled_dot_product_attention_float32",
        "nfn_native_tile_attention_forward_stats_reset",
        "nfn_native_tile_attention_forward_row_launch_count",
        "nfn_native_tile_attention_forward_tk_launch_count",
        "nfn_native_tile_attention_backward_tk_launch_count",
        "nfn_native_tile_attention_tk_workspace_allocation_count",
        "nfn_native_tile_attention_tk_workspace_element_capacity",
        "nfn_native_tile_attention_tk_workspace_row_capacity",
        "nfn_native_tile_attention_forward_row_fallback_count",
        "nfn_native_tile_attention_forward_scalar_launch_count",
        "nfn_native_tile_attention_forward_row_last_error",
        "nfn_native_tile_attention_forward_row_prelaunch_clear_error",
        "nfn_native_tile_attention_forward_row_prelaunch_peek_error",
        "nfn_native_tile_attention_forward_row_grid_x",
        "nfn_native_tile_attention_forward_row_grid_y",
        "nfn_native_tile_attention_forward_row_grid_z",
        "nfn_native_tile_attention_forward_row_block_x",
        "nfn_native_tile_attention_forward_row_attr_status",
        "nfn_native_tile_attention_forward_row_attr_max_threads_per_block",
        "nfn_native_tile_attention_forward_row_attr_num_regs",
        "nfn_native_tile_attention_forward_row_attr_shared_size_bytes",
        "nfn_native_tile_attention_forward_row_attr_const_size_bytes",
        "nfn_native_tile_attention_forward_row_attr_local_size_bytes",
        "nfn_native_tile_trainer_linear_stats_reset",
        "nfn_native_tile_trainer_linear_bf16_cache_reset",
        "nfn_native_tile_trainer_linear_bf16_gemm_count",
        "nfn_native_tile_trainer_linear_tk_gemm_count",
        "nfn_native_tile_trainer_linear_tk_float_out_gemm_count",
        "nfn_native_tile_trainer_linear_cublaslt_gemm_count",
        "nfn_native_tile_trainer_linear_sgemm_count",
        "nfn_native_tile_trainer_linear_bf16_a_pack_count",
        "nfn_native_tile_trainer_linear_bf16_a_cache_hit_count",
        "nfn_native_tile_trainer_linear_bf16_cache_reset_count",
        "nfn_native_tile_trainer_linear_bf16_workspace_allocation_count",
        "nfn_native_tile_trainer_linear_bf16_workspace_a_capacity",
        "nfn_native_tile_trainer_linear_bf16_workspace_b_capacity",
        "nfn_native_tile_trainer_linear_bf16_cached_a_capacity",
        "nfn_native_tile_trainer_linear_bf16_cache_entry_count",
        "nfn_native_tile_trainer_linear_shape_stats_count",
        "nfn_native_tile_trainer_linear_shape_stats_entry",
        "nfn_native_tile_scaled_dot_product_attention_backward_float32",
        "nfn_native_tile_scaled_dot_product_attention_backward_from_merged_grad_float32",
        "nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_merged_grad_float32",
        "nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_reuse_forward_from_merged_grad_float32",
        "nfn_native_tile_scaled_dot_product_attention_store_tk_bf16_float32",
        "nfn_native_tile_attention_tk_store_forward_workspace_bf16",
        "nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_saved_tk_bf16_from_merged_grad_float32",
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32",
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_store_lse_bf16_float32",
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_merged_grad_float32",
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_saved_lse_bf16_from_merged_grad_float32",
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_merged_grad_float32",
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_merged_grad_float32",
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_bf16_merged_grad_float32",
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_bf16_merged_grad_float32",
        "nfn_native_tile_scaled_residual_add_float32",
        "nfn_native_tile_linear_bias_residual_add_float32",
        "nfn_native_tile_linear_bias_residual_add_bf16_linear_float32",
        "nfn_native_tile_linear_bias_residual_layer_norm_float32",
        "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_float32",
        "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_linear_float32",
        "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_residual_float32",
        "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_float32",
        "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_residual_bf16_norm_float32",
        "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_bf16_norm_float32",
        "nfn_native_tile_gelu_float32",
        "nfn_native_tile_gelu_add_bias_float32",
        "nfn_native_tile_gelu_backward_float32",
        "nfn_native_tile_token_cross_entropy_partials_float32",
        "nfn_native_tile_token_cross_entropy_partials_bf16_bits",
        "nfn_native_tile_token_cross_entropy_backward_with_workspace_float32",
        "nfn_native_tile_token_cross_entropy_backward_inplace_with_workspace_float32",
    };
}

std::vector<BufferPlan> build_gpt2_parameter_layout(const Config& cfg) {
    std::vector<BufferPlan> layout;
    std::int64_t offset = 0;
    constexpr std::int64_t dim = 768;
    constexpr std::int64_t padded_vocab = 50304;
    const std::int64_t mlp_dim = dim * 4;

    auto add = [&](std::string name, std::vector<std::int64_t> shape, bool weight_decay) {
        BufferPlan buffer;
        buffer.name = std::move(name);
        buffer.shape = std::move(shape);
        buffer.offset = offset;
        buffer.count = shape_count(buffer.shape);
        buffer.weight_decay = weight_decay;
        offset += buffer.count;
        layout.push_back(std::move(buffer));
    };

    add("wte.weight", {padded_vocab, dim}, true);
    add("wpe.weight", {cfg.seq_len, dim}, true);
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        const std::string prefix = "h." + std::to_string(layer) + ".";
        add(prefix + "ln_1.weight", {dim}, false);
        add(prefix + "ln_1.bias", {dim}, false);
        add(prefix + "attn.c_attn.weight", {3 * dim, dim}, true);
        add(prefix + "attn.c_attn.bias", {3 * dim}, false);
        add(prefix + "attn.c_proj.weight", {dim, dim}, true);
        add(prefix + "attn.c_proj.bias", {dim}, false);
        add(prefix + "ln_2.weight", {dim}, false);
        add(prefix + "ln_2.bias", {dim}, false);
        add(prefix + "mlp.c_fc.weight", {mlp_dim, dim}, true);
        add(prefix + "mlp.c_fc.bias", {mlp_dim}, false);
        add(prefix + "mlp.c_proj.weight", {dim, mlp_dim}, true);
        add(prefix + "mlp.c_proj.bias", {dim}, false);
    }
    add("ln_f.weight", {dim}, false);
    add("ln_f.bias", {dim}, false);
    return layout;
}

std::int64_t layout_count(const std::vector<BufferPlan>& layout) {
    if (layout.empty()) {
        return 0;
    }
    const BufferPlan& last = layout.back();
    return last.offset + last.count;
}

void print_shape_json(const std::vector<std::int64_t>& shape) {
    std::cout << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << shape[i];
    }
    std::cout << "]";
}

void print_string_array_json(const std::vector<std::string>& values, const std::string& indent) {
    std::cout << "[\n";
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::cout << indent << "\"" << json_escape(values[i]) << "\"";
        if (i + 1 != values.size()) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout << "  ]";
}

void print_parameter_layout_json(const std::vector<BufferPlan>& layout) {
    std::int64_t decay = 0;
    std::int64_t no_decay = 0;
    for (const BufferPlan& buffer : layout) {
        if (buffer.weight_decay) {
            ++decay;
        } else {
            ++no_decay;
        }
    }
    const std::int64_t total = layout_count(layout);
    std::cout
        << "{\n"
        << "    \"buffer_count\": " << layout.size() << ",\n"
        << "    \"decay_buffer_count\": " << decay << ",\n"
        << "    \"no_decay_buffer_count\": " << no_decay << ",\n"
        << "    \"total_parameters\": " << total << ",\n"
        << "    \"gradient_elements\": " << total << ",\n"
        << "    \"adamw_state_elements\": " << (total * 2) << ",\n"
        << "    \"buffers\": [\n";
    for (std::size_t i = 0; i < layout.size(); ++i) {
        const BufferPlan& buffer = layout[i];
        std::cout << "      {\"name\": \"" << json_escape(buffer.name) << "\", \"shape\": ";
        print_shape_json(buffer.shape);
        std::cout << ", \"offset\": " << buffer.offset
                  << ", \"count\": " << buffer.count
                  << ", \"weight_decay\": " << (buffer.weight_decay ? "true" : "false") << "}";
        if (i + 1 != layout.size()) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout << "    ]\n  }";
}

std::vector<StagePlan> build_gpt2_stage_plan(const Config& cfg) {
    std::vector<StagePlan> stages;
    const std::int64_t tokens = static_cast<std::int64_t>(cfg.batch_size) * cfg.seq_len;
    constexpr std::int64_t dim = 768;
    constexpr std::int64_t padded_vocab = 50304;
    const std::int64_t hidden = tokens * dim;
    const std::int64_t qkv = tokens * dim * 3;
    const std::int64_t mlp_hidden = tokens * dim * 4;
    const std::int64_t logits = tokens * padded_vocab;
    const std::int64_t parameters = layout_count(build_gpt2_parameter_layout(cfg));

    auto add = [&](std::string name, std::string phase, std::string kernel_abi, std::int64_t elements) {
        StagePlan stage;
        stage.name = std::move(name);
        stage.phase = std::move(phase);
        stage.kernel_abi = std::move(kernel_abi);
        stage.elements = elements;
        stages.push_back(std::move(stage));
    };

    add("wte.forward", "forward", "nfn_native_tile_token_embedding_float32", hidden);
    add("wpe.forward", "forward", "nfn_native_tile_absolute_position_embedding_float32", hidden);
    add("embedding_residual_add.forward", "forward", "nfn_native_tile_scaled_residual_add_float32", hidden);
    const bool packed_qkv_attention_enabled = packed_qkv_attention_default_enabled();
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        const std::string prefix = "h." + std::to_string(layer) + ".";
        add(prefix + "ln_1.forward", "forward", "nfn_native_tile_layer_norm_float32", hidden);
        if (packed_qkv_attention_enabled) {
            add(prefix + "attn.c_attn.forward", "forward", "nfn_native_tile_linear_bf16_output_float32(no_bias)", qkv);
            add(prefix + "attn.qkv.bias_inplace_bf16", "forward", "nfn_native_tile_bf16_bits_add_bias_inplace_float32", qkv);
            add(prefix + "attn.sdpa.forward", "forward", "nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32", hidden);
        } else {
            add(prefix + "attn.c_attn.forward", "forward", "nfn_native_tile_linear_float32(no_bias)", qkv);
            add(prefix + "attn.qkv.bias_split_to_heads", "forward", "nfn_native_tile_split_qkv_to_heads_add_bias_float32", qkv);
            add(prefix + "attn.sdpa.forward", "forward", "nfn_native_tile_scaled_dot_product_attention_float32", hidden);
            add(prefix + "attn.heads.merge", "forward", "nfn_native_tile_merge_heads_float32", hidden);
        }
        add(prefix + "attn.c_proj.forward", "forward", "nfn_native_tile_linear_float32(no_bias)", hidden);
        add(prefix + "attn.bias_residual_add.forward", "forward", "nfn_native_tile_linear_bias_residual_add_float32", hidden);
        add(prefix + "ln_2.forward", "forward", "nfn_native_tile_layer_norm_float32", hidden);
        add(prefix + "mlp.c_fc.forward", "forward", "nfn_native_tile_linear_float32(no_bias)", mlp_hidden);
        add(prefix + "mlp.bias_gelu.forward", "forward", "nfn_native_tile_gelu_add_bias_float32", mlp_hidden);
        add(prefix + "mlp.c_proj.forward", "forward", "nfn_native_tile_linear_float32(no_bias)", hidden);
        add(prefix + "mlp.bias_residual_add.forward", "forward", "nfn_native_tile_linear_bias_residual_add_float32", hidden);
    }
    add("ln_f.forward", "forward", "nfn_native_tile_layer_norm_float32", hidden);
    add("lm_head.forward_tied", "forward", "nfn_native_tile_linear_float32", logits);
    add("token_cross_entropy.forward", "forward", "nfn_native_tile_token_cross_entropy_partials_float32", tokens);
    add("token_cross_entropy.backward", "backward", "nfn_native_tile_token_cross_entropy_backward_with_workspace_float32", logits);
    add("lm_head.backward_input", "backward", "nfn_native_tile_linear_backward_input_float32", hidden);
    add("lm_head.backward_weight_tied", "backward", "nfn_native_tile_linear_backward_weight_accumulate_float32", padded_vocab * dim);
    for (int layer = cfg.num_layers - 1; layer >= 0; --layer) {
        const std::string prefix = "h." + std::to_string(layer) + ".";
        add(prefix + "mlp.c_proj.backward", "backward", "linear input/weight/weight-accumulate/bias/bias-accumulate backward native ABI", hidden + mlp_hidden);
        add(prefix + "mlp.gelu.backward", "backward", "nfn_native_tile_gelu_backward_float32", mlp_hidden);
        add(prefix + "mlp.c_fc.backward", "backward", "linear input/weight/weight-accumulate/bias/bias-accumulate backward native ABI", hidden + mlp_hidden);
        add(prefix + "ln_2.backward", "backward", "nfn_native_tile_layer_norm_backward_input_float32", hidden);
        add(prefix + "attn.c_proj.backward", "backward", "linear input/weight/weight-accumulate/bias/bias-accumulate backward native ABI", hidden);
        add(
            prefix + "attn.sdpa.backward_to_qkv_from_merged_grad",
            "backward",
            packed_qkv_attention_enabled
                ? "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_merged_grad_float32"
                : "nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_merged_grad_float32",
            qkv);
        add(prefix + "attn.c_attn.backward", "backward", "linear input/weight/weight-accumulate/bias/bias-accumulate backward native ABI", hidden + qkv);
        add(prefix + "ln_1.backward", "backward", "nfn_native_tile_layer_norm_backward_input_float32", hidden);
    }
    add("wpe.backward", "backward", "nfn_native_tile_absolute_position_embedding_backward_float32", hidden);
    add("wpe.backward_accumulate", "backward", "nfn_native_tile_absolute_position_embedding_backward_accumulate_float32", hidden);
    add("wte.backward", "backward", "nfn_native_tile_token_embedding_backward_weight_float32", hidden);
    add("gradient_zero", "optimizer", "nfn_native_tile_fill_float32", parameters);
    add("gradient_clip", "optimizer", "nfn_native_tile_global_norm_clip_scale_float32", 1);
    add("gradient_scale", "optimizer", "nfn_native_tile_scale_inplace_by_device_float32", parameters);
    add("adamw_step", "optimizer", "nfn_native_tile_adamw_step_float32", parameters);
    return stages;
}

void print_stage_plan_json(const std::vector<StagePlan>& stages, const std::string& status) {
    std::int64_t forward = 0;
    std::int64_t backward = 0;
    std::int64_t optimizer = 0;
    std::int64_t max_elements = 0;
    for (const StagePlan& stage : stages) {
        if (stage.phase == "forward") {
            ++forward;
        } else if (stage.phase == "backward") {
            ++backward;
        } else if (stage.phase == "optimizer") {
            ++optimizer;
        }
        if (stage.elements > max_elements) {
            max_elements = stage.elements;
        }
    }
    std::cout
        << "{\n"
        << "    \"status\": \"" << json_escape(status) << "\",\n"
        << "    \"stage_count\": " << stages.size() << ",\n"
        << "    \"forward_stage_count\": " << forward << ",\n"
        << "    \"backward_stage_count\": " << backward << ",\n"
        << "    \"optimizer_stage_count\": " << optimizer << ",\n"
        << "    \"max_stage_elements\": " << max_elements << ",\n"
        << "    \"stages\": [\n";
    for (std::size_t i = 0; i < stages.size(); ++i) {
        const StagePlan& stage = stages[i];
        std::cout << "      {\"name\": \"" << json_escape(stage.name)
                  << "\", \"phase\": \"" << json_escape(stage.phase)
                  << "\", \"kernel_abi\": \"" << json_escape(stage.kernel_abi)
                  << "\", \"elements\": " << stage.elements << "}";
        if (i + 1 != stages.size()) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout << "    ]\n  }";
}

std::string default_tile_ops_lib(const char* program) {
    std::string env_value = env_or_empty("NFN_NATIVE_TRAIN_TILE_OPS_LIB");
    if (!env_value.empty()) {
        return env_value;
    }
    fs::path exe_path(program);
    if (exe_path.has_parent_path()) {
        fs::path sibling = exe_path.parent_path() / "libnfn_native_train_tile_ops.so";
        if (fs::exists(sibling)) {
            return sibling.string();
        }
    }
    fs::path build_path = fs::current_path() / "build" / "libnfn_native_train_tile_ops.so";
    if (fs::exists(build_path)) {
        return build_path.string();
    }
    return "libnfn_native_train_tile_ops.so";
}

bool print_tile_plan(
    const Config& cfg,
    const neuralfn::native_train::TokenShardDataset& dataset,
    const char* program,
    bool include_symbol_check) {
    const std::int64_t tokens = static_cast<std::int64_t>(cfg.batch_size) * cfg.seq_len;
    const std::int64_t hidden = tokens * 768;
    const std::int64_t requested_lm_head_chunk_rows =
        cfg.lm_head_row_chunk_size > 0 ? cfg.lm_head_row_chunk_size : 1;
    const std::int64_t lm_head_chunk_rows =
        tokens < requested_lm_head_chunk_rows ? tokens : requested_lm_head_chunk_rows;
    constexpr std::int64_t public_vocab = 50257;
    constexpr std::int64_t padded_vocab = 50304;
    const std::int64_t logits = lm_head_chunk_rows * padded_vocab;
    const std::int64_t attention_row_count = tokens * 12;
    const std::int64_t attention_scalar_output_count = hidden;
    const bool packed_qkv_attention_enabled = packed_qkv_attention_default_enabled();
    const bool bf16_qkv_grad_handoff_enabled =
        packed_qkv_attention_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_BF16_QKV_GRAD_HANDOFF",
                              "NFN_NATIVE_GPT2_BF16_QKV_GRAD_HANDOFF"}),
            true);
    const bool bf16_attention_grad_out_handoff_enabled =
        packed_qkv_attention_enabled &&
        bf16_qkv_grad_handoff_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT",
                              "NFN_NATIVE_GPT2_BF16_ATTENTION_GRAD_OUT"}),
            false);
    const bool ln1_bf16_qkv_forward_enabled =
        packed_qkv_attention_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_LN1_BF16_QKV_FORWARD",
                              "NFN_NATIVE_GPT2_LN1_BF16_QKV_FORWARD"}),
            true);
    const bool direct_bf16_qkv_grad_scratch_enabled =
        bf16_qkv_grad_handoff_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_DIRECT_BF16_QKV_GRAD_SCRATCH",
                              "NFN_NATIVE_GPT2_DIRECT_BF16_QKV_GRAD_SCRATCH"}),
            true);
    const bool bf16_qkv_dweight_enabled =
        direct_bf16_qkv_grad_scratch_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_BF16_QKV_DWEIGHT",
                              "NFN_NATIVE_GPT2_BF16_QKV_DWEIGHT"}),
            true);
    const bool fuse_qkv_bias_tk_gemm_enabled =
        packed_qkv_attention_enabled &&
            env_flag_enabled_or_default(
                env_or_empty_any({"NFN_NATIVE_GPT_FUSE_QKV_BIAS_TK_GEMM",
                                  "NFN_NATIVE_GPT2_FUSE_QKV_BIAS_TK_GEMM"}),
                true);
    const std::int64_t qkv_activation_elements = hidden * 3;
    const bool fuse_attention_residual_ln2_enabled = fuse_attention_residual_ln2_default_enabled();
    const bool bf16_projection_residual_enabled =
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_BF16_PROJECTION_RESIDUAL",
                              "NFN_NATIVE_GPT2_BF16_PROJECTION_RESIDUAL"}),
            true);
    const std::string store_packed_attention_activations_env =
        env_or_empty_any({"NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_ACTIVATIONS",
                          "NFN_NATIVE_GPT2_STORE_PACKED_ATTENTION_ACTIVATIONS"});
    const bool store_packed_attention_activations_enabled =
        packed_qkv_attention_enabled &&
        env_flag_enabled_or_default(store_packed_attention_activations_env, true);
    const bool store_packed_attention_lse_enabled =
        store_packed_attention_activations_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE",
                              "NFN_NATIVE_GPT2_STORE_PACKED_ATTENTION_LSE"}),
            true);
    const bool store_packed_attention_ln1_stats_enabled =
        store_packed_attention_activations_enabled &&
        ln1_bf16_qkv_forward_enabled &&
        bf16_qkv_dweight_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_STATS",
                              "NFN_NATIVE_GPT2_STORE_PACKED_ATTENTION_LN1_STATS"}),
            true);
    constexpr std::int64_t activation_tape_count = 1;
    const std::int64_t packed_qkv_attention_bf16_elements =
        packed_qkv_attention_enabled ? (tokens * 768 * 4 * activation_tape_count) : 0;
    const std::int64_t packed_qkv_attention_bf16_bytes =
        packed_qkv_attention_bf16_elements * static_cast<std::int64_t>(sizeof(std::uint16_t));
    const std::int64_t attention_grad_out_bf16_elements =
        bf16_attention_grad_out_handoff_enabled ? hidden : 0;
    const std::int64_t attention_grad_out_bf16_bytes =
        attention_grad_out_bf16_elements * static_cast<std::int64_t>(sizeof(std::uint16_t));
    const std::int64_t projection_bf16_scratch_elements =
        bf16_projection_residual_enabled ? (hidden * activation_tape_count) : 0;
    const std::int64_t projection_bf16_scratch_bytes =
        projection_bf16_scratch_elements * static_cast<std::int64_t>(sizeof(std::uint16_t));
    const bool packed_qkv_float_attention_tape_elided = packed_qkv_attention_enabled;
    const std::int64_t packed_qkv_float_attention_tape_elements_elided =
        packed_qkv_float_attention_tape_elided ? (tokens * 768 * 8) : 0;
    const std::int64_t stored_packed_attention_block_count =
        store_packed_attention_activations_enabled && cfg.num_layers > 0
            ? std::min<std::int64_t>(
                  cfg.num_layers,
                  env_nonnegative_i64_or({"NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS",
                                          "NFN_NATIVE_GPT2_STORE_PACKED_ATTENTION_BLOCKS"},
                                         kDefaultStoredPackedAttentionBlocks))
            : 0;
    const std::int64_t stored_packed_attention_bf16_elements =
        stored_packed_attention_block_count * tokens * 768 * 4;
    const std::int64_t stored_packed_attention_bf16_bytes =
        stored_packed_attention_bf16_elements * static_cast<std::int64_t>(sizeof(std::uint16_t));
    const std::int64_t stored_packed_attention_ln1_stats_block_count =
        store_packed_attention_ln1_stats_enabled && stored_packed_attention_block_count > 0
            ? std::min<std::int64_t>(
                  stored_packed_attention_block_count,
                  std::max<std::int64_t>(cfg.num_layers - 1, 0))
            : 0;
    const std::int64_t stored_packed_attention_ln1_stats_elements =
        stored_packed_attention_ln1_stats_block_count * tokens * 2;
    const std::int64_t stored_packed_attention_ln1_stats_bytes =
        stored_packed_attention_ln1_stats_elements * static_cast<std::int64_t>(sizeof(float));
    const std::int64_t stored_packed_attention_lse_elements =
        store_packed_attention_lse_enabled ? stored_packed_attention_block_count * attention_row_count : 0;
    const std::int64_t stored_packed_attention_lse_bytes =
        stored_packed_attention_lse_elements * static_cast<std::int64_t>(sizeof(float));
    const std::string tile_ops = cfg.tile_ops_lib.empty() ? default_tile_ops_lib(program) : cfg.tile_ops_lib;
    std::vector<std::string> symbols = required_tile_symbols();
    const std::vector<BufferPlan> parameter_layout = build_gpt2_parameter_layout(cfg);
    const std::vector<StagePlan> stage_plan = build_gpt2_stage_plan(cfg);
    const bool native_runnable = selected_graph_is_native_runnable(cfg);
    const bool shipped_template = selected_template_is_shipped(cfg);
    const std::string support_status = selected_graph_support_status(cfg);
    const std::string plan_status = native_runnable ? "native-transformer-lm-ready" : support_status;
    const std::string stage_status = native_runnable ? "ready" : "requires-selected-graph-wiring";

    bool loaded = false;
    bool all_symbols = false;
    std::string error;
    std::vector<bool> found(symbols.size(), false);
    if (include_symbol_check) {
        void* handle = dlopen(tile_ops.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            const char* dl_error = dlerror();
            error = dl_error == nullptr ? "dlopen failed" : dl_error;
        } else {
            loaded = true;
            all_symbols = true;
            for (std::size_t i = 0; i < symbols.size(); ++i) {
                found[i] = dlsym(handle, symbols[i].c_str()) != nullptr;
                all_symbols = all_symbols && found[i];
            }
            dlclose(handle);
        }
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(cfg.model_family) << "\",\n"
        << "  \"backend\": \"tile-cuda\",\n"
        << "  \"status\": \"" << json_escape(plan_status) << "\",\n"
        << "  \"template_name\": \"" << json_escape(normalize_template_name(cfg.template_name)) << "\",\n"
        << "  \"resolved_native_template_name\": \"" << json_escape(resolved_native_template_name(cfg.template_name)) << "\",\n"
        << "  \"graph_file\": \"" << json_escape(cfg.graph_file) << "\",\n"
        << "  \"architecture_source\": \"" << json_escape(selected_architecture_source(cfg)) << "\",\n"
        << "  \"architecture_contract\": \"" << json_escape(dense_gpt_architecture_contract(cfg)) << "\",\n"
        << "  \"model_family_context_policy\": \"" << json_escape(model_family_context_policy(cfg)) << "\",\n"
        << "  \"native_cuda_activation\": \"" << json_escape(cfg.activation) << "\",\n"
        << "  \"template_known\": " << (shipped_template ? "true" : "false") << ",\n"
        << "  \"selected_graph_support_status\": \"" << json_escape(support_status) << "\",\n"
        << "  \"selected_graph_native_runnable\": " << (native_runnable ? "true" : "false") << ",\n"
        << "  \"shipped_template_catalog_count\": " << shipped_gpt_template_presets().size() << ",\n"
        << "  \"shipped_template_catalog\": ";
    print_string_array_json(shipped_gpt_template_presets(), "    ");
    std::cout
        << ",\n"
        << "  \"dataset_alias\": \"" << json_escape(cfg.dataset_alias) << "\",\n"
        << "  \"token_shards_resolved\": " << (!dataset.train_shards.empty() ? "true" : "false") << ",\n"
        << "  \"dataset_path\": \"" << json_escape(dataset.dataset_path.string()) << "\",\n"
        << "  \"train_shard\": \"" << json_escape(dataset.train_shards.empty() ? std::string() : dataset.train_shards[0].path.string()) << "\",\n"
        << "  \"val_shard\": \"" << json_escape(dataset.val_shards.empty() ? std::string() : dataset.val_shards[0].path.string()) << "\",\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_ops) << "\",\n"
        << "  \"shape\": {\"num_layers\": " << cfg.num_layers
        << ", \"model_dim\": 768, \"num_heads\": 12, \"vocab_size\": " << public_vocab
        << ", \"padded_vocab_size\": " << padded_vocab << ", \"seq_len\": "
        << cfg.seq_len << ", \"batch_size\": " << cfg.batch_size << "},\n"
        << "  \"schedule\": {\"max_steps\": " << cfg.max_steps
        << ", \"train_batch_tokens\": " << cfg.train_batch_tokens
        << ", \"eval_every_steps\": " << cfg.eval_every_steps
        << ", \"warmup_steps\": " << cfg.warmup_steps << "},\n"
        << "  \"checkpoint_export_enabled\": " << (cfg.write_checkpoint ? "true" : "false") << ",\n"
        << "  \"estimated_stage_elements\": {\"hidden\": " << hidden
        << ", \"logits\": " << logits
        << ", \"lm_head_row_chunk_size\": " << lm_head_chunk_rows << "},\n"
        << "  \"attention_forward_strategy\": \""
        << (packed_qkv_attention_enabled ? "tk-sm120-packed-qkv-bf16-flashattention" : "row-vector-tile-score-reuse")
        << "\",\n"
        << "  \"attention_forward_row_count\": " << attention_row_count << ",\n"
        << "  \"attention_forward_scalar_output_count\": " << attention_scalar_output_count << ",\n"
        << "  \"attention_forward_score_reuse_value_dim\": " << kAttentionForwardValueReuse << ",\n"
        << "  \"attention_forward_scalar_cta_elision_factor\": " << kAttentionForwardValueReuse << ",\n"
        << "  \"attention_forward_value_chunk_size\": " << kAttentionForwardValueReuse << ",\n"
        << "  \"attention_forward_scalar_launch_fallback_enabled\": true,\n"
        << "  \"attention_forward_row_launch_auto_disable_enabled\": true,\n"
        << "  \"packed_qkv_attention_enabled\": " << (packed_qkv_attention_enabled ? "true" : "false") << ",\n"
        << "  \"packed_qkv_attention_bf16_elements\": " << packed_qkv_attention_bf16_elements << ",\n"
        << "  \"packed_qkv_attention_bf16_bytes\": " << packed_qkv_attention_bf16_bytes << ",\n"
        << "  \"packed_qkv_float_attention_tape_elided\": "
        << (packed_qkv_float_attention_tape_elided ? "true" : "false") << ",\n"
        << "  \"packed_qkv_float_attention_tape_elements_elided\": "
        << packed_qkv_float_attention_tape_elements_elided << ",\n"
        << "  \"packed_qkv_float_attention_tape_bytes_elided\": "
        << (packed_qkv_float_attention_tape_elements_elided * static_cast<std::int64_t>(sizeof(float))) << ",\n"
        << "  \"qkv_forward_layout_strategy\": \""
        << (packed_qkv_attention_enabled ? "packed-qkv-bf16-no-split" : "fused-split-to-heads")
        << "\",\n"
        << "  \"qkv_forward_ln1_bf16_enabled\": "
        << (ln1_bf16_qkv_forward_enabled ? "true" : "false") << ",\n"
        << "  \"qkv_forward_layout_kernel_launches_per_block\": " << (packed_qkv_attention_enabled ? 0 : 1) << ",\n"
        << "  \"qkv_forward_layout_legacy_launches_per_block\": 4,\n"
        << "  \"qkv_forward_layout_launches_elided_per_block\": " << (packed_qkv_attention_enabled ? 4 : 3) << ",\n"
        << "  \"qkv_bias_layout_strategy\": \""
        << (packed_qkv_attention_enabled
                ? (fuse_qkv_bias_tk_gemm_enabled
                       ? "packed-qkv-bf16-bias-fused-tk-gemm"
                       : "packed-qkv-bf16-bias-inplace")
                : "fused-qkv-bias-split-to-heads")
        << "\",\n"
        << "  \"qkv_bias_fused_tk_gemm_enabled\": " << (fuse_qkv_bias_tk_gemm_enabled ? "true" : "false") << ",\n"
        << "  \"qkv_bias_layout_kernel_launches_per_block\": "
        << (packed_qkv_attention_enabled && fuse_qkv_bias_tk_gemm_enabled ? 0 : 1) << ",\n"
        << "  \"qkv_bias_layout_legacy_launches_per_block\": 2,\n"
        << "  \"qkv_bias_layout_launches_elided_per_block\": "
        << (packed_qkv_attention_enabled && fuse_qkv_bias_tk_gemm_enabled ? 2 : 1) << ",\n"
        << "  \"qkv_backward_layout_strategy\": \""
        << (packed_qkv_attention_enabled
                ? (bf16_qkv_grad_handoff_enabled ? "packed-qkv-bf16-gradient-handoff" : "packed-qkv-bf16-gradient-unpack")
                : "fused-heads-to-qkv")
        << "\",\n"
        << "  \"qkv_backward_layout_kernel_launches_per_block\": 1,\n"
        << "  \"qkv_backward_layout_legacy_launches_per_block\": 4,\n"
        << "  \"qkv_backward_layout_launches_elided_per_block\": 3,\n"
        << "  \"attention_backward_bf16_qkv_grad_handoff_enabled\": "
        << (bf16_qkv_grad_handoff_enabled ? "true" : "false") << ",\n"
        << "  \"attention_backward_bf16_grad_out_handoff_enabled\": "
        << (bf16_attention_grad_out_handoff_enabled ? "true" : "false") << ",\n"
        << "  \"attention_backward_grad_out_dtype\": \""
        << (bf16_attention_grad_out_handoff_enabled ? "bf16" : "float32") << "\",\n"
        << "  \"attention_backward_bf16_grad_out_scratch_elements\": "
        << attention_grad_out_bf16_elements << ",\n"
        << "  \"attention_backward_bf16_grad_out_scratch_bytes\": "
        << attention_grad_out_bf16_bytes << ",\n"
        << "  \"attention_backward_direct_bf16_qkv_grad_scratch_enabled\": "
        << (direct_bf16_qkv_grad_scratch_enabled ? "true" : "false") << ",\n"
        << "  \"attention_backward_direct_bf16_qkv_grad_scratch_elements\": "
        << (direct_bf16_qkv_grad_scratch_enabled ? qkv_activation_elements : 0) << ",\n"
        << "  \"attention_backward_qkv_bridge_strategy\": \""
        << (packed_qkv_attention_enabled
                ? (bf16_attention_grad_out_handoff_enabled
                       ? "tk-sm120-packed-qkv-bf16-grad-out-direct-bf16-qkv-handoff"
                   : bf16_qkv_grad_handoff_enabled
                       ? (direct_bf16_qkv_grad_scratch_enabled
                              ? "tk-sm120-packed-qkv-direct-bf16-grad-scratch-handoff"
                              : "tk-sm120-packed-qkv-packed-bf16-grad-handoff")
                       : "tk-sm120-packed-qkv-packed-grad-bridge")
                : "fused-bf16-heads-to-row-qkv")
        << "\",\n"
        << "  \"attention_backward_qkv_bridge_kernel_launches_per_block\": "
        << (packed_qkv_attention_enabled ? 2 : 1) << ",\n"
        << "  \"attention_backward_qkv_bridge_legacy_launches_per_block\": 4,\n"
        << "  \"attention_backward_qkv_bridge_launches_elided_per_block\": 3,\n"
        << "  \"bf16_projection_residual_enabled\": "
        << (bf16_projection_residual_enabled ? "true" : "false") << ",\n"
        << "  \"attention_projection_input_strategy\": \""
        << (bf16_projection_residual_enabled
                ? (packed_qkv_attention_enabled
                       ? "packed-o-bf16-direct-gemm-bf16-residual-consumer"
                       : "float32-attention-output-bf16-gemm-bf16-residual-consumer")
                : (packed_qkv_attention_enabled ? "packed-o-bf16-direct-gemm" : "float32-attention-output-bf16-gemm"))
        << "\",\n"
        << "  \"attention_packed_output_unpack_strategy\": \""
        << (packed_qkv_attention_enabled ? "elided-direct-bf16-projection" : "not-packed")
        << "\",\n"
        << "  \"mlp_fc_bias_gelu_strategy\": \"fused-bias-preactivation-gelu\",\n"
        << "  \"mlp_fc_bias_gelu_kernel_launches_per_block\": 1,\n"
        << "  \"mlp_fc_bias_gelu_legacy_launches_per_block\": 2,\n"
        << "  \"mlp_fc_bias_gelu_launches_elided_per_block\": 1,\n"
        << "  \"mlp_proj_forward_activation_strategy\": \""
        << (bf16_projection_residual_enabled
                ? "fused-gelu-bf16-act-direct-bf16-output-gemm"
                : "fused-gelu-bf16-act-direct-gemm")
        << "\",\n"
        << "  \"mlp_forward_act_bf16_elements\": 0,\n"
        << "  \"mlp_forward_act_bf16_bytes\": 0,\n"
        << "  \"projection_bf16_scratch_elements\": " << projection_bf16_scratch_elements << ",\n"
        << "  \"projection_bf16_scratch_bytes\": " << projection_bf16_scratch_bytes << ",\n"
        << "  \"projection_bias_residual_strategy\": \""
        << (bf16_projection_residual_enabled
                ? "fused-bf16-linear-bias-residual-add"
                : "fused-linear-bias-residual-add")
        << "\",\n"
        << "  \"attention_residual_ln2_strategy\": \""
        << (fuse_attention_residual_ln2_enabled
                ? (bf16_projection_residual_enabled
                       ? "fused-bf16-linear-bias-residual-layernorm"
                       : "fused-linear-bias-residual-layernorm")
                : "disabled")
        << "\",\n"
        << "  \"attention_residual_ln2_kernel_launches_per_block\": "
        << (fuse_attention_residual_ln2_enabled ? 1 : 0) << ",\n"
        << "  \"attention_residual_ln2_legacy_launches_per_block\": 2,\n"
        << "  \"attention_residual_ln2_launches_elided_per_block\": "
        << (fuse_attention_residual_ln2_enabled ? 1 : 0) << ",\n"
        << "  \"projection_bias_residual_kernel_launches_per_block\": 2,\n"
        << "  \"projection_bias_residual_legacy_launches_per_block\": 4,\n"
        << "  \"projection_bias_residual_launches_elided_per_block\": 2,\n"
        << "  \"attention_backward_grad_layout_strategy\": \"merged-grad-out-direct\",\n"
        << "  \"attention_backward_grad_layout_kernel_launches_per_block\": 0,\n"
        << "  \"attention_backward_grad_layout_legacy_launches_per_block\": 1,\n"
        << "  \"attention_backward_grad_layout_launches_elided_per_block\": 1,\n"
        << "  \"attention_backward_strategy\": \""
        << (packed_qkv_attention_enabled
                ? (stored_packed_attention_block_count > 0
                       ? (bf16_qkv_grad_handoff_enabled
                              ? (direct_bf16_qkv_grad_scratch_enabled
                                     ? "tk-sm120-packed-qkv-bf16-saved-activation-backward-direct-bf16-grad-scratch-handoff"
                                     : "tk-sm120-packed-qkv-bf16-saved-activation-backward-bf16-grad-handoff")
                              : "tk-sm120-packed-qkv-bf16-saved-activation-backward-bridge")
                       : (bf16_qkv_grad_handoff_enabled
                              ? (direct_bf16_qkv_grad_scratch_enabled
                                     ? "tk-sm120-packed-qkv-bf16-backward-direct-bf16-grad-scratch-handoff"
                                     : "tk-sm120-packed-qkv-bf16-backward-bf16-grad-handoff")
                              : "tk-sm120-packed-qkv-bf16-backward-bridge"))
                : "tk-sm120-bf16-reuse-forward-workspace-bridge")
        << "\",\n"
        << "  \"attention_backward_reuses_forward_workspace\": true,\n"
        << "  \"attention_backward_uses_saved_forward_workspace\": "
        << (stored_packed_attention_block_count > 0 ? "true" : "false") << ",\n"
        << "  \"attention_activation_storage_strategy\": \"disabled\",\n"
        << "  \"packed_attention_activation_storage_strategy\": \""
        << (stored_packed_attention_block_count > 0 ? "packed-qkv-o-bf16-forward-store-direct-backward" : "disabled")
        << "\",\n"
        << "  \"stored_packed_attention_activation_blocks\": " << stored_packed_attention_block_count << ",\n"
        << "  \"stored_packed_attention_bf16_elements\": " << stored_packed_attention_bf16_elements << ",\n"
        << "  \"stored_packed_attention_bf16_bytes\": " << stored_packed_attention_bf16_bytes << ",\n"
        << "  \"stored_packed_attention_ln1_stats_enabled\": "
        << (store_packed_attention_ln1_stats_enabled ? "true" : "false") << ",\n"
        << "  \"stored_packed_attention_ln1_stats_blocks\": " << stored_packed_attention_ln1_stats_block_count << ",\n"
        << "  \"stored_packed_attention_ln1_stats_elements\": " << stored_packed_attention_ln1_stats_elements << ",\n"
        << "  \"stored_packed_attention_ln1_stats_bytes\": " << stored_packed_attention_ln1_stats_bytes << ",\n"
        << "  \"stored_packed_attention_lse_elements\": " << stored_packed_attention_lse_elements << ",\n"
        << "  \"stored_packed_attention_lse_bytes\": " << stored_packed_attention_lse_bytes << ",\n"
        << "  \"stored_packed_attention_lse_enabled\": "
        << (store_packed_attention_lse_enabled ? "true" : "false") << ",\n"
        << "  \"stored_packed_attention_store_blocks\": 0,\n"
        << "  \"stored_packed_attention_restore_blocks\": 0,\n"
        << "  \"stored_packed_attention_backward_kernel_launches\": 0,\n"
        << "  \"stored_packed_attention_backward_consumer_strategy\": \""
        << (stored_packed_attention_block_count > 0
                ? (store_packed_attention_lse_enabled
                       ? "saved-packed-qkv-o-lse-bf16-backward-to-qkv"
                       : "saved-packed-qkv-o-workspace-lse-bf16-backward-to-qkv")
                : "disabled")
        << "\",\n"
        << "  \"attention_backward_recompute_forward_elided_per_block\": 1,\n"
        << "  \"attention_backward_row_count\": " << attention_row_count << ",\n"
        << "  \"attention_backward_scalar_output_count\": " << (attention_scalar_output_count * 3) << ",\n"
        << "  \"attention_backward_score_reuse_dim\": " << kAttentionBackwardDimReuse << ",\n"
        << "  \"attention_backward_scalar_cta_elision_factor\": " << (kAttentionBackwardDimReuse * 3) << ",\n"
        << "  \"parameter_layout\": ";
    print_parameter_layout_json(parameter_layout);
    std::cout
        << ",\n"
        << "  \"training_step_plan\": ";
    print_stage_plan_json(stage_plan, stage_status);
    std::cout
        << ",\n"
        << "  \"available_native_kernels\": [\n";
    for (std::size_t i = 0; i < symbols.size(); ++i) {
        std::cout << "    \"" << json_escape(symbols[i]) << "\"";
        if (i + 1 != symbols.size()) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout
        << "  ],\n"
        << "  \"required_native_work\": [\n";
    if (!native_runnable) {
        if (!cfg.graph_file.empty() || shipped_template) {
            std::cout
                << "    \"compile the selected GPT template or custom graph into a native C++ Tile trainer plan\",\n"
                << "    \"map every selected graph node to raw Tile trainer ABI calls without Torch or graph-editor tensor flow\",\n"
                << "    \"emit native checkpoints and inference metadata for the selected graph shape\"\n";
        } else {
            std::cout
                << "    \"choose a template from shipped_template_catalog or pass --graph-file for an explicit custom graph\"\n";
        }
    }
    std::cout
        << "  ],\n"
        << "  \"remaining_validation\": [\n"
        << "    \"live-validate SM120 throughput against /mnt/disk2/dev/open-source/llm.kittens/train-sm120.sh on a GPU-visible process\"\n"
        << "  ]";
    if (include_symbol_check) {
        std::cout << ",\n"
                  << "  \"tile_ops_check\": {\n"
                  << "    \"loaded\": " << (loaded ? "true" : "false") << ",\n"
                  << "    \"all_required_symbols_found\": " << (all_symbols ? "true" : "false") << ",\n"
                  << "    \"symbols\": [\n";
        for (std::size_t i = 0; i < symbols.size(); ++i) {
            std::cout << "      {\"name\": \"" << json_escape(symbols[i]) << "\", \"found\": "
                      << (found[i] ? "true" : "false") << "}";
            if (i + 1 != symbols.size()) {
                std::cout << ",";
            }
            std::cout << "\n";
        }
        std::cout << "    ]";
        if (!error.empty()) {
            std::cout << ",\n"
                      << "    \"error\": \"" << json_escape(error) << "\"\n";
        } else {
            std::cout << "\n";
        }
        std::cout << "  }";
    }
    std::cout << "\n}\n";
    return include_symbol_check ? (loaded && all_symbols) : false;
}

int print_selected_graph_unsupported_json(const Config& cfg, const neuralfn::native_train::TokenShardDataset& dataset) {
    const bool shipped_template = selected_template_is_shipped(cfg);
    const std::string support_status = selected_graph_support_status(cfg);
    const std::string status = support_status == "unknown-template" ? "unknown-template"
                                                                     : "selected-graph-native-trainer-missing";
    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(cfg.model_family) << "\",\n"
        << "  \"backend\": \"" << json_escape(cfg.backend) << "\",\n"
        << "  \"template_name\": \"" << json_escape(normalize_template_name(cfg.template_name)) << "\",\n"
        << "  \"resolved_native_template_name\": \"" << json_escape(resolved_native_template_name(cfg.template_name)) << "\",\n"
        << "  \"graph_file\": \"" << json_escape(cfg.graph_file) << "\",\n"
        << "  \"architecture_source\": \"" << json_escape(selected_architecture_source(cfg)) << "\",\n"
        << "  \"architecture_contract\": \"" << json_escape(dense_gpt_architecture_contract(cfg)) << "\",\n"
        << "  \"model_family_context_policy\": \"" << json_escape(model_family_context_policy(cfg)) << "\",\n"
        << "  \"native_cuda_activation\": \"" << json_escape(cfg.activation) << "\",\n"
        << "  \"template_known\": " << (shipped_template ? "true" : "false") << ",\n"
        << "  \"selected_graph_support_status\": \"" << json_escape(support_status) << "\",\n"
        << "  \"selected_graph_native_runnable\": false,\n"
        << "  \"status\": \"" << json_escape(status) << "\",\n"
        << "  \"shipped_template_catalog_count\": " << shipped_gpt_template_presets().size() << ",\n"
        << "  \"dataset_alias\": \"" << json_escape(cfg.dataset_alias) << "\",\n"
        << "  \"dataset_path\": \"" << json_escape(dataset.dataset_path.string()) << "\",\n"
        << "  \"required_native_work\": [\n";
    if (support_status == "unknown-template") {
        std::cout
            << "    \"choose a template from the shipped GPT template catalog or pass --graph-file for an explicit custom graph\"\n";
    } else {
        std::cout
            << "    \"compile the selected GPT template or custom graph into a native C++ Tile trainer plan\",\n"
            << "    \"map every selected graph node to raw Tile trainer ABI calls without Torch or graph-editor tensor flow\",\n"
            << "    \"emit native checkpoints and inference metadata for the selected graph shape\"\n";
    }
    std::cout
        << "  ]\n"
        << "}\n";
    return 2;
}

int print_tile_ops_smoke_json(const Config& cfg, const char* program) {
    constexpr std::int64_t kElements = 16;
    constexpr float kExpectedValue = 3.5f;
    constexpr int kCudaMemcpyDeviceToHost = 2;

    const std::string tile_lib_path = cfg.tile_ops_lib.empty() ? default_tile_ops_lib(program) : cfg.tile_ops_lib;
    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(cfg);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool kernel_loaded = false;
    bool passed = false;
    double max_abs_error = 0.0;
    std::string error;

    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);

    void* tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    void* cuda_handle = nullptr;
    FillFn fill = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed with CUDA error " << code;
        if (cuda_get_error_string != nullptr) {
            const char* message = cuda_get_error_string(code);
            if (message != nullptr) {
                out << ": " << message;
            }
        }
        return out.str();
    };

    if (tile_handle == nullptr) {
        error = dl_last_error("dlopen tile ops failed");
    } else {
        tile_loaded = true;
        fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
        kernel_loaded = fill != nullptr;
        if (fill == nullptr) {
            error = dl_last_error("dlsym nfn_native_tile_fill_float32 failed");
        }
    }

    if (error.empty()) {
        std::string runtime_error;
        for (const std::string& candidate : runtime_candidates) {
            cuda_lib_path = candidate;
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_runtime_loaded = true;
                break;
            }
            runtime_error = dl_last_error("dlopen CUDA runtime failed");
        }
        if (cuda_handle == nullptr) {
            error = runtime_error.empty() ? "could not load CUDA runtime" : runtime_error;
        }
    }

    if (error.empty()) {
        cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
        cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
        cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
        cuda_device_synchronize = load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
        cuda_get_error_string = load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
        if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr || cuda_device_synchronize == nullptr) {
            error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaDeviceSynchronize";
        }
    }

    float* device_values = nullptr;
    if (error.empty()) {
        const int status = cuda_malloc(reinterpret_cast<void**>(&device_values), sizeof(float) * kElements);
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc");
        }
    }
    if (error.empty()) {
        const int status = fill(device_values, kElements, kExpectedValue, nullptr);
        if (status != 0) {
            error = cuda_error(status, "nfn_native_tile_fill_float32");
        }
    }
    if (error.empty()) {
        const int status = cuda_device_synchronize();
        if (status != 0) {
            error = cuda_error(status, "cudaDeviceSynchronize");
        }
    }

    std::vector<float> host_values(static_cast<std::size_t>(kElements), 0.0f);
    if (error.empty()) {
        const int status = cuda_memcpy(host_values.data(), device_values, sizeof(float) * kElements, kCudaMemcpyDeviceToHost);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy device-to-host");
        }
    }
    if (error.empty()) {
        for (float value : host_values) {
            const double abs_error = std::fabs(static_cast<double>(value) - static_cast<double>(kExpectedValue));
            if (abs_error > max_abs_error) {
                max_abs_error = abs_error;
            }
        }
        passed = max_abs_error <= 1e-6;
        if (!passed) {
            std::ostringstream out;
            out << "fill smoke max_abs_error " << max_abs_error << " exceeded tolerance";
            error = out.str();
        }
    }

    if (device_values != nullptr && cuda_free != nullptr) {
        const int status = cuda_free(device_values);
        if (status != 0 && error.empty()) {
            error = cuda_error(status, "cudaFree");
        }
    }
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(cfg.model_family) << "\",\n"
        << "  \"backend\": \"tile-cuda\",\n"
        << "  \"smoke\": \"tile_ops_fill\",\n"
        << "  \"token_shards_resolved\": false,\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"kernel\": \"nfn_native_tile_fill_float32\",\n"
        << "  \"kernel_loaded\": " << (kernel_loaded ? "true" : "false") << ",\n"
        << "  \"elements\": " << kElements << ",\n"
        << "  \"expected_value\": " << kExpectedValue << ",\n"
        << "  \"max_abs_error\": " << max_abs_error << ",\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";

    return passed ? 0 : 2;
}

int print_optimizer_smoke_json(const Config& cfg, const char* program) {
    constexpr float kInitialParam = 1.0f;
    constexpr float kInitialGrad = 0.5f;
    constexpr float kInitialMoment = 0.0f;
    constexpr float kLearningRate = 0.1f;
    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.95f;
    constexpr float kEps = 1e-8f;
    constexpr float kWeightDecay = 0.1f;
    constexpr float kBiasCorrection1 = 1.0f - kBeta1;
    constexpr float kSqrtBiasCorrection2 = 0.22360679775f;
    constexpr int kCudaMemcpyDeviceToHost = 2;

    const std::vector<BufferPlan> layout = build_gpt2_parameter_layout(cfg);
    const std::int64_t total_parameters = layout_count(layout);
    std::int64_t decay_buffer_count = 0;
    std::int64_t no_decay_buffer_count = 0;
    std::int64_t decay_elements = 0;
    std::int64_t no_decay_elements = 0;
    for (const BufferPlan& buffer : layout) {
        if (buffer.weight_decay) {
            decay_buffer_count += 1;
            decay_elements += buffer.count;
        } else {
            no_decay_buffer_count += 1;
            no_decay_elements += buffer.count;
        }
    }

    const float expected_m = kBeta1 * kInitialMoment + (1.0f - kBeta1) * kInitialGrad;
    const float expected_v = kBeta2 * kInitialMoment + (1.0f - kBeta2) * kInitialGrad * kInitialGrad;
    auto expected_param_for_decay = [&](float weight_decay) {
        return kInitialParam * (1.0f - kLearningRate * weight_decay) -
               (kLearningRate / kBiasCorrection1) * expected_m /
                   (std::sqrt(expected_v) / kSqrtBiasCorrection2 + kEps);
    };

    const std::string tile_lib_path = cfg.tile_ops_lib.empty() ? default_tile_ops_lib(program) : cfg.tile_ops_lib;
    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(cfg);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool fill_loaded = false;
    bool adamw_loaded = false;
    bool passed = false;
    double max_param_abs_error = 0.0;
    double max_exp_avg_abs_error = 0.0;
    double max_exp_avg_sq_abs_error = 0.0;
    std::int64_t sample_count = 0;
    std::string error;

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
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);

    void* tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    void* cuda_handle = nullptr;
    FillFn fill = nullptr;
    AdamWFn adamw = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed with CUDA error " << code;
        if (cuda_get_error_string != nullptr) {
            const char* message = cuda_get_error_string(code);
            if (message != nullptr) {
                out << ": " << message;
            }
        }
        return out.str();
    };

    if (tile_handle == nullptr) {
        error = dl_last_error("dlopen tile ops failed");
    } else {
        tile_loaded = true;
        fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
        adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
        fill_loaded = fill != nullptr;
        adamw_loaded = adamw != nullptr;
        if (fill == nullptr || adamw == nullptr) {
            error = dl_last_error("dlsym fill/adamw failed");
        }
    }

    if (error.empty()) {
        std::string runtime_error;
        for (const std::string& candidate : runtime_candidates) {
            cuda_lib_path = candidate;
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_runtime_loaded = true;
                break;
            }
            runtime_error = dl_last_error("dlopen CUDA runtime failed");
        }
        if (cuda_handle == nullptr) {
            error = runtime_error.empty() ? "could not load CUDA runtime" : runtime_error;
        }
    }

    if (error.empty()) {
        cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
        cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
        cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
        cuda_device_synchronize = load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
        cuda_get_error_string = load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
        if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr || cuda_device_synchronize == nullptr) {
            error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaDeviceSynchronize";
        }
    }

    float* param = nullptr;
    float* grad = nullptr;
    float* exp_avg = nullptr;
    float* exp_avg_sq = nullptr;
    auto allocate = [&](float** ptr, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        const int status = cuda_malloc(
            reinterpret_cast<void**>(ptr),
            sizeof(float) * static_cast<std::size_t>(total_parameters));
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc " + name);
        }
    };
    allocate(&param, "param");
    allocate(&grad, "grad");
    allocate(&exp_avg, "exp_avg");
    allocate(&exp_avg_sq, "exp_avg_sq");

    auto fill_buffer = [&](float* ptr, float value, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        const int status = fill(ptr, total_parameters, value, nullptr);
        if (status != 0) {
            error = cuda_error(status, "fill " + name);
        }
    };
    fill_buffer(param, kInitialParam, "param");
    fill_buffer(grad, kInitialGrad, "grad");
    fill_buffer(exp_avg, kInitialMoment, "exp_avg");
    fill_buffer(exp_avg_sq, kInitialMoment, "exp_avg_sq");

    std::int64_t adamw_step_calls = 0;
    if (error.empty()) {
        for (const BufferPlan& buffer : layout) {
            const float buffer_weight_decay = buffer.weight_decay ? kWeightDecay : 0.0f;
            const int status = adamw(
                param + buffer.offset,
                grad + buffer.offset,
                exp_avg + buffer.offset,
                exp_avg_sq + buffer.offset,
                buffer.count,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                buffer_weight_decay,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr);
            if (status != 0) {
                error = cuda_error(status, "nfn_native_tile_adamw_step_float32 " + buffer.name);
                break;
            }
            adamw_step_calls += 1;
        }
    }

    if (error.empty()) {
        const int status = cuda_device_synchronize();
        if (status != 0) {
            error = cuda_error(status, "cudaDeviceSynchronize");
        }
    }

    auto copy_sample = [&](const float* device_ptr, std::int64_t offset, const std::string& name) -> float {
        float value = 0.0f;
        if (!error.empty()) {
            return value;
        }
        const int status = cuda_memcpy(
            &value,
            device_ptr + offset,
            sizeof(float),
            kCudaMemcpyDeviceToHost);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy sample " + name);
        }
        return value;
    };
    if (error.empty()) {
        for (const BufferPlan& buffer : layout) {
            const float expected_param = expected_param_for_decay(buffer.weight_decay ? kWeightDecay : 0.0f);
            const float sampled_param = copy_sample(param, buffer.offset, buffer.name + ".param");
            const float sampled_exp_avg = copy_sample(exp_avg, buffer.offset, buffer.name + ".exp_avg");
            const float sampled_exp_avg_sq = copy_sample(exp_avg_sq, buffer.offset, buffer.name + ".exp_avg_sq");
            if (!error.empty()) {
                break;
            }
            const double param_error = std::fabs(static_cast<double>(sampled_param) - expected_param);
            const double exp_avg_error = std::fabs(static_cast<double>(sampled_exp_avg) - expected_m);
            const double exp_avg_sq_error = std::fabs(static_cast<double>(sampled_exp_avg_sq) - expected_v);
            if (param_error > max_param_abs_error) {
                max_param_abs_error = param_error;
            }
            if (exp_avg_error > max_exp_avg_abs_error) {
                max_exp_avg_abs_error = exp_avg_error;
            }
            if (exp_avg_sq_error > max_exp_avg_sq_abs_error) {
                max_exp_avg_sq_abs_error = exp_avg_sq_error;
            }
            sample_count += 1;
        }
    }

    if (error.empty()) {
        passed = max_param_abs_error <= 1e-5 && max_exp_avg_abs_error <= 1e-6 && max_exp_avg_sq_abs_error <= 1e-6;
        if (!passed) {
            std::ostringstream out;
            out << "AdamW smoke exceeded tolerance: param=" << max_param_abs_error
                << " exp_avg=" << max_exp_avg_abs_error
                << " exp_avg_sq=" << max_exp_avg_sq_abs_error;
            error = out.str();
        }
    }

    if (param != nullptr && cuda_free != nullptr) {
        const int status = cuda_free(param);
        if (status != 0 && error.empty()) {
            error = cuda_error(status, "cudaFree param");
        }
    }
    if (grad != nullptr && cuda_free != nullptr) {
        const int status = cuda_free(grad);
        if (status != 0 && error.empty()) {
            error = cuda_error(status, "cudaFree grad");
        }
    }
    if (exp_avg != nullptr && cuda_free != nullptr) {
        const int status = cuda_free(exp_avg);
        if (status != 0 && error.empty()) {
            error = cuda_error(status, "cudaFree exp_avg");
        }
    }
    if (exp_avg_sq != nullptr && cuda_free != nullptr) {
        const int status = cuda_free(exp_avg_sq);
        if (status != 0 && error.empty()) {
            error = cuda_error(status, "cudaFree exp_avg_sq");
        }
    }
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(cfg.model_family) << "\",\n"
        << "  \"backend\": \"tile-cuda\",\n"
        << "  \"smoke\": \"optimizer_step\",\n"
        << "  \"token_shards_resolved\": false,\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"fill_kernel_loaded\": " << (fill_loaded ? "true" : "false") << ",\n"
        << "  \"optimizer_kernel\": \"nfn_native_tile_adamw_step_float32\",\n"
        << "  \"optimizer_kernel_loaded\": " << (adamw_loaded ? "true" : "false") << ",\n"
        << "  \"parameter_buffer_count\": " << layout.size() << ",\n"
        << "  \"total_parameters\": " << total_parameters << ",\n"
        << "  \"adamw_step_calls\": " << adamw_step_calls << ",\n"
        << "  \"decay_buffer_count\": " << decay_buffer_count << ",\n"
        << "  \"no_decay_buffer_count\": " << no_decay_buffer_count << ",\n"
        << "  \"decay_elements\": " << decay_elements << ",\n"
        << "  \"no_decay_elements\": " << no_decay_elements << ",\n"
        << "  \"sampled_buffer_count\": " << sample_count << ",\n"
        << "  \"expected_decay_param\": " << expected_param_for_decay(kWeightDecay) << ",\n"
        << "  \"expected_no_decay_param\": " << expected_param_for_decay(0.0f) << ",\n"
        << "  \"expected_exp_avg\": " << expected_m << ",\n"
        << "  \"expected_exp_avg_sq\": " << expected_v << ",\n"
        << "  \"max_param_abs_error\": " << max_param_abs_error << ",\n"
        << "  \"max_exp_avg_abs_error\": " << max_exp_avg_abs_error << ",\n"
        << "  \"max_exp_avg_sq_abs_error\": " << max_exp_avg_sq_abs_error << ",\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";

    return passed ? 0 : 2;
}

int print_lm_step_smoke_json(const Config& cfg, const char* program) {
    constexpr std::int64_t kRows = 2;
    constexpr std::int64_t kVocab = 50257;
    constexpr std::int64_t kPaddedVocab = 50304;
    constexpr std::int64_t kDim = 768;
    constexpr std::int64_t kWeightElements = kPaddedVocab * kDim;
    constexpr std::int64_t kActivationElements = kRows * kDim;
    constexpr std::int64_t kLogitElements = kRows * kPaddedVocab;
    constexpr float kInitialWeight = 0.01f;
    constexpr float kLearningRate = 0.1f;
    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.95f;
    constexpr float kEps = 1e-8f;
    constexpr float kWeightDecay = 0.1f;
    constexpr float kBiasCorrection1 = 1.0f - kBeta1;
    constexpr float kSqrtBiasCorrection2 = 0.22360679775f;
    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;
    const std::int64_t host_tokens[kRows] = {0, 3};
    const std::int64_t host_targets[kRows] = {1, 4};
    const double expected_loss = std::log(static_cast<double>(kPaddedVocab));
    const float inv_vocab = 1.0f / static_cast<float>(kPaddedVocab);

    auto expected_grad_for_token = [&](std::int64_t token) {
        const bool is_target = token == host_targets[0] || token == host_targets[1];
        return kInitialWeight * (is_target ? (inv_vocab - 0.5f) : inv_vocab);
    };
    auto expected_param_for_grad = [&](float grad) {
        const float next_m = (1.0f - kBeta1) * grad;
        const float next_v = (1.0f - kBeta2) * grad * grad;
        return kInitialWeight * (1.0f - kLearningRate * kWeightDecay) -
               (kLearningRate / kBiasCorrection1) * next_m /
                   (std::sqrt(next_v) / kSqrtBiasCorrection2 + kEps);
    };

    const std::string tile_lib_path = cfg.tile_ops_lib.empty() ? default_tile_ops_lib(program) : cfg.tile_ops_lib;
    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(cfg);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool passed = false;
    double loss_abs_error = 0.0;
    double max_grad_abs_error = 0.0;
    double max_weight_abs_error = 0.0;
    std::string error;

    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using TokenEmbeddingFn = int (*)(const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using TokenEmbeddingBackwardWeightFn = int (*)(
        const std::int64_t*, const float*, float*, std::int64_t, std::int64_t, void*);
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBackwardInputFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardBiasFn = int (*)(const float*, float*, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyPartialsFn = int (*)(
        const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyBackwardWorkspaceFn = int (*)(
        const float*,
        const std::int64_t*,
        float*,
        float*,
        float*,
        std::int64_t,
        std::int64_t,
        float,
        void*);
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
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);

    void* tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    void* cuda_handle = nullptr;
    FillFn fill = nullptr;
    TokenEmbeddingFn token_embedding = nullptr;
    TokenEmbeddingBackwardWeightFn token_embedding_backward_weight = nullptr;
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightFn linear_backward_weight = nullptr;
    LinearBackwardBiasFn linear_backward_bias = nullptr;
    TokenCrossEntropyPartialsFn ce_partials = nullptr;
    TokenCrossEntropyBackwardWorkspaceFn ce_backward_workspace = nullptr;
    AdamWFn adamw = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed with CUDA error " << code;
        if (cuda_get_error_string != nullptr) {
            const char* message = cuda_get_error_string(code);
            if (message != nullptr) {
                out << ": " << message;
            }
        }
        return out.str();
    };

    if (tile_handle == nullptr) {
        error = dl_last_error("dlopen tile ops failed");
    } else {
        tile_loaded = true;
        fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
        token_embedding = load_symbol<TokenEmbeddingFn>(tile_handle, "nfn_native_tile_token_embedding_float32");
        token_embedding_backward_weight = load_symbol<TokenEmbeddingBackwardWeightFn>(
            tile_handle, "nfn_native_tile_token_embedding_backward_weight_float32");
        linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
        linear_backward_input = load_symbol<LinearBackwardInputFn>(
            tile_handle, "nfn_native_tile_linear_backward_input_float32");
        linear_backward_weight = load_symbol<LinearBackwardWeightFn>(
            tile_handle, "nfn_native_tile_linear_backward_weight_float32");
        linear_backward_bias = load_symbol<LinearBackwardBiasFn>(
            tile_handle, "nfn_native_tile_linear_backward_bias_float32");
        ce_partials = load_symbol<TokenCrossEntropyPartialsFn>(
            tile_handle, "nfn_native_tile_token_cross_entropy_partials_float32");
        ce_backward_workspace = load_symbol<TokenCrossEntropyBackwardWorkspaceFn>(
            tile_handle, "nfn_native_tile_token_cross_entropy_backward_with_workspace_float32");
        adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
        if (fill == nullptr || token_embedding == nullptr || token_embedding_backward_weight == nullptr ||
            linear == nullptr || linear_backward_input == nullptr || linear_backward_weight == nullptr ||
            linear_backward_bias == nullptr ||
            ce_partials == nullptr || ce_backward_workspace == nullptr || adamw == nullptr) {
            error = dl_last_error("dlsym GPT-2 LM-step kernels failed");
        }
    }

    if (error.empty()) {
        std::string runtime_error;
        for (const std::string& candidate : runtime_candidates) {
            cuda_lib_path = candidate;
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_runtime_loaded = true;
                break;
            }
            runtime_error = dl_last_error("dlopen CUDA runtime failed");
        }
        if (cuda_handle == nullptr) {
            error = runtime_error.empty() ? "could not load CUDA runtime" : runtime_error;
        }
    }

    if (error.empty()) {
        cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
        cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
        cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
        cuda_device_synchronize = load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
        cuda_get_error_string = load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
        if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr || cuda_device_synchronize == nullptr) {
            error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaDeviceSynchronize";
        }
    }

    float* weight = nullptr;
    float* grad_weight = nullptr;
    float* exp_avg = nullptr;
    float* exp_avg_sq = nullptr;
    float* hidden = nullptr;
    float* logits = nullptr;
    float* loss_partials = nullptr;
    float* row_max = nullptr;
    float* row_denom = nullptr;
    float* grad_logits = nullptr;
    float* grad_hidden = nullptr;
    std::int64_t* token_ids = nullptr;
    std::int64_t* targets = nullptr;

    auto allocate = [&](auto** ptr, std::size_t bytes, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        const int status = cuda_malloc(reinterpret_cast<void**>(ptr), bytes);
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc " + name);
        }
    };
    allocate(&weight, sizeof(float) * static_cast<std::size_t>(kWeightElements), "weight");
    allocate(&grad_weight, sizeof(float) * static_cast<std::size_t>(kWeightElements), "grad_weight");
    allocate(&exp_avg, sizeof(float) * static_cast<std::size_t>(kWeightElements), "exp_avg");
    allocate(&exp_avg_sq, sizeof(float) * static_cast<std::size_t>(kWeightElements), "exp_avg_sq");
    allocate(&hidden, sizeof(float) * static_cast<std::size_t>(kActivationElements), "hidden");
    allocate(&logits, sizeof(float) * static_cast<std::size_t>(kLogitElements), "logits");
    allocate(&loss_partials, sizeof(float), "loss_partials");
    allocate(&row_max, sizeof(float) * static_cast<std::size_t>(kRows), "row_max");
    allocate(&row_denom, sizeof(float) * static_cast<std::size_t>(kRows), "row_denom");
    allocate(&grad_logits, sizeof(float) * static_cast<std::size_t>(kLogitElements), "grad_logits");
    allocate(&grad_hidden, sizeof(float) * static_cast<std::size_t>(kActivationElements), "grad_hidden");
    allocate(&token_ids, sizeof(std::int64_t) * static_cast<std::size_t>(kRows), "token_ids");
    allocate(&targets, sizeof(std::int64_t) * static_cast<std::size_t>(kRows), "targets");

    auto fill_buffer = [&](float* ptr, std::int64_t elements, float value, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        const int status = fill(ptr, elements, value, nullptr);
        if (status != 0) {
            error = cuda_error(status, "fill " + name);
        }
    };
    fill_buffer(weight, kWeightElements, kInitialWeight, "weight");
    fill_buffer(grad_weight, kWeightElements, 0.0f, "grad_weight");
    fill_buffer(exp_avg, kWeightElements, 0.0f, "exp_avg");
    fill_buffer(exp_avg_sq, kWeightElements, 0.0f, "exp_avg_sq");
    fill_buffer(grad_hidden, kActivationElements, 0.0f, "grad_hidden");

    auto copy_to_device = [&](void* dst, const void* src, std::size_t bytes, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        const int status = cuda_memcpy(dst, src, bytes, kCudaMemcpyHostToDevice);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy host-to-device " + name);
        }
    };
    copy_to_device(token_ids, host_tokens, sizeof(host_tokens), "token_ids");
    copy_to_device(targets, host_targets, sizeof(host_targets), "targets");

    auto run = [&](int status, const std::string& name) {
        if (status != 0 && error.empty()) {
            error = cuda_error(status, name);
        }
    };
    if (error.empty()) {
        run(token_embedding(weight, token_ids, hidden, kRows, kDim, nullptr), "token_embedding");
    }
    if (error.empty()) {
        run(linear(hidden, weight, nullptr, logits, kRows, kDim, kPaddedVocab, false, nullptr), "linear");
    }
    if (error.empty()) {
        run(ce_partials(logits, targets, loss_partials, kRows, kPaddedVocab, nullptr), "token_cross_entropy_partials");
    }
    if (error.empty()) {
        run(ce_backward_workspace(
                logits,
                targets,
                row_max,
                row_denom,
                grad_logits,
                kRows,
                kPaddedVocab,
                1.0f / static_cast<float>(kRows),
                nullptr),
            "token_cross_entropy_backward_with_workspace");
    }
    if (error.empty()) {
        run(linear_backward_input(grad_logits, weight, grad_hidden, kRows, kDim, kPaddedVocab, nullptr),
            "linear_backward_input");
    }
    if (error.empty()) {
        run(linear_backward_weight(hidden, grad_logits, grad_weight, kRows, kDim, kPaddedVocab, nullptr),
            "linear_backward_weight");
    }
    if (error.empty()) {
        run(token_embedding_backward_weight(token_ids, grad_hidden, grad_weight, kRows, kDim, nullptr),
            "token_embedding_backward_weight");
    }
    if (error.empty()) {
        run(adamw(
                weight,
                grad_weight,
                exp_avg,
                exp_avg_sq,
                kWeightElements,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                kWeightDecay,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr),
            "adamw_step");
    }
    if (error.empty()) {
        run(cuda_device_synchronize(), "cudaDeviceSynchronize");
    }

    auto copy_float_sample = [&](const float* device_ptr, std::int64_t offset, const std::string& name) -> float {
        float value = 0.0f;
        if (!error.empty()) {
            return value;
        }
        const int status = cuda_memcpy(&value, device_ptr + offset, sizeof(float), kCudaMemcpyDeviceToHost);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy sample " + name);
        }
        return value;
    };
    if (error.empty()) {
        const float sampled_loss = copy_float_sample(loss_partials, 0, "loss_partials");
        loss_abs_error = std::fabs(static_cast<double>(sampled_loss) - expected_loss);
        const std::int64_t sampled_tokens[] = {0, 1, 3, 4, 17};
        for (std::int64_t token : sampled_tokens) {
            const std::int64_t offset = token * kDim;
            const float expected_grad = expected_grad_for_token(token);
            const float expected_weight = expected_param_for_grad(expected_grad);
            const float sampled_grad = copy_float_sample(grad_weight, offset, "grad_weight");
            const float sampled_weight = copy_float_sample(weight, offset, "weight");
            if (!error.empty()) {
                break;
            }
            const double grad_error = std::fabs(static_cast<double>(sampled_grad) - expected_grad);
            const double weight_error = std::fabs(static_cast<double>(sampled_weight) - expected_weight);
            if (grad_error > max_grad_abs_error) {
                max_grad_abs_error = grad_error;
            }
            if (weight_error > max_weight_abs_error) {
                max_weight_abs_error = weight_error;
            }
        }
    }

    if (error.empty()) {
        passed = loss_abs_error <= 1e-4 && max_grad_abs_error <= 1e-5 && max_weight_abs_error <= 1e-4;
        if (!passed) {
            std::ostringstream out;
            out << "GPT-2 LM smoke exceeded tolerance: loss=" << loss_abs_error
                << " grad=" << max_grad_abs_error
                << " weight=" << max_weight_abs_error;
            error = out.str();
        }
    }

    auto free_device = [&](void* ptr, const std::string& name) {
        if (ptr != nullptr && cuda_free != nullptr) {
            const int status = cuda_free(ptr);
            if (status != 0 && error.empty()) {
                error = cuda_error(status, "cudaFree " + name);
            }
        }
    };
    free_device(weight, "weight");
    free_device(grad_weight, "grad_weight");
    free_device(exp_avg, "exp_avg");
    free_device(exp_avg_sq, "exp_avg_sq");
    free_device(hidden, "hidden");
    free_device(logits, "logits");
    free_device(loss_partials, "loss_partials");
    free_device(row_max, "row_max");
    free_device(row_denom, "row_denom");
    free_device(grad_logits, "grad_logits");
    free_device(grad_hidden, "grad_hidden");
    free_device(token_ids, "token_ids");
    free_device(targets, "targets");
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(cfg.model_family) << "\",\n"
        << "  \"backend\": \"tile-cuda\",\n"
        << "  \"smoke\": \"lm_step\",\n"
        << "  \"token_shards_resolved\": false,\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"rows\": " << kRows << ",\n"
        << "  \"vocab\": " << kVocab << ",\n"
        << "  \"padded_vocab\": " << kPaddedVocab << ",\n"
        << "  \"model_dim\": " << kDim << ",\n"
        << "  \"kernels\": [\n"
        << "    \"nfn_native_tile_token_embedding_float32\",\n"
        << "    \"nfn_native_tile_linear_float32\",\n"
        << "    \"nfn_native_tile_token_cross_entropy_partials_float32\",\n"
        << "    \"nfn_native_tile_token_cross_entropy_backward_with_workspace_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_input_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_weight_float32\",\n"
        << "    \"nfn_native_tile_token_embedding_backward_weight_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"expected_loss\": " << expected_loss << ",\n"
        << "  \"loss_abs_error\": " << loss_abs_error << ",\n"
        << "  \"max_grad_abs_error\": " << max_grad_abs_error << ",\n"
        << "  \"max_weight_abs_error\": " << max_weight_abs_error << ",\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";

    return passed ? 0 : 2;
}

int print_embedding_lm_step_smoke_json(
    const Config& cfg,
    const neuralfn::native_train::TokenShardDataset& dataset,
    const char* program) {
    constexpr std::int64_t kBatch = 1;
    constexpr std::int64_t kSeq = 2;
    constexpr std::int64_t kRows = kBatch * kSeq;
    constexpr std::int64_t kVocab = 50257;
    constexpr std::int64_t kPaddedVocab = 50304;
    constexpr std::int64_t kDim = 768;
    constexpr std::int64_t kTokenWeightElements = kPaddedVocab * kDim;
    constexpr std::int64_t kPositionWeightElements = kSeq * kDim;
    constexpr std::int64_t kActivationElements = kRows * kDim;
    constexpr std::int64_t kLogitElements = kRows * kPaddedVocab;
    constexpr float kInitialTokenWeight = 0.01f;
    constexpr float kInitialPositionWeight = 0.02f;
    constexpr float kInitialLnWeight = 1.0f;
    constexpr float kInitialLnBias = 0.0f;
    constexpr float kResidualScale = 1.0f;
    constexpr float kLearningRate = 0.1f;
    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.95f;
    constexpr float kEps = 1e-8f;
    constexpr float kNormEps = 1e-5f;
    constexpr float kWeightDecay = 0.1f;
    constexpr float kBiasCorrection1 = 1.0f - kBeta1;
    constexpr float kSqrtBiasCorrection2 = 0.22360679775f;
    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;

    neuralfn::native_train::BatchPlan batch_plan;
    neuralfn::native_train::TokenBatch batch;
    bool have_batch = false;
    std::string error;
    try {
        batch_plan = neuralfn::native_train::build_batch_plan(dataset, kSeq, kBatch, kRows);
        neuralfn::native_train::SequentialTokenBatchSampler sampler(dataset.train_shards, kSeq, kBatch);
        have_batch = sampler.next(batch);
        if (!have_batch) {
            error = "not enough train tokens to build one GPT-2 embedding/LM smoke batch";
        }
    } catch (const std::exception& exc) {
        error = exc.what();
    }

    std::int64_t host_tokens[kRows] = {0, 1};
    std::int64_t host_targets[kRows] = {1, 2};
    if (error.empty()) {
        if (batch.tokens.size() != static_cast<std::size_t>(kRows) ||
            batch.targets.size() != static_cast<std::size_t>(kRows)) {
            error = "sampled GPT-2 embedding/LM smoke batch has unexpected size";
        } else {
            for (std::int64_t i = 0; i < kRows; ++i) {
                host_tokens[i] = static_cast<std::int64_t>(batch.tokens[static_cast<std::size_t>(i)]);
                host_targets[i] = static_cast<std::int64_t>(batch.targets[static_cast<std::size_t>(i)]);
                if (host_tokens[i] < 0 || host_tokens[i] >= kVocab ||
                    host_targets[i] < 0 || host_targets[i] >= kVocab) {
                    std::ostringstream out;
                    out << "sampled token/target id exceeds GPT-2 vocab at row " << i
                        << ": token=" << host_tokens[i]
                        << " target=" << host_targets[i]
                        << " vocab=" << kVocab;
                    error = out.str();
                    break;
                }
            }
        }
    }

    const std::string tile_lib_path = cfg.tile_ops_lib.empty() ? default_tile_ops_lib(program) : cfg.tile_ops_lib;
    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(cfg);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool passed = false;
    double max_abs_sample = 0.0;
    double max_weight_delta = 0.0;

    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using TokenEmbeddingFn = int (*)(const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using TokenEmbeddingBackwardWeightFn = int (*)(
        const std::int64_t*, const float*, float*, std::int64_t, std::int64_t, void*);
    using PositionEmbeddingFn = int (*)(const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using PositionEmbeddingBackwardFn = int (*)(const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using ResidualAddFn = int (*)(const float*, const float*, const float*, float*, std::int64_t, void*);
    using LayerNormFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, float, void*);
    using LayerNormBackwardInputFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, float, void*);
    using LayerNormBackwardAffineFn = int (*)(
        const float*, const float*, float*, float*, std::int64_t, std::int64_t, float, void*);
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBackwardInputFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyPartialsFn = int (*)(
        const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyBackwardWorkspaceFn = int (*)(
        const float*,
        const std::int64_t*,
        float*,
        float*,
        float*,
        std::int64_t,
        std::int64_t,
        float,
        void*);
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
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);

    void* tile_handle = nullptr;
    void* cuda_handle = nullptr;
    FillFn fill = nullptr;
    TokenEmbeddingFn token_embedding = nullptr;
    TokenEmbeddingBackwardWeightFn token_embedding_backward_weight = nullptr;
    PositionEmbeddingFn position_embedding = nullptr;
    PositionEmbeddingBackwardFn position_embedding_backward = nullptr;
    ResidualAddFn residual_add = nullptr;
    LayerNormFn layer_norm = nullptr;
    LayerNormBackwardInputFn layer_norm_backward_input = nullptr;
    LayerNormBackwardAffineFn layer_norm_backward_affine = nullptr;
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightFn linear_backward_weight = nullptr;
    TokenCrossEntropyPartialsFn ce_partials = nullptr;
    TokenCrossEntropyBackwardWorkspaceFn ce_backward_workspace = nullptr;
    AdamWFn adamw = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed with CUDA error " << code;
        if (cuda_get_error_string != nullptr) {
            const char* message = cuda_get_error_string(code);
            if (message != nullptr) {
                out << ": " << message;
            }
        }
        return out.str();
    };

    if (error.empty()) {
        tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (tile_handle == nullptr) {
            error = dl_last_error("dlopen tile ops failed");
        } else {
            tile_loaded = true;
            fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
            token_embedding = load_symbol<TokenEmbeddingFn>(tile_handle, "nfn_native_tile_token_embedding_float32");
            token_embedding_backward_weight = load_symbol<TokenEmbeddingBackwardWeightFn>(
                tile_handle, "nfn_native_tile_token_embedding_backward_weight_float32");
            position_embedding = load_symbol<PositionEmbeddingFn>(
                tile_handle, "nfn_native_tile_absolute_position_embedding_float32");
            position_embedding_backward = load_symbol<PositionEmbeddingBackwardFn>(
                tile_handle, "nfn_native_tile_absolute_position_embedding_backward_float32");
            residual_add = load_symbol<ResidualAddFn>(tile_handle, "nfn_native_tile_scaled_residual_add_float32");
            layer_norm = load_symbol<LayerNormFn>(tile_handle, "nfn_native_tile_layer_norm_float32");
            layer_norm_backward_input = load_symbol<LayerNormBackwardInputFn>(
                tile_handle, "nfn_native_tile_layer_norm_backward_input_float32");
            layer_norm_backward_affine = load_symbol<LayerNormBackwardAffineFn>(
                tile_handle, "nfn_native_tile_layer_norm_backward_affine_float32");
            linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
            linear_backward_input = load_symbol<LinearBackwardInputFn>(
                tile_handle, "nfn_native_tile_linear_backward_input_float32");
            linear_backward_weight = load_symbol<LinearBackwardWeightFn>(
                tile_handle, "nfn_native_tile_linear_backward_weight_float32");
            ce_partials = load_symbol<TokenCrossEntropyPartialsFn>(
                tile_handle, "nfn_native_tile_token_cross_entropy_partials_float32");
            ce_backward_workspace = load_symbol<TokenCrossEntropyBackwardWorkspaceFn>(
                tile_handle, "nfn_native_tile_token_cross_entropy_backward_with_workspace_float32");
            adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
            if (fill == nullptr || token_embedding == nullptr || token_embedding_backward_weight == nullptr ||
                position_embedding == nullptr || position_embedding_backward == nullptr || residual_add == nullptr ||
                layer_norm == nullptr || layer_norm_backward_input == nullptr || layer_norm_backward_affine == nullptr ||
                linear == nullptr || linear_backward_input == nullptr || linear_backward_weight == nullptr ||
                ce_partials == nullptr || ce_backward_workspace == nullptr || adamw == nullptr) {
                error = dl_last_error("dlsym GPT-2 embedding/LM-step kernels failed");
            }
        }
    }

    if (error.empty()) {
        std::string runtime_error;
        for (const std::string& candidate : runtime_candidates) {
            cuda_lib_path = candidate;
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_runtime_loaded = true;
                break;
            }
            runtime_error = dl_last_error("dlopen CUDA runtime failed");
        }
        if (cuda_handle == nullptr) {
            error = runtime_error.empty() ? "could not load CUDA runtime" : runtime_error;
        }
    }

    if (error.empty()) {
        cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
        cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
        cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
        cuda_device_synchronize = load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
        cuda_get_error_string = load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
        if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr || cuda_device_synchronize == nullptr) {
            error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaDeviceSynchronize";
        }
    }

    float* token_weight = nullptr;
    float* position_weight = nullptr;
    float* ln_weight = nullptr;
    float* ln_bias = nullptr;
    float* grad_token_weight = nullptr;
    float* grad_position_weight = nullptr;
    float* grad_ln_weight = nullptr;
    float* grad_ln_bias = nullptr;
    float* token_exp_avg = nullptr;
    float* token_exp_avg_sq = nullptr;
    float* position_exp_avg = nullptr;
    float* position_exp_avg_sq = nullptr;
    float* ln_weight_exp_avg = nullptr;
    float* ln_weight_exp_avg_sq = nullptr;
    float* ln_bias_exp_avg = nullptr;
    float* ln_bias_exp_avg_sq = nullptr;
    float* token_out = nullptr;
    float* position_out = nullptr;
    float* residual = nullptr;
    float* ln_out = nullptr;
    float* logits = nullptr;
    float* loss_partials = nullptr;
    float* row_max = nullptr;
    float* row_denom = nullptr;
    float* grad_logits = nullptr;
    float* grad_ln = nullptr;
    float* grad_residual = nullptr;
    float* residual_scale = nullptr;
    std::int64_t* token_ids = nullptr;
    std::int64_t* targets = nullptr;

    auto allocate = [&](auto** ptr, std::size_t bytes, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        const int status = cuda_malloc(reinterpret_cast<void**>(ptr), bytes);
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc " + name);
        }
    };
    allocate(&token_weight, sizeof(float) * static_cast<std::size_t>(kTokenWeightElements), "token_weight");
    allocate(&position_weight, sizeof(float) * static_cast<std::size_t>(kPositionWeightElements), "position_weight");
    allocate(&ln_weight, sizeof(float) * static_cast<std::size_t>(kDim), "ln_weight");
    allocate(&ln_bias, sizeof(float) * static_cast<std::size_t>(kDim), "ln_bias");
    allocate(&grad_token_weight, sizeof(float) * static_cast<std::size_t>(kTokenWeightElements), "grad_token_weight");
    allocate(&grad_position_weight, sizeof(float) * static_cast<std::size_t>(kPositionWeightElements), "grad_position_weight");
    allocate(&grad_ln_weight, sizeof(float) * static_cast<std::size_t>(kDim), "grad_ln_weight");
    allocate(&grad_ln_bias, sizeof(float) * static_cast<std::size_t>(kDim), "grad_ln_bias");
    allocate(&token_exp_avg, sizeof(float) * static_cast<std::size_t>(kTokenWeightElements), "token_exp_avg");
    allocate(&token_exp_avg_sq, sizeof(float) * static_cast<std::size_t>(kTokenWeightElements), "token_exp_avg_sq");
    allocate(&position_exp_avg, sizeof(float) * static_cast<std::size_t>(kPositionWeightElements), "position_exp_avg");
    allocate(&position_exp_avg_sq, sizeof(float) * static_cast<std::size_t>(kPositionWeightElements), "position_exp_avg_sq");
    allocate(&ln_weight_exp_avg, sizeof(float) * static_cast<std::size_t>(kDim), "ln_weight_exp_avg");
    allocate(&ln_weight_exp_avg_sq, sizeof(float) * static_cast<std::size_t>(kDim), "ln_weight_exp_avg_sq");
    allocate(&ln_bias_exp_avg, sizeof(float) * static_cast<std::size_t>(kDim), "ln_bias_exp_avg");
    allocate(&ln_bias_exp_avg_sq, sizeof(float) * static_cast<std::size_t>(kDim), "ln_bias_exp_avg_sq");
    allocate(&token_out, sizeof(float) * static_cast<std::size_t>(kActivationElements), "token_out");
    allocate(&position_out, sizeof(float) * static_cast<std::size_t>(kActivationElements), "position_out");
    allocate(&residual, sizeof(float) * static_cast<std::size_t>(kActivationElements), "residual");
    allocate(&ln_out, sizeof(float) * static_cast<std::size_t>(kActivationElements), "ln_out");
    allocate(&logits, sizeof(float) * static_cast<std::size_t>(kLogitElements), "logits");
    allocate(&loss_partials, sizeof(float), "loss_partials");
    allocate(&row_max, sizeof(float) * static_cast<std::size_t>(kRows), "row_max");
    allocate(&row_denom, sizeof(float) * static_cast<std::size_t>(kRows), "row_denom");
    allocate(&grad_logits, sizeof(float) * static_cast<std::size_t>(kLogitElements), "grad_logits");
    allocate(&grad_ln, sizeof(float) * static_cast<std::size_t>(kActivationElements), "grad_ln");
    allocate(&grad_residual, sizeof(float) * static_cast<std::size_t>(kActivationElements), "grad_residual");
    allocate(&residual_scale, sizeof(float), "residual_scale");
    allocate(&token_ids, sizeof(std::int64_t) * static_cast<std::size_t>(kRows), "token_ids");
    allocate(&targets, sizeof(std::int64_t) * static_cast<std::size_t>(kRows), "targets");

    auto fill_buffer = [&](float* ptr, std::int64_t elements, float value, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        const int status = fill(ptr, elements, value, nullptr);
        if (status != 0) {
            error = cuda_error(status, "fill " + name);
        }
    };
    fill_buffer(token_weight, kTokenWeightElements, kInitialTokenWeight, "token_weight");
    fill_buffer(position_weight, kPositionWeightElements, kInitialPositionWeight, "position_weight");
    fill_buffer(ln_weight, kDim, kInitialLnWeight, "ln_weight");
    fill_buffer(ln_bias, kDim, kInitialLnBias, "ln_bias");
    fill_buffer(grad_token_weight, kTokenWeightElements, 0.0f, "grad_token_weight");
    fill_buffer(grad_position_weight, kPositionWeightElements, 0.0f, "grad_position_weight");
    fill_buffer(grad_ln_weight, kDim, 0.0f, "grad_ln_weight");
    fill_buffer(grad_ln_bias, kDim, 0.0f, "grad_ln_bias");
    fill_buffer(token_exp_avg, kTokenWeightElements, 0.0f, "token_exp_avg");
    fill_buffer(token_exp_avg_sq, kTokenWeightElements, 0.0f, "token_exp_avg_sq");
    fill_buffer(position_exp_avg, kPositionWeightElements, 0.0f, "position_exp_avg");
    fill_buffer(position_exp_avg_sq, kPositionWeightElements, 0.0f, "position_exp_avg_sq");
    fill_buffer(ln_weight_exp_avg, kDim, 0.0f, "ln_weight_exp_avg");
    fill_buffer(ln_weight_exp_avg_sq, kDim, 0.0f, "ln_weight_exp_avg_sq");
    fill_buffer(ln_bias_exp_avg, kDim, 0.0f, "ln_bias_exp_avg");
    fill_buffer(ln_bias_exp_avg_sq, kDim, 0.0f, "ln_bias_exp_avg_sq");
    fill_buffer(residual_scale, 1, kResidualScale, "residual_scale");

    auto copy_to_device = [&](void* dst, const void* src, std::size_t bytes, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        const int status = cuda_memcpy(dst, src, bytes, kCudaMemcpyHostToDevice);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy host-to-device " + name);
        }
    };
    copy_to_device(token_ids, host_tokens, sizeof(host_tokens), "token_ids");
    copy_to_device(targets, host_targets, sizeof(host_targets), "targets");

    auto run = [&](int status, const std::string& name) {
        if (status != 0 && error.empty()) {
            error = cuda_error(status, name);
        }
    };
    if (error.empty()) {
        run(token_embedding(token_weight, token_ids, token_out, kRows, kDim, nullptr), "wte.forward");
    }
    if (error.empty()) {
        run(position_embedding(position_weight, position_out, kBatch, kSeq, kDim, nullptr), "wpe.forward");
    }
    if (error.empty()) {
        run(residual_add(token_out, position_out, residual_scale, residual, kActivationElements, nullptr),
            "embedding_residual.forward");
    }
    if (error.empty()) {
        run(layer_norm(residual, ln_weight, ln_bias, ln_out, kRows, kDim, kNormEps, nullptr), "ln_f.forward");
    }
    if (error.empty()) {
        run(linear(ln_out, token_weight, nullptr, logits, kRows, kDim, kPaddedVocab, false, nullptr), "lm_head.forward");
    }
    if (error.empty()) {
        run(ce_partials(logits, targets, loss_partials, kRows, kPaddedVocab, nullptr), "token_cross_entropy.partials");
    }
    if (error.empty()) {
        run(ce_backward_workspace(
                logits,
                targets,
                row_max,
                row_denom,
                grad_logits,
                kRows,
                kPaddedVocab,
                1.0f / static_cast<float>(kRows),
                nullptr),
            "token_cross_entropy.backward_with_workspace");
    }
    if (error.empty()) {
        run(linear_backward_input(grad_logits, token_weight, grad_ln, kRows, kDim, kPaddedVocab, nullptr),
            "lm_head.backward_input");
    }
    if (error.empty()) {
        run(linear_backward_weight(ln_out, grad_logits, grad_token_weight, kRows, kDim, kPaddedVocab, nullptr),
            "lm_head.backward_weight");
    }
    if (error.empty()) {
        run(layer_norm_backward_affine(residual, grad_ln, grad_ln_weight, grad_ln_bias, kRows, kDim, kNormEps, nullptr),
            "ln_f.backward_affine");
    }
    if (error.empty()) {
        run(layer_norm_backward_input(residual, grad_ln, ln_weight, grad_residual, kRows, kDim, kNormEps, nullptr),
            "ln_f.backward_input");
    }
    if (error.empty()) {
        run(token_embedding_backward_weight(token_ids, grad_residual, grad_token_weight, kRows, kDim, nullptr),
            "wte.backward_weight");
    }
    if (error.empty()) {
        run(position_embedding_backward(grad_residual, grad_position_weight, kBatch, kSeq, kDim, nullptr),
            "wpe.backward_weight");
    }
    if (error.empty()) {
        run(adamw(token_weight, grad_token_weight, token_exp_avg, token_exp_avg_sq, kTokenWeightElements, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2, nullptr), "wte.adamw");
    }
    if (error.empty()) {
        run(adamw(position_weight, grad_position_weight, position_exp_avg, position_exp_avg_sq, kPositionWeightElements, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2, nullptr), "wpe.adamw");
    }
    if (error.empty()) {
        run(adamw(ln_weight, grad_ln_weight, ln_weight_exp_avg, ln_weight_exp_avg_sq, kDim, kLearningRate, kBeta1, kBeta2, kEps, kWeightDecay, kBiasCorrection1, kSqrtBiasCorrection2, nullptr), "ln_f.weight.adamw");
    }
    if (error.empty()) {
        run(adamw(ln_bias, grad_ln_bias, ln_bias_exp_avg, ln_bias_exp_avg_sq, kDim, kLearningRate, kBeta1, kBeta2, kEps, 0.0f, kBiasCorrection1, kSqrtBiasCorrection2, nullptr), "ln_f.bias.adamw");
    }
    if (error.empty()) {
        run(cuda_device_synchronize(), "cudaDeviceSynchronize");
    }

    auto copy_float_sample = [&](const float* device_ptr, std::int64_t offset, const std::string& name) -> float {
        float value = 0.0f;
        if (!error.empty()) {
            return value;
        }
        const int status = cuda_memcpy(&value, device_ptr + offset, sizeof(float), kCudaMemcpyDeviceToHost);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy sample " + name);
        }
        return value;
    };
    float sampled_residual = 0.0f;
    float sampled_ln_out = 0.0f;
    float sampled_loss = 0.0f;
    float sampled_grad_token = 0.0f;
    float sampled_grad_position = 0.0f;
    float sampled_grad_ln_weight = 0.0f;
    float sampled_token_weight = 0.0f;
    float sampled_position_weight = 0.0f;
    float sampled_ln_weight = 0.0f;
    if (error.empty()) {
        sampled_residual = copy_float_sample(residual, 0, "residual");
        sampled_ln_out = copy_float_sample(ln_out, 0, "ln_out");
        sampled_loss = copy_float_sample(loss_partials, 0, "loss_partials");
        sampled_grad_token = copy_float_sample(grad_token_weight, host_targets[0] * kDim, "grad_token_weight");
        sampled_grad_position = copy_float_sample(grad_position_weight, 0, "grad_position_weight");
        sampled_grad_ln_weight = copy_float_sample(grad_ln_weight, 0, "grad_ln_weight");
        sampled_token_weight = copy_float_sample(token_weight, 0, "token_weight");
        sampled_position_weight = copy_float_sample(position_weight, 0, "position_weight");
        sampled_ln_weight = copy_float_sample(ln_weight, 0, "ln_weight");
    }

    auto record_abs = [&](float value) {
        if (!std::isfinite(value)) {
            return false;
        }
        const double abs_value = std::fabs(static_cast<double>(value));
        if (abs_value > max_abs_sample) {
            max_abs_sample = abs_value;
        }
        return true;
    };
    if (error.empty()) {
        const bool finite_samples =
            record_abs(sampled_residual) && record_abs(sampled_ln_out) &&
            record_abs(sampled_loss) && record_abs(sampled_grad_token) &&
            record_abs(sampled_grad_position) && record_abs(sampled_grad_ln_weight) &&
            record_abs(sampled_token_weight) && record_abs(sampled_position_weight) &&
            record_abs(sampled_ln_weight);
        max_weight_delta = std::fabs(static_cast<double>(sampled_token_weight) - kInitialTokenWeight);
        const double position_delta = std::fabs(static_cast<double>(sampled_position_weight) - kInitialPositionWeight);
        const double ln_delta = std::fabs(static_cast<double>(sampled_ln_weight) - kInitialLnWeight);
        if (position_delta > max_weight_delta) {
            max_weight_delta = position_delta;
        }
        if (ln_delta > max_weight_delta) {
            max_weight_delta = ln_delta;
        }
        passed = finite_samples && sampled_loss > 0.0f && max_abs_sample > 0.0 && max_weight_delta > 0.0;
        if (!passed) {
            std::ostringstream out_message;
            out_message << "GPT-2 embedding/LM smoke did not produce finite loss and weight updates: sample="
                        << max_abs_sample << " loss=" << sampled_loss
                        << " weight_delta=" << max_weight_delta;
            error = out_message.str();
        }
    }

    auto free_device = [&](void* ptr, const std::string& name) {
        if (ptr != nullptr && cuda_free != nullptr) {
            const int status = cuda_free(ptr);
            if (status != 0 && error.empty()) {
                error = cuda_error(status, "cudaFree " + name);
            }
        }
    };
    free_device(token_weight, "token_weight");
    free_device(position_weight, "position_weight");
    free_device(ln_weight, "ln_weight");
    free_device(ln_bias, "ln_bias");
    free_device(grad_token_weight, "grad_token_weight");
    free_device(grad_position_weight, "grad_position_weight");
    free_device(grad_ln_weight, "grad_ln_weight");
    free_device(grad_ln_bias, "grad_ln_bias");
    free_device(token_exp_avg, "token_exp_avg");
    free_device(token_exp_avg_sq, "token_exp_avg_sq");
    free_device(position_exp_avg, "position_exp_avg");
    free_device(position_exp_avg_sq, "position_exp_avg_sq");
    free_device(ln_weight_exp_avg, "ln_weight_exp_avg");
    free_device(ln_weight_exp_avg_sq, "ln_weight_exp_avg_sq");
    free_device(ln_bias_exp_avg, "ln_bias_exp_avg");
    free_device(ln_bias_exp_avg_sq, "ln_bias_exp_avg_sq");
    free_device(token_out, "token_out");
    free_device(position_out, "position_out");
    free_device(residual, "residual");
    free_device(ln_out, "ln_out");
    free_device(logits, "logits");
    free_device(loss_partials, "loss_partials");
    free_device(row_max, "row_max");
    free_device(row_denom, "row_denom");
    free_device(grad_logits, "grad_logits");
    free_device(grad_ln, "grad_ln");
    free_device(grad_residual, "grad_residual");
    free_device(residual_scale, "residual_scale");
    free_device(token_ids, "token_ids");
    free_device(targets, "targets");
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(cfg.model_family) << "\",\n"
        << "  \"backend\": \"tile-cuda\",\n"
        << "  \"smoke\": \"embedding_lm_step\",\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"dataset_loaded\": true,\n"
        << "  \"batch_loaded\": " << (have_batch ? "true" : "false") << ",\n"
        << "  \"token_shards\": " << neuralfn::native_train::token_shard_dataset_json(dataset, &batch_plan) << ",\n"
        << "  \"sample_batch\": ";
    if (have_batch) {
        std::cout << neuralfn::native_train::token_batch_json(batch);
    } else {
        std::cout << "null";
    }
    std::cout
        << ",\n"
        << "  \"batch\": " << kBatch << ",\n"
        << "  \"seq\": " << kSeq << ",\n"
        << "  \"rows\": " << kRows << ",\n"
        << "  \"vocab\": " << kVocab << ",\n"
        << "  \"padded_vocab\": " << kPaddedVocab << ",\n"
        << "  \"model_dim\": " << kDim << ",\n"
        << "  \"weight_update_count\": 4,\n"
        << "  \"kernels\": [\n"
        << "    \"nfn_native_tile_token_embedding_float32\",\n"
        << "    \"nfn_native_tile_absolute_position_embedding_float32\",\n"
        << "    \"nfn_native_tile_scaled_residual_add_float32\",\n"
        << "    \"nfn_native_tile_layer_norm_float32\",\n"
        << "    \"nfn_native_tile_linear_float32\",\n"
        << "    \"nfn_native_tile_token_cross_entropy_partials_float32\",\n"
        << "    \"nfn_native_tile_token_cross_entropy_backward_with_workspace_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_input_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_weight_float32\",\n"
        << "    \"nfn_native_tile_layer_norm_backward_affine_float32\",\n"
        << "    \"nfn_native_tile_layer_norm_backward_input_float32\",\n"
        << "    \"nfn_native_tile_token_embedding_backward_weight_float32\",\n"
        << "    \"nfn_native_tile_absolute_position_embedding_backward_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"sample_residual\": " << sampled_residual << ",\n"
        << "  \"sample_ln_out\": " << sampled_ln_out << ",\n"
        << "  \"sample_loss\": " << sampled_loss << ",\n"
        << "  \"sample_grad_token_weight\": " << sampled_grad_token << ",\n"
        << "  \"sample_grad_position_weight\": " << sampled_grad_position << ",\n"
        << "  \"sample_grad_ln_weight\": " << sampled_grad_ln_weight << ",\n"
        << "  \"sample_updated_token_weight\": " << sampled_token_weight << ",\n"
        << "  \"sample_updated_position_weight\": " << sampled_position_weight << ",\n"
        << "  \"sample_updated_ln_weight\": " << sampled_ln_weight << ",\n"
        << "  \"max_abs_sample\": " << max_abs_sample << ",\n"
        << "  \"max_weight_delta\": " << max_weight_delta << ",\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";

    return passed ? 0 : 2;
}

int run_embedding_lm_training_json(
    const Config& cfg,
    const neuralfn::native_train::TokenShardDataset& dataset,
    const char* program) {
    constexpr std::int64_t kVocab = 50257;
    constexpr std::int64_t kPaddedVocab = 50304;
    constexpr std::int64_t kDim = 768;
    constexpr float kInitialTokenWeight = 0.01f;
    constexpr float kInitialPositionWeight = 0.02f;
    constexpr float kInitialLnWeight = 1.0f;
    constexpr float kInitialLnBias = 0.0f;
    constexpr float kResidualScale = 1.0f;
    constexpr float kNormEps = 1e-5f;
    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.95f;
    constexpr float kAdamEps = 1e-8f;
    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;

    const std::int64_t train_batch_size = cfg.batch_size;
    const std::int64_t eval_batch_size = cfg.eval_batch_size > 0 ? cfg.eval_batch_size : cfg.batch_size;
    const std::int64_t seq_len = cfg.seq_len;
    const std::int64_t train_rows = train_batch_size * seq_len;
    const std::int64_t eval_rows = eval_batch_size * seq_len;
    const std::int64_t max_rows = train_rows > eval_rows ? train_rows : eval_rows;
    const std::int64_t token_weight_elements = kPaddedVocab * kDim;
    const std::int64_t position_weight_elements = seq_len * kDim;
    const std::int64_t activation_elements = max_rows * kDim;
    const std::int64_t logit_elements = max_rows * kPaddedVocab;
    std::string error;
    neuralfn::native_train::BatchPlan batch_plan;
    try {
        batch_plan = neuralfn::native_train::build_batch_plan(
            dataset, seq_len, train_batch_size, cfg.train_batch_tokens);
    } catch (const std::exception& exc) {
        error = exc.what();
    }
    if (error.empty() && (train_rows <= 0 || eval_rows <= 0 || seq_len <= 0 || cfg.max_steps <= 0)) {
        error = "batch size, eval batch size, seq_len, and max_steps must be positive";
    }

    const std::string tile_lib_path = cfg.tile_ops_lib.empty() ? default_tile_ops_lib(program) : cfg.tile_ops_lib;
    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(cfg);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool passed = false;
    std::int64_t steps_completed = 0;
    std::int64_t epochs_completed = 0;
    std::int64_t tokens_processed = 0;
    double final_loss_sum = 0.0;
    double final_loss_mean = 0.0;
    std::vector<ValidationLossRecord> validation_losses;

    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using TokenEmbeddingFn = int (*)(const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using TokenEmbeddingBackwardWeightFn = int (*)(
        const std::int64_t*, const float*, float*, std::int64_t, std::int64_t, void*);
    using PositionEmbeddingFn = int (*)(const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using PositionEmbeddingBackwardFn = int (*)(const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using ResidualAddFn = int (*)(const float*, const float*, const float*, float*, std::int64_t, void*);
    using LayerNormFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, float, void*);
    using LayerNormBackwardInputFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, float, void*);
    using LayerNormBackwardAffineFn = int (*)(
        const float*, const float*, float*, float*, std::int64_t, std::int64_t, float, void*);
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBackwardInputFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyPartialsFn = int (*)(
        const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyBackwardWorkspaceFn = int (*)(
        const float*,
        const std::int64_t*,
        float*,
        float*,
        float*,
        std::int64_t,
        std::int64_t,
        float,
        void*);
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
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);

    void* tile_handle = nullptr;
    void* cuda_handle = nullptr;
    FillFn fill = nullptr;
    TokenEmbeddingFn token_embedding = nullptr;
    TokenEmbeddingBackwardWeightFn token_embedding_backward_weight = nullptr;
    PositionEmbeddingFn position_embedding = nullptr;
    PositionEmbeddingBackwardFn position_embedding_backward = nullptr;
    ResidualAddFn residual_add = nullptr;
    LayerNormFn layer_norm = nullptr;
    LayerNormBackwardInputFn layer_norm_backward_input = nullptr;
    LayerNormBackwardAffineFn layer_norm_backward_affine = nullptr;
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightFn linear_backward_weight = nullptr;
    TokenCrossEntropyPartialsFn ce_partials = nullptr;
    TokenCrossEntropyBackwardWorkspaceFn ce_backward_workspace = nullptr;
    AdamWFn adamw = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed with CUDA error " << code;
        if (cuda_get_error_string != nullptr) {
            const char* message = cuda_get_error_string(code);
            if (message != nullptr) {
                out << ": " << message;
            }
        }
        return out.str();
    };

    if (error.empty()) {
        tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (tile_handle == nullptr) {
            error = dl_last_error("dlopen tile ops failed");
        } else {
            tile_loaded = true;
            fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
            token_embedding = load_symbol<TokenEmbeddingFn>(tile_handle, "nfn_native_tile_token_embedding_float32");
            token_embedding_backward_weight = load_symbol<TokenEmbeddingBackwardWeightFn>(
                tile_handle, "nfn_native_tile_token_embedding_backward_weight_float32");
            position_embedding = load_symbol<PositionEmbeddingFn>(
                tile_handle, "nfn_native_tile_absolute_position_embedding_float32");
            position_embedding_backward = load_symbol<PositionEmbeddingBackwardFn>(
                tile_handle, "nfn_native_tile_absolute_position_embedding_backward_float32");
            residual_add = load_symbol<ResidualAddFn>(tile_handle, "nfn_native_tile_scaled_residual_add_float32");
            layer_norm = load_symbol<LayerNormFn>(tile_handle, "nfn_native_tile_layer_norm_float32");
            layer_norm_backward_input = load_symbol<LayerNormBackwardInputFn>(
                tile_handle, "nfn_native_tile_layer_norm_backward_input_float32");
            layer_norm_backward_affine = load_symbol<LayerNormBackwardAffineFn>(
                tile_handle, "nfn_native_tile_layer_norm_backward_affine_float32");
            linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
            linear_backward_input = load_symbol<LinearBackwardInputFn>(
                tile_handle, "nfn_native_tile_linear_backward_input_float32");
            linear_backward_weight = load_symbol<LinearBackwardWeightFn>(
                tile_handle, "nfn_native_tile_linear_backward_weight_float32");
            ce_partials = load_symbol<TokenCrossEntropyPartialsFn>(
                tile_handle, "nfn_native_tile_token_cross_entropy_partials_float32");
            ce_backward_workspace = load_symbol<TokenCrossEntropyBackwardWorkspaceFn>(
                tile_handle, "nfn_native_tile_token_cross_entropy_backward_with_workspace_float32");
            adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
            if (fill == nullptr || token_embedding == nullptr || token_embedding_backward_weight == nullptr ||
                position_embedding == nullptr || position_embedding_backward == nullptr || residual_add == nullptr ||
                layer_norm == nullptr || layer_norm_backward_input == nullptr || layer_norm_backward_affine == nullptr ||
                linear == nullptr || linear_backward_input == nullptr || linear_backward_weight == nullptr ||
                ce_partials == nullptr || ce_backward_workspace == nullptr || adamw == nullptr) {
                error = dl_last_error("dlsym GPT-2 embedding/LM training kernels failed");
            }
        }
    }

    if (error.empty()) {
        std::string runtime_error;
        for (const std::string& candidate : runtime_candidates) {
            cuda_lib_path = candidate;
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_runtime_loaded = true;
                break;
            }
            runtime_error = dl_last_error("dlopen CUDA runtime failed");
        }
        if (cuda_handle == nullptr) {
            error = runtime_error.empty() ? "could not load CUDA runtime" : runtime_error;
        }
    }

    if (error.empty()) {
        cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
        cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
        cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
        cuda_device_synchronize = load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
        cuda_get_error_string = load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
        if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr || cuda_device_synchronize == nullptr) {
            error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaDeviceSynchronize";
        }
    }

    float* token_weight = nullptr;
    float* position_weight = nullptr;
    float* ln_weight = nullptr;
    float* ln_bias = nullptr;
    float* grad_token_weight = nullptr;
    float* grad_position_weight = nullptr;
    float* grad_ln_weight = nullptr;
    float* grad_ln_bias = nullptr;
    float* token_exp_avg = nullptr;
    float* token_exp_avg_sq = nullptr;
    float* position_exp_avg = nullptr;
    float* position_exp_avg_sq = nullptr;
    float* ln_weight_exp_avg = nullptr;
    float* ln_weight_exp_avg_sq = nullptr;
    float* ln_bias_exp_avg = nullptr;
    float* ln_bias_exp_avg_sq = nullptr;
    float* token_out = nullptr;
    float* position_out = nullptr;
    float* residual = nullptr;
    float* ln_out = nullptr;
    float* logits = nullptr;
    float* loss_partials = nullptr;
    float* row_max = nullptr;
    float* row_denom = nullptr;
    float* grad_logits = nullptr;
    float* grad_ln = nullptr;
    float* grad_residual = nullptr;
    float* residual_scale = nullptr;
    std::int64_t* token_ids = nullptr;
    std::int64_t* targets = nullptr;

    auto allocate = [&](auto** ptr, std::size_t bytes, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        const int status = cuda_malloc(reinterpret_cast<void**>(ptr), bytes);
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc " + name);
        }
    };
    allocate(&token_weight, sizeof(float) * static_cast<std::size_t>(token_weight_elements), "token_weight");
    allocate(&position_weight, sizeof(float) * static_cast<std::size_t>(position_weight_elements), "position_weight");
    allocate(&ln_weight, sizeof(float) * static_cast<std::size_t>(kDim), "ln_weight");
    allocate(&ln_bias, sizeof(float) * static_cast<std::size_t>(kDim), "ln_bias");
    allocate(&grad_token_weight, sizeof(float) * static_cast<std::size_t>(token_weight_elements), "grad_token_weight");
    allocate(&grad_position_weight, sizeof(float) * static_cast<std::size_t>(position_weight_elements), "grad_position_weight");
    allocate(&grad_ln_weight, sizeof(float) * static_cast<std::size_t>(kDim), "grad_ln_weight");
    allocate(&grad_ln_bias, sizeof(float) * static_cast<std::size_t>(kDim), "grad_ln_bias");
    allocate(&token_exp_avg, sizeof(float) * static_cast<std::size_t>(token_weight_elements), "token_exp_avg");
    allocate(&token_exp_avg_sq, sizeof(float) * static_cast<std::size_t>(token_weight_elements), "token_exp_avg_sq");
    allocate(&position_exp_avg, sizeof(float) * static_cast<std::size_t>(position_weight_elements), "position_exp_avg");
    allocate(&position_exp_avg_sq, sizeof(float) * static_cast<std::size_t>(position_weight_elements), "position_exp_avg_sq");
    allocate(&ln_weight_exp_avg, sizeof(float) * static_cast<std::size_t>(kDim), "ln_weight_exp_avg");
    allocate(&ln_weight_exp_avg_sq, sizeof(float) * static_cast<std::size_t>(kDim), "ln_weight_exp_avg_sq");
    allocate(&ln_bias_exp_avg, sizeof(float) * static_cast<std::size_t>(kDim), "ln_bias_exp_avg");
    allocate(&ln_bias_exp_avg_sq, sizeof(float) * static_cast<std::size_t>(kDim), "ln_bias_exp_avg_sq");
    allocate(&token_out, sizeof(float) * static_cast<std::size_t>(activation_elements), "token_out");
    allocate(&position_out, sizeof(float) * static_cast<std::size_t>(activation_elements), "position_out");
    allocate(&residual, sizeof(float) * static_cast<std::size_t>(activation_elements), "residual");
    allocate(&ln_out, sizeof(float) * static_cast<std::size_t>(activation_elements), "ln_out");
    allocate(&logits, sizeof(float) * static_cast<std::size_t>(logit_elements), "logits");
    allocate(&loss_partials, sizeof(float), "loss_partials");
    allocate(&row_max, sizeof(float) * static_cast<std::size_t>(max_rows), "row_max");
    allocate(&row_denom, sizeof(float) * static_cast<std::size_t>(max_rows), "row_denom");
    allocate(&grad_logits, sizeof(float) * static_cast<std::size_t>(logit_elements), "grad_logits");
    allocate(&grad_ln, sizeof(float) * static_cast<std::size_t>(activation_elements), "grad_ln");
    allocate(&grad_residual, sizeof(float) * static_cast<std::size_t>(activation_elements), "grad_residual");
    allocate(&residual_scale, sizeof(float), "residual_scale");
    allocate(&token_ids, sizeof(std::int64_t) * static_cast<std::size_t>(max_rows), "token_ids");
    allocate(&targets, sizeof(std::int64_t) * static_cast<std::size_t>(max_rows), "targets");

    auto run = [&](int status, const std::string& name) {
        if (status != 0 && error.empty()) {
            error = cuda_error(status, name);
        }
    };
    if (error.empty()) {
        run(fill(token_weight, token_weight_elements, kInitialTokenWeight, nullptr), "token_weight.fill");
        run(fill(position_weight, position_weight_elements, kInitialPositionWeight, nullptr), "position_weight.fill");
        run(fill(ln_weight, kDim, kInitialLnWeight, nullptr), "ln_weight.fill");
        run(fill(ln_bias, kDim, kInitialLnBias, nullptr), "ln_bias.fill");
        run(fill(grad_token_weight, token_weight_elements, 0.0f, nullptr), "grad_token_weight.zero");
        run(fill(grad_position_weight, position_weight_elements, 0.0f, nullptr), "grad_position_weight.zero");
        run(fill(grad_ln_weight, kDim, 0.0f, nullptr), "grad_ln_weight.zero");
        run(fill(grad_ln_bias, kDim, 0.0f, nullptr), "grad_ln_bias.zero");
        run(fill(token_exp_avg, token_weight_elements, 0.0f, nullptr), "token_exp_avg.zero");
        run(fill(token_exp_avg_sq, token_weight_elements, 0.0f, nullptr), "token_exp_avg_sq.zero");
        run(fill(position_exp_avg, position_weight_elements, 0.0f, nullptr), "position_exp_avg.zero");
        run(fill(position_exp_avg_sq, position_weight_elements, 0.0f, nullptr), "position_exp_avg_sq.zero");
        run(fill(ln_weight_exp_avg, kDim, 0.0f, nullptr), "ln_weight_exp_avg.zero");
        run(fill(ln_weight_exp_avg_sq, kDim, 0.0f, nullptr), "ln_weight_exp_avg_sq.zero");
        run(fill(ln_bias_exp_avg, kDim, 0.0f, nullptr), "ln_bias_exp_avg.zero");
        run(fill(ln_bias_exp_avg_sq, kDim, 0.0f, nullptr), "ln_bias_exp_avg_sq.zero");
        run(fill(residual_scale, 1, kResidualScale, nullptr), "residual_scale.fill");
    }

    neuralfn::native_train::SequentialTokenBatchSampler sampler(dataset.train_shards, seq_len, train_batch_size);
    neuralfn::native_train::SequentialTokenBatchSampler val_sampler(dataset.val_shards, seq_len, eval_batch_size);
    neuralfn::native_train::TokenBatch batch;
    neuralfn::native_train::TokenBatch val_batch;
    std::vector<std::int64_t> host_tokens(static_cast<std::size_t>(max_rows), 0);
    std::vector<std::int64_t> host_targets(static_cast<std::size_t>(max_rows), 0);
    std::vector<float> host_loss(1, 0.0f);

    auto load_batch = [&](const neuralfn::native_train::TokenBatch& source, std::int64_t row_count, const std::string& label) {
        if (!error.empty()) {
            return;
        }
        if (static_cast<std::int64_t>(source.tokens.size()) != row_count ||
            static_cast<std::int64_t>(source.targets.size()) != row_count) {
            std::ostringstream out;
            out << label << " token batch has wrong size: tokens=" << source.tokens.size()
                << " targets=" << source.targets.size() << " expected=" << row_count;
            error = out.str();
            return;
        }
        for (std::int64_t i = 0; i < row_count; ++i) {
            const std::uint16_t token = source.tokens[static_cast<std::size_t>(i)];
            const std::uint16_t target = source.targets[static_cast<std::size_t>(i)];
            if (token >= kVocab || target >= kVocab) {
                std::ostringstream out;
                out << label << " token/target id exceeds GPT-2 vocab at row " << i
                    << ": token=" << token << " target=" << target << " vocab=" << kVocab;
                error = out.str();
                return;
            }
            host_tokens[static_cast<std::size_t>(i)] = static_cast<std::int64_t>(token);
            host_targets[static_cast<std::size_t>(i)] = static_cast<std::int64_t>(target);
        }
        run(cuda_memcpy(
                token_ids,
                host_tokens.data(),
                sizeof(std::int64_t) * static_cast<std::size_t>(row_count),
                kCudaMemcpyHostToDevice),
            label + ".cudaMemcpy token_ids");
        run(cuda_memcpy(
                targets,
                host_targets.data(),
                sizeof(std::int64_t) * static_cast<std::size_t>(row_count),
                kCudaMemcpyHostToDevice),
            label + ".cudaMemcpy targets");
    };

    auto forward_loss = [&](const neuralfn::native_train::TokenBatch& source, std::int64_t row_count, const std::string& label)
        -> double {
        load_batch(source, row_count, label);
        if (error.empty()) {
            run(token_embedding(token_weight, token_ids, token_out, row_count, kDim, nullptr), label + ".wte.forward");
        }
        if (error.empty()) {
            const std::int64_t batch_size_for_rows = row_count / seq_len;
            run(position_embedding(position_weight, position_out, batch_size_for_rows, seq_len, kDim, nullptr),
                label + ".wpe.forward");
        }
        if (error.empty()) {
            run(residual_add(token_out, position_out, residual_scale, residual, row_count * kDim, nullptr),
                label + ".embedding_residual.forward");
        }
        if (error.empty()) {
            run(layer_norm(residual, ln_weight, ln_bias, ln_out, row_count, kDim, kNormEps, nullptr),
                label + ".ln_f.forward");
        }
        if (error.empty()) {
            run(linear(ln_out, token_weight, nullptr, logits, row_count, kDim, kPaddedVocab, false, nullptr),
                label + ".lm_head.forward");
        }
        if (error.empty()) {
            run(ce_partials(logits, targets, loss_partials, row_count, kPaddedVocab, nullptr),
                label + ".token_cross_entropy.partials");
        }
        if (error.empty()) {
            run(cuda_device_synchronize(), label + ".cudaDeviceSynchronize");
        }
        if (error.empty()) {
            run(cuda_memcpy(host_loss.data(), loss_partials, sizeof(float), kCudaMemcpyDeviceToHost),
                label + ".cudaMemcpy loss");
        }
        return error.empty() ? static_cast<double>(host_loss[0]) : 0.0;
    };

    auto run_validation = [&](std::int64_t step) {
        if (!error.empty() || cfg.eval_every_steps <= 0 || cfg.eval_batches <= 0) {
            return;
        }
        ValidationLossRecord record;
        record.step = step;
        for (std::int64_t batch_index = 0; batch_index < cfg.eval_batches; ++batch_index) {
            if (!val_sampler.next(val_batch)) {
                val_sampler.reset();
                if (!val_sampler.next(val_batch)) {
                    error = "not enough validation tokens to build one GPT-2 embedding/LM validation batch";
                    break;
                }
            }
            const double loss_sum = forward_loss(val_batch, eval_rows, "validation");
            if (!error.empty()) {
                break;
            }
            record.batches += 1;
            record.tokens += eval_rows;
            record.loss_sum += loss_sum;
        }
        if (error.empty() && record.batches > 0 && record.tokens > 0) {
            record.loss_mean = record.loss_sum / static_cast<double>(record.tokens);
            validation_losses.push_back(record);
        }
    };

    for (std::int64_t step = 1; step <= cfg.max_steps && error.empty(); ++step) {
        if (!sampler.next(batch)) {
            sampler.reset();
            epochs_completed += 1;
            if (!sampler.next(batch)) {
                error = "not enough train tokens to build one GPT-2 embedding/LM train batch";
                break;
            }
        }
        const double train_loss_sum = forward_loss(batch, train_rows, "train");
        if (error.empty()) {
            run(fill(grad_token_weight, token_weight_elements, 0.0f, nullptr), "grad_token_weight.zero");
            run(fill(grad_position_weight, position_weight_elements, 0.0f, nullptr), "grad_position_weight.zero");
            run(fill(grad_ln_weight, kDim, 0.0f, nullptr), "grad_ln_weight.zero");
            run(fill(grad_ln_bias, kDim, 0.0f, nullptr), "grad_ln_bias.zero");
        }
        if (error.empty()) {
            run(ce_backward_workspace(
                    logits,
                    targets,
                    row_max,
                    row_denom,
                    grad_logits,
                    train_rows,
                    kPaddedVocab,
                    1.0f / static_cast<float>(train_rows),
                    nullptr),
                "token_cross_entropy.backward");
        }
        if (error.empty()) {
            run(linear_backward_input(grad_logits, token_weight, grad_ln, train_rows, kDim, kPaddedVocab, nullptr),
                "lm_head.backward_input");
        }
        if (error.empty()) {
            run(linear_backward_weight(ln_out, grad_logits, grad_token_weight, train_rows, kDim, kPaddedVocab, nullptr),
                "lm_head.backward_weight");
        }
        if (error.empty()) {
            run(layer_norm_backward_affine(residual, grad_ln, grad_ln_weight, grad_ln_bias, train_rows, kDim, kNormEps, nullptr),
                "ln_f.backward_affine");
        }
        if (error.empty()) {
            run(layer_norm_backward_input(residual, grad_ln, ln_weight, grad_residual, train_rows, kDim, kNormEps, nullptr),
                "ln_f.backward_input");
        }
        if (error.empty()) {
            run(token_embedding_backward_weight(token_ids, grad_residual, grad_token_weight, train_rows, kDim, nullptr),
                "wte.backward_weight");
        }
        if (error.empty()) {
            run(position_embedding_backward(grad_residual, grad_position_weight, train_batch_size, seq_len, kDim, nullptr),
                "wpe.backward_weight");
        }
        const float bias_correction1 = 1.0f - std::pow(kBeta1, static_cast<float>(step));
        const float sqrt_bias_correction2 = std::sqrt(1.0f - std::pow(kBeta2, static_cast<float>(step)));
        if (error.empty()) {
            run(adamw(token_weight, grad_token_weight, token_exp_avg, token_exp_avg_sq, token_weight_elements, static_cast<float>(cfg.learning_rate), kBeta1, kBeta2, kAdamEps, static_cast<float>(cfg.weight_decay), bias_correction1, sqrt_bias_correction2, nullptr), "wte.adamw");
            run(adamw(position_weight, grad_position_weight, position_exp_avg, position_exp_avg_sq, position_weight_elements, static_cast<float>(cfg.learning_rate), kBeta1, kBeta2, kAdamEps, static_cast<float>(cfg.weight_decay), bias_correction1, sqrt_bias_correction2, nullptr), "wpe.adamw");
            run(adamw(ln_weight, grad_ln_weight, ln_weight_exp_avg, ln_weight_exp_avg_sq, kDim, static_cast<float>(cfg.learning_rate), kBeta1, kBeta2, kAdamEps, static_cast<float>(cfg.weight_decay), bias_correction1, sqrt_bias_correction2, nullptr), "ln_f.weight.adamw");
            run(adamw(ln_bias, grad_ln_bias, ln_bias_exp_avg, ln_bias_exp_avg_sq, kDim, static_cast<float>(cfg.learning_rate), kBeta1, kBeta2, kAdamEps, 0.0f, bias_correction1, sqrt_bias_correction2, nullptr), "ln_f.bias.adamw");
        }
        if (error.empty()) {
            run(cuda_device_synchronize(), "train.cudaDeviceSynchronize");
        }
        if (error.empty()) {
            steps_completed = step;
            tokens_processed += train_rows;
            final_loss_sum = train_loss_sum;
            final_loss_mean = final_loss_sum / static_cast<double>(train_rows);
            if (cfg.eval_every_steps > 0 && (step % cfg.eval_every_steps) == 0) {
                run_validation(step);
            }
        }
    }

    passed = error.empty() && steps_completed == cfg.max_steps && std::isfinite(final_loss_mean);

    auto free_device = [&](void* ptr, const std::string& name) {
        if (ptr != nullptr && cuda_free != nullptr) {
            const int status = cuda_free(ptr);
            if (status != 0 && error.empty()) {
                error = cuda_error(status, "cudaFree " + name);
            }
        }
    };
    free_device(token_weight, "token_weight");
    free_device(position_weight, "position_weight");
    free_device(ln_weight, "ln_weight");
    free_device(ln_bias, "ln_bias");
    free_device(grad_token_weight, "grad_token_weight");
    free_device(grad_position_weight, "grad_position_weight");
    free_device(grad_ln_weight, "grad_ln_weight");
    free_device(grad_ln_bias, "grad_ln_bias");
    free_device(token_exp_avg, "token_exp_avg");
    free_device(token_exp_avg_sq, "token_exp_avg_sq");
    free_device(position_exp_avg, "position_exp_avg");
    free_device(position_exp_avg_sq, "position_exp_avg_sq");
    free_device(ln_weight_exp_avg, "ln_weight_exp_avg");
    free_device(ln_weight_exp_avg_sq, "ln_weight_exp_avg_sq");
    free_device(ln_bias_exp_avg, "ln_bias_exp_avg");
    free_device(ln_bias_exp_avg_sq, "ln_bias_exp_avg_sq");
    free_device(token_out, "token_out");
    free_device(position_out, "position_out");
    free_device(residual, "residual");
    free_device(ln_out, "ln_out");
    free_device(logits, "logits");
    free_device(loss_partials, "loss_partials");
    free_device(row_max, "row_max");
    free_device(row_denom, "row_denom");
    free_device(grad_logits, "grad_logits");
    free_device(grad_ln, "grad_ln");
    free_device(grad_residual, "grad_residual");
    free_device(residual_scale, "residual_scale");
    free_device(token_ids, "token_ids");
    free_device(targets, "targets");
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(cfg.model_family) << "\",\n"
        << "  \"backend\": \"tile-cuda\",\n"
        << "  \"status\": \"" << (passed ? "native-embedding-lm-trained" : "native-embedding-lm-failed") << "\",\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"token_shards\": " << neuralfn::native_train::token_shard_dataset_json(dataset, &batch_plan) << ",\n"
        << "  \"batch_size\": " << train_batch_size << ",\n"
        << "  \"eval_batch_size\": " << eval_batch_size << ",\n"
        << "  \"seq_len\": " << seq_len << ",\n"
        << "  \"rows\": " << train_rows << ",\n"
        << "  \"eval_rows\": " << eval_rows << ",\n"
        << "  \"vocab\": " << kVocab << ",\n"
        << "  \"padded_vocab\": " << kPaddedVocab << ",\n"
        << "  \"model_dim\": " << kDim << ",\n"
        << "  \"max_steps\": " << cfg.max_steps << ",\n"
        << "  \"eval_every_steps\": " << cfg.eval_every_steps << ",\n"
        << "  \"eval_batches\": " << cfg.eval_batches << ",\n"
        << "  \"steps_completed\": " << steps_completed << ",\n"
        << "  \"epochs_completed\": " << epochs_completed << ",\n"
        << "  \"tokens_processed\": " << tokens_processed << ",\n"
        << "  \"learning_rate\": " << cfg.learning_rate << ",\n"
        << "  \"weight_decay\": " << cfg.weight_decay << ",\n"
        << "  \"final_loss_sum\": " << final_loss_sum << ",\n"
        << "  \"final_loss_mean\": " << final_loss_mean << ",\n"
        << "  \"validation\": {\n"
        << "    \"eval_every_steps\": " << cfg.eval_every_steps << ",\n"
        << "    \"eval_batches\": " << cfg.eval_batches << ",\n"
        << "    \"eval_batch_size\": " << eval_batch_size << ",\n"
        << "    \"eval_count\": " << validation_losses.size() << ",\n"
        << "    \"losses\": [\n";
    for (std::size_t i = 0; i < validation_losses.size(); ++i) {
        const ValidationLossRecord& record = validation_losses[i];
        std::cout
            << "      {\"step\": " << record.step
            << ", \"batches\": " << record.batches
            << ", \"tokens\": " << record.tokens
            << ", \"loss_sum\": " << record.loss_sum
            << ", \"loss_mean\": " << record.loss_mean << "}";
        if (i + 1 != validation_losses.size()) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout
        << "    ]\n"
        << "  },\n"
        << "  \"kernels\": [\n"
        << "    \"nfn_native_tile_fill_float32\",\n"
        << "    \"nfn_native_tile_token_embedding_float32\",\n"
        << "    \"nfn_native_tile_absolute_position_embedding_float32\",\n"
        << "    \"nfn_native_tile_scaled_residual_add_float32\",\n"
        << "    \"nfn_native_tile_layer_norm_float32\",\n"
        << "    \"nfn_native_tile_linear_float32\",\n"
        << "    \"nfn_native_tile_token_cross_entropy_partials_float32\",\n"
        << "    \"nfn_native_tile_token_cross_entropy_backward_with_workspace_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_input_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_weight_float32\",\n"
        << "    \"nfn_native_tile_layer_norm_backward_affine_float32\",\n"
        << "    \"nfn_native_tile_layer_norm_backward_input_float32\",\n"
        << "    \"nfn_native_tile_token_embedding_backward_weight_float32\",\n"
        << "    \"nfn_native_tile_absolute_position_embedding_backward_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";

    return passed ? 0 : 2;
}

int print_attention_step_smoke_json(const Config& cfg, const char* program) {
    constexpr std::int64_t kBatch = 1;
    constexpr std::int64_t kSeq = 2;
    constexpr std::int64_t kHeads = 1;
    constexpr std::int64_t kDim = 768;
    constexpr std::int64_t kRows = kBatch * kSeq;
    constexpr std::int64_t kQkvDim = kDim * 3;
    constexpr std::int64_t kActivationElements = kRows * kDim;
    constexpr std::int64_t kQkvActivationElements = kRows * kQkvDim;
    constexpr std::int64_t kQkvWeightElements = kQkvDim * kDim;
    constexpr std::int64_t kOutWeightElements = kDim * kDim;
    constexpr float kInputValue = 0.01f;
    constexpr float kQWeight = 0.02f;
    constexpr float kKWeight = 0.01f;
    constexpr float kVWeight = -0.015f;
    constexpr float kOutWeight = 0.005f;
    constexpr float kGradFinal = 0.25f;
    constexpr float kLearningRate = 0.1f;
    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.95f;
    constexpr float kEps = 1e-8f;
    constexpr float kWeightDecay = 0.1f;
    constexpr float kBiasCorrection1 = 1.0f - kBeta1;
    constexpr float kSqrtBiasCorrection2 = 0.22360679775f;
    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;

    const std::string tile_lib_path = cfg.tile_ops_lib.empty() ? default_tile_ops_lib(program) : cfg.tile_ops_lib;
    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(cfg);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool passed = false;
    double max_abs_sample = 0.0;
    double max_weight_delta = 0.0;
    std::string error;

    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using SplitQkvFn = int (*)(
        const float*, float*, float*, float*, std::int64_t, std::int64_t, void*);
    using MergeQkvFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, void*);
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBackwardInputFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using AttentionFn = int (*)(
        const float*,
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
    using AttentionBackwardFn = int (*)(
        const float*,
        const float*,
        const float*,
        const float*,
        float*,
        float*,
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
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);

    void* tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    void* cuda_handle = nullptr;
    FillFn fill = nullptr;
    SplitQkvFn split_qkv = nullptr;
    MergeQkvFn merge_qkv = nullptr;
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightFn linear_backward_weight = nullptr;
    AttentionFn attention = nullptr;
    AttentionBackwardFn attention_backward = nullptr;
    AdamWFn adamw = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed with CUDA error " << code;
        if (cuda_get_error_string != nullptr) {
            const char* message = cuda_get_error_string(code);
            if (message != nullptr) {
                out << ": " << message;
            }
        }
        return out.str();
    };

    if (tile_handle == nullptr) {
        error = dl_last_error("dlopen tile ops failed");
    } else {
        tile_loaded = true;
        fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
        split_qkv = load_symbol<SplitQkvFn>(tile_handle, "nfn_native_tile_split_qkv_float32");
        merge_qkv = load_symbol<MergeQkvFn>(tile_handle, "nfn_native_tile_merge_qkv_float32");
        linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
        linear_backward_input = load_symbol<LinearBackwardInputFn>(
            tile_handle, "nfn_native_tile_linear_backward_input_float32");
        linear_backward_weight = load_symbol<LinearBackwardWeightFn>(
            tile_handle, "nfn_native_tile_linear_backward_weight_float32");
        attention = load_symbol<AttentionFn>(tile_handle, "nfn_native_tile_scaled_dot_product_attention_float32");
        attention_backward = load_symbol<AttentionBackwardFn>(
            tile_handle, "nfn_native_tile_scaled_dot_product_attention_backward_float32");
        adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
        if (fill == nullptr || split_qkv == nullptr || merge_qkv == nullptr || linear == nullptr ||
            linear_backward_input == nullptr || linear_backward_weight == nullptr || attention == nullptr ||
            attention_backward == nullptr || adamw == nullptr) {
            error = dl_last_error("dlsym GPT-2 attention-step kernels failed");
        }
    }

    if (error.empty()) {
        std::string runtime_error;
        for (const std::string& candidate : runtime_candidates) {
            cuda_lib_path = candidate;
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_runtime_loaded = true;
                break;
            }
            runtime_error = dl_last_error("dlopen CUDA runtime failed");
        }
        if (cuda_handle == nullptr) {
            error = runtime_error.empty() ? "could not load CUDA runtime" : runtime_error;
        }
    }

    if (error.empty()) {
        cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
        cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
        cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
        cuda_device_synchronize = load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
        cuda_get_error_string = load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
        if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr || cuda_device_synchronize == nullptr) {
            error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaDeviceSynchronize";
        }
    }

    std::vector<float> host_qkv_weight(static_cast<std::size_t>(kQkvWeightElements), 0.0f);
    for (std::int64_t row = 0; row < kQkvDim; ++row) {
        const float value = row < kDim ? kQWeight : (row < 2 * kDim ? kKWeight : kVWeight);
        const std::int64_t row_offset = row * kDim;
        for (std::int64_t col = 0; col < kDim; ++col) {
            host_qkv_weight[static_cast<std::size_t>(row_offset + col)] = value;
        }
    }

    float* x = nullptr;
    float* qkv_weight = nullptr;
    float* out_weight = nullptr;
    float* grad_qkv_weight = nullptr;
    float* grad_out_weight = nullptr;
    float* qkv_weight_exp_avg = nullptr;
    float* qkv_weight_exp_avg_sq = nullptr;
    float* out_weight_exp_avg = nullptr;
    float* out_weight_exp_avg_sq = nullptr;
    float* qkv = nullptr;
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* q_heads = nullptr;
    float* k_heads = nullptr;
    float* v_heads = nullptr;
    float* attn_heads = nullptr;
    float* attn_out = nullptr;
    float* out = nullptr;
    float* grad_out = nullptr;
    float* grad_attn = nullptr;
    float* grad_q = nullptr;
    float* grad_k = nullptr;
    float* grad_v = nullptr;
    float* grad_qkv = nullptr;
    float* grad_x = nullptr;

    auto allocate = [&](float** ptr, std::int64_t elements, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        const int status = cuda_malloc(
            reinterpret_cast<void**>(ptr),
            sizeof(float) * static_cast<std::size_t>(elements));
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc " + name);
        }
    };
    allocate(&x, kActivationElements, "x");
    allocate(&qkv_weight, kQkvWeightElements, "qkv_weight");
    allocate(&out_weight, kOutWeightElements, "out_weight");
    allocate(&grad_qkv_weight, kQkvWeightElements, "grad_qkv_weight");
    allocate(&grad_out_weight, kOutWeightElements, "grad_out_weight");
    allocate(&qkv_weight_exp_avg, kQkvWeightElements, "qkv_weight_exp_avg");
    allocate(&qkv_weight_exp_avg_sq, kQkvWeightElements, "qkv_weight_exp_avg_sq");
    allocate(&out_weight_exp_avg, kOutWeightElements, "out_weight_exp_avg");
    allocate(&out_weight_exp_avg_sq, kOutWeightElements, "out_weight_exp_avg_sq");
    allocate(&qkv, kQkvActivationElements, "qkv");
    allocate(&q, kActivationElements, "q");
    allocate(&k, kActivationElements, "k");
    allocate(&v, kActivationElements, "v");
    allocate(&attn_out, kActivationElements, "attn_out");
    allocate(&out, kActivationElements, "out");
    allocate(&grad_out, kActivationElements, "grad_out");
    allocate(&grad_attn, kActivationElements, "grad_attn");
    allocate(&grad_q, kActivationElements, "grad_q");
    allocate(&grad_k, kActivationElements, "grad_k");
    allocate(&grad_v, kActivationElements, "grad_v");
    allocate(&grad_qkv, kQkvActivationElements, "grad_qkv");
    allocate(&grad_x, kActivationElements, "grad_x");

    auto fill_buffer = [&](float* ptr, std::int64_t elements, float value, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        const int status = fill(ptr, elements, value, nullptr);
        if (status != 0) {
            error = cuda_error(status, "fill " + name);
        }
    };
    fill_buffer(x, kActivationElements, kInputValue, "x");
    fill_buffer(out_weight, kOutWeightElements, kOutWeight, "out_weight");
    fill_buffer(grad_qkv_weight, kQkvWeightElements, 0.0f, "grad_qkv_weight");
    fill_buffer(grad_out_weight, kOutWeightElements, 0.0f, "grad_out_weight");
    fill_buffer(qkv_weight_exp_avg, kQkvWeightElements, 0.0f, "qkv_weight_exp_avg");
    fill_buffer(qkv_weight_exp_avg_sq, kQkvWeightElements, 0.0f, "qkv_weight_exp_avg_sq");
    fill_buffer(out_weight_exp_avg, kOutWeightElements, 0.0f, "out_weight_exp_avg");
    fill_buffer(out_weight_exp_avg_sq, kOutWeightElements, 0.0f, "out_weight_exp_avg_sq");
    fill_buffer(grad_out, kActivationElements, kGradFinal, "grad_out");

    if (error.empty()) {
        const int status = cuda_memcpy(
            qkv_weight,
            host_qkv_weight.data(),
            sizeof(float) * static_cast<std::size_t>(kQkvWeightElements),
            kCudaMemcpyHostToDevice);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy host-to-device qkv_weight");
        }
    }

    auto run = [&](int status, const std::string& name) {
        if (status != 0 && error.empty()) {
            error = cuda_error(status, name);
        }
    };
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(kDim));
    if (error.empty()) {
        run(linear(x, qkv_weight, nullptr, qkv, kRows, kDim, kQkvDim, false, nullptr), "qkv linear");
    }
    if (error.empty()) {
        run(split_qkv(qkv, q, k, v, kRows, kDim, nullptr), "split_qkv");
    }
    if (error.empty()) {
        run(attention(
                q,
                k,
                v,
                attn_out,
                kActivationElements,
                kHeads,
                kHeads,
                kSeq,
                kSeq,
                kDim,
                kDim,
                attention_scale,
                false,
                false,
                false,
                0,
                0,
                0,
                0,
                nullptr),
            "scaled_dot_product_attention");
    }
    if (error.empty()) {
        run(linear(attn_out, out_weight, nullptr, out, kRows, kDim, kDim, false, nullptr), "out linear");
    }
    if (error.empty()) {
        run(linear_backward_weight(attn_out, grad_out, grad_out_weight, kRows, kDim, kDim, nullptr),
            "out linear_backward_weight");
    }
    if (error.empty()) {
        run(linear_backward_input(grad_out, out_weight, grad_attn, kRows, kDim, kDim, nullptr),
            "out linear_backward_input");
    }
    if (error.empty()) {
        run(attention_backward(
                q,
                k,
                v,
                grad_attn,
                grad_q,
                grad_k,
                grad_v,
                kBatch,
                kHeads,
                kHeads,
                kSeq,
                kSeq,
                kDim,
                kDim,
                attention_scale,
                false,
                false,
                false,
                0,
                0,
                0,
                0,
                nullptr),
            "scaled_dot_product_attention_backward");
    }
    if (error.empty()) {
        run(merge_qkv(grad_q, grad_k, grad_v, grad_qkv, kRows, kDim, nullptr), "merge_qkv");
    }
    if (error.empty()) {
        run(linear_backward_weight(x, grad_qkv, grad_qkv_weight, kRows, kDim, kQkvDim, nullptr),
            "qkv linear_backward_weight");
    }
    if (error.empty()) {
        run(linear_backward_input(grad_qkv, qkv_weight, grad_x, kRows, kDim, kQkvDim, nullptr),
            "qkv linear_backward_input");
    }
    if (error.empty()) {
        run(adamw(
                qkv_weight,
                grad_qkv_weight,
                qkv_weight_exp_avg,
                qkv_weight_exp_avg_sq,
                kQkvWeightElements,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                kWeightDecay,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr),
            "adamw qkv_weight");
    }
    if (error.empty()) {
        run(adamw(
                out_weight,
                grad_out_weight,
                out_weight_exp_avg,
                out_weight_exp_avg_sq,
                kOutWeightElements,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                kWeightDecay,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr),
            "adamw out_weight");
    }
    if (error.empty()) {
        run(cuda_device_synchronize(), "cudaDeviceSynchronize");
    }

    auto copy_float_sample = [&](const float* device_ptr, std::int64_t offset, const std::string& name) -> float {
        float value = 0.0f;
        if (!error.empty()) {
            return value;
        }
        const int status = cuda_memcpy(&value, device_ptr + offset, sizeof(float), kCudaMemcpyDeviceToHost);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy sample " + name);
        }
        return value;
    };
    float sampled_q = 0.0f;
    float sampled_attn = 0.0f;
    float sampled_out = 0.0f;
    float sampled_grad_x = 0.0f;
    float sampled_grad_qkv_weight = 0.0f;
    float sampled_grad_out_weight = 0.0f;
    float sampled_qkv_weight = 0.0f;
    float sampled_out_weight = 0.0f;
    if (error.empty()) {
        sampled_q = copy_float_sample(q, 0, "q");
        sampled_attn = copy_float_sample(attn_out, 0, "attn_out");
        sampled_out = copy_float_sample(out, 0, "out");
        sampled_grad_x = copy_float_sample(grad_x, 0, "grad_x");
        sampled_grad_qkv_weight = copy_float_sample(grad_qkv_weight, 0, "grad_qkv_weight");
        sampled_grad_out_weight = copy_float_sample(grad_out_weight, 0, "grad_out_weight");
        sampled_qkv_weight = copy_float_sample(qkv_weight, 0, "qkv_weight");
        sampled_out_weight = copy_float_sample(out_weight, 0, "out_weight");
    }

    auto record_abs = [&](float value) {
        if (!std::isfinite(value)) {
            return false;
        }
        const double abs_value = std::fabs(static_cast<double>(value));
        if (abs_value > max_abs_sample) {
            max_abs_sample = abs_value;
        }
        return true;
    };
    if (error.empty()) {
        const bool finite_samples =
            record_abs(sampled_q) && record_abs(sampled_attn) && record_abs(sampled_out) &&
            record_abs(sampled_grad_x) && record_abs(sampled_grad_qkv_weight) &&
            record_abs(sampled_grad_out_weight) && record_abs(sampled_qkv_weight) &&
            record_abs(sampled_out_weight);
        max_weight_delta = std::fabs(static_cast<double>(sampled_qkv_weight) - kQWeight);
        const double out_weight_delta = std::fabs(static_cast<double>(sampled_out_weight) - kOutWeight);
        if (out_weight_delta > max_weight_delta) {
            max_weight_delta = out_weight_delta;
        }
        passed = finite_samples && max_abs_sample > 0.0 && max_weight_delta > 0.0;
        if (!passed) {
            std::ostringstream out_message;
            out_message << "GPT-2 attention smoke did not produce finite nonzero samples and weight updates: sample="
                        << max_abs_sample << " weight_delta=" << max_weight_delta;
            error = out_message.str();
        }
    }

    auto free_device = [&](void* ptr, const std::string& name) {
        if (ptr != nullptr && cuda_free != nullptr) {
            const int status = cuda_free(ptr);
            if (status != 0 && error.empty()) {
                error = cuda_error(status, "cudaFree " + name);
            }
        }
    };
    free_device(x, "x");
    free_device(qkv_weight, "qkv_weight");
    free_device(out_weight, "out_weight");
    free_device(grad_qkv_weight, "grad_qkv_weight");
    free_device(grad_out_weight, "grad_out_weight");
    free_device(qkv_weight_exp_avg, "qkv_weight_exp_avg");
    free_device(qkv_weight_exp_avg_sq, "qkv_weight_exp_avg_sq");
    free_device(out_weight_exp_avg, "out_weight_exp_avg");
    free_device(out_weight_exp_avg_sq, "out_weight_exp_avg_sq");
    free_device(qkv, "qkv");
    free_device(q, "q");
    free_device(k, "k");
    free_device(v, "v");
    free_device(q_heads, "q_heads");
    free_device(k_heads, "k_heads");
    free_device(v_heads, "v_heads");
    free_device(attn_heads, "attn_heads");
    free_device(attn_out, "attn_out");
    free_device(out, "out");
    free_device(grad_out, "grad_out");
    free_device(grad_attn, "grad_attn");
    free_device(grad_q, "grad_q");
    free_device(grad_k, "grad_k");
    free_device(grad_v, "grad_v");
    free_device(grad_qkv, "grad_qkv");
    free_device(grad_x, "grad_x");
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(cfg.model_family) << "\",\n"
        << "  \"backend\": \"tile-cuda\",\n"
        << "  \"smoke\": \"attention_step\",\n"
        << "  \"token_shards_resolved\": false,\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"batch\": " << kBatch << ",\n"
        << "  \"seq\": " << kSeq << ",\n"
        << "  \"heads\": " << kHeads << ",\n"
        << "  \"model_dim\": " << kDim << ",\n"
        << "  \"kernels\": [\n"
        << "    \"nfn_native_tile_linear_float32\",\n"
        << "    \"nfn_native_tile_split_qkv_float32\",\n"
        << "    \"nfn_native_tile_scaled_dot_product_attention_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_weight_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_bias_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_input_float32\",\n"
        << "    \"nfn_native_tile_scaled_dot_product_attention_backward_float32\",\n"
        << "    \"nfn_native_tile_merge_qkv_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"sample_q\": " << sampled_q << ",\n"
        << "  \"sample_attn_out\": " << sampled_attn << ",\n"
        << "  \"sample_out\": " << sampled_out << ",\n"
        << "  \"sample_grad_x\": " << sampled_grad_x << ",\n"
        << "  \"sample_grad_qkv_weight\": " << sampled_grad_qkv_weight << ",\n"
        << "  \"sample_grad_out_weight\": " << sampled_grad_out_weight << ",\n"
        << "  \"sample_updated_qkv_weight\": " << sampled_qkv_weight << ",\n"
        << "  \"sample_updated_out_weight\": " << sampled_out_weight << ",\n"
        << "  \"max_abs_sample\": " << max_abs_sample << ",\n"
        << "  \"max_weight_delta\": " << max_weight_delta << ",\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";

    return passed ? 0 : 2;
}

int print_mlp_step_smoke_json(const Config& cfg, const char* program) {
    constexpr std::int64_t kRows = 2;
    constexpr std::int64_t kDim = 768;
    constexpr std::int64_t kHidden = kDim * 4;
    constexpr std::int64_t kActivationElements = kRows * kDim;
    constexpr std::int64_t kHiddenElements = kRows * kHidden;
    constexpr std::int64_t kFcWeightElements = kHidden * kDim;
    constexpr std::int64_t kProjWeightElements = kDim * kHidden;
    constexpr float kInputValue = 0.01f;
    constexpr float kFcWeight = 0.02f;
    constexpr float kProjWeight = 0.005f;
    constexpr float kGradFinal = 0.25f;
    constexpr float kLearningRate = 0.1f;
    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.95f;
    constexpr float kEps = 1e-8f;
    constexpr float kWeightDecay = 0.1f;
    constexpr float kBiasCorrection1 = 1.0f - kBeta1;
    constexpr float kSqrtBiasCorrection2 = 0.22360679775f;
    constexpr int kCudaMemcpyDeviceToHost = 2;

    const std::string tile_lib_path = cfg.tile_ops_lib.empty() ? default_tile_ops_lib(program) : cfg.tile_ops_lib;
    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(cfg);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool passed = false;
    double max_abs_sample = 0.0;
    double max_weight_delta = 0.0;
    std::string error;

    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBackwardInputFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using GeluFn = int (*)(const float*, float*, std::int64_t, void*);
    using GeluBackwardFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
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
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);

    void* tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    void* cuda_handle = nullptr;
    FillFn fill = nullptr;
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightFn linear_backward_weight = nullptr;
    GeluFn gelu = nullptr;
    GeluBackwardFn gelu_backward = nullptr;
    AdamWFn adamw = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed with CUDA error " << code;
        if (cuda_get_error_string != nullptr) {
            const char* message = cuda_get_error_string(code);
            if (message != nullptr) {
                out << ": " << message;
            }
        }
        return out.str();
    };

    if (tile_handle == nullptr) {
        error = dl_last_error("dlopen tile ops failed");
    } else {
        tile_loaded = true;
        fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
        linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
        linear_backward_input = load_symbol<LinearBackwardInputFn>(
            tile_handle, "nfn_native_tile_linear_backward_input_float32");
        linear_backward_weight = load_symbol<LinearBackwardWeightFn>(
            tile_handle, "nfn_native_tile_linear_backward_weight_float32");
        gelu = load_symbol<GeluFn>(tile_handle, "nfn_native_tile_gelu_float32");
        gelu_backward = load_symbol<GeluBackwardFn>(tile_handle, "nfn_native_tile_gelu_backward_float32");
        adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
        if (fill == nullptr || linear == nullptr || linear_backward_input == nullptr ||
            linear_backward_weight == nullptr || gelu == nullptr || gelu_backward == nullptr || adamw == nullptr) {
            error = dl_last_error("dlsym GPT-2 MLP-step kernels failed");
        }
    }

    if (error.empty()) {
        std::string runtime_error;
        for (const std::string& candidate : runtime_candidates) {
            cuda_lib_path = candidate;
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_runtime_loaded = true;
                break;
            }
            runtime_error = dl_last_error("dlopen CUDA runtime failed");
        }
        if (cuda_handle == nullptr) {
            error = runtime_error.empty() ? "could not load CUDA runtime" : runtime_error;
        }
    }

    if (error.empty()) {
        cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
        cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
        cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
        cuda_device_synchronize = load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
        cuda_get_error_string = load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
        if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr || cuda_device_synchronize == nullptr) {
            error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaDeviceSynchronize";
        }
    }

    float* x = nullptr;
    float* fc_weight = nullptr;
    float* proj_weight = nullptr;
    float* grad_fc_weight = nullptr;
    float* grad_proj_weight = nullptr;
    float* fc_exp_avg = nullptr;
    float* fc_exp_avg_sq = nullptr;
    float* proj_exp_avg = nullptr;
    float* proj_exp_avg_sq = nullptr;
    float* fc_out = nullptr;
    float* act = nullptr;
    float* out = nullptr;
    float* grad_out = nullptr;
    float* grad_act = nullptr;
    float* grad_fc_out = nullptr;
    float* grad_x = nullptr;

    auto allocate = [&](float** ptr, std::int64_t elements, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        const int status = cuda_malloc(
            reinterpret_cast<void**>(ptr),
            sizeof(float) * static_cast<std::size_t>(elements));
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc " + name);
        }
    };
    allocate(&x, kActivationElements, "x");
    allocate(&fc_weight, kFcWeightElements, "fc_weight");
    allocate(&proj_weight, kProjWeightElements, "proj_weight");
    allocate(&grad_fc_weight, kFcWeightElements, "grad_fc_weight");
    allocate(&grad_proj_weight, kProjWeightElements, "grad_proj_weight");
    allocate(&fc_exp_avg, kFcWeightElements, "fc_exp_avg");
    allocate(&fc_exp_avg_sq, kFcWeightElements, "fc_exp_avg_sq");
    allocate(&proj_exp_avg, kProjWeightElements, "proj_exp_avg");
    allocate(&proj_exp_avg_sq, kProjWeightElements, "proj_exp_avg_sq");
    allocate(&fc_out, kHiddenElements, "fc_out");
    allocate(&act, kHiddenElements, "act");
    allocate(&out, kActivationElements, "out");
    allocate(&grad_out, kActivationElements, "grad_out");
    allocate(&grad_act, kHiddenElements, "grad_act");
    allocate(&grad_fc_out, kHiddenElements, "grad_fc_out");
    allocate(&grad_x, kActivationElements, "grad_x");

    auto fill_buffer = [&](float* ptr, std::int64_t elements, float value, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        const int status = fill(ptr, elements, value, nullptr);
        if (status != 0) {
            error = cuda_error(status, "fill " + name);
        }
    };
    fill_buffer(x, kActivationElements, kInputValue, "x");
    fill_buffer(fc_weight, kFcWeightElements, kFcWeight, "fc_weight");
    fill_buffer(proj_weight, kProjWeightElements, kProjWeight, "proj_weight");
    fill_buffer(grad_fc_weight, kFcWeightElements, 0.0f, "grad_fc_weight");
    fill_buffer(grad_proj_weight, kProjWeightElements, 0.0f, "grad_proj_weight");
    fill_buffer(fc_exp_avg, kFcWeightElements, 0.0f, "fc_exp_avg");
    fill_buffer(fc_exp_avg_sq, kFcWeightElements, 0.0f, "fc_exp_avg_sq");
    fill_buffer(proj_exp_avg, kProjWeightElements, 0.0f, "proj_exp_avg");
    fill_buffer(proj_exp_avg_sq, kProjWeightElements, 0.0f, "proj_exp_avg_sq");
    fill_buffer(grad_out, kActivationElements, kGradFinal, "grad_out");

    auto run = [&](int status, const std::string& name) {
        if (status != 0 && error.empty()) {
            error = cuda_error(status, name);
        }
    };
    if (error.empty()) {
        run(linear(x, fc_weight, nullptr, fc_out, kRows, kDim, kHidden, false, nullptr), "mlp.c_fc.forward");
    }
    if (error.empty()) {
        run(gelu(fc_out, act, kHiddenElements, nullptr), "mlp.gelu.forward");
    }
    if (error.empty()) {
        run(linear(act, proj_weight, nullptr, out, kRows, kHidden, kDim, false, nullptr), "mlp.c_proj.forward");
    }
    if (error.empty()) {
        run(linear_backward_weight(act, grad_out, grad_proj_weight, kRows, kHidden, kDim, nullptr),
            "mlp.c_proj.backward_weight");
    }
    if (error.empty()) {
        run(linear_backward_input(grad_out, proj_weight, grad_act, kRows, kHidden, kDim, nullptr),
            "mlp.c_proj.backward_input");
    }
    if (error.empty()) {
        run(gelu_backward(fc_out, grad_act, grad_fc_out, kHiddenElements, nullptr), "mlp.gelu.backward");
    }
    if (error.empty()) {
        run(linear_backward_weight(x, grad_fc_out, grad_fc_weight, kRows, kDim, kHidden, nullptr),
            "mlp.c_fc.backward_weight");
    }
    if (error.empty()) {
        run(linear_backward_input(grad_fc_out, fc_weight, grad_x, kRows, kDim, kHidden, nullptr),
            "mlp.c_fc.backward_input");
    }
    if (error.empty()) {
        run(adamw(
                fc_weight,
                grad_fc_weight,
                fc_exp_avg,
                fc_exp_avg_sq,
                kFcWeightElements,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                kWeightDecay,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr),
            "mlp.c_fc.adamw");
    }
    if (error.empty()) {
        run(adamw(
                proj_weight,
                grad_proj_weight,
                proj_exp_avg,
                proj_exp_avg_sq,
                kProjWeightElements,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                kWeightDecay,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr),
            "mlp.c_proj.adamw");
    }
    if (error.empty()) {
        run(cuda_device_synchronize(), "cudaDeviceSynchronize");
    }

    auto copy_float_sample = [&](const float* device_ptr, std::int64_t offset, const std::string& name) -> float {
        float value = 0.0f;
        if (!error.empty()) {
            return value;
        }
        const int status = cuda_memcpy(&value, device_ptr + offset, sizeof(float), kCudaMemcpyDeviceToHost);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy sample " + name);
        }
        return value;
    };
    float sampled_fc_out = 0.0f;
    float sampled_act = 0.0f;
    float sampled_out = 0.0f;
    float sampled_grad_x = 0.0f;
    float sampled_grad_fc_weight = 0.0f;
    float sampled_grad_proj_weight = 0.0f;
    float sampled_fc_weight = 0.0f;
    float sampled_proj_weight = 0.0f;
    if (error.empty()) {
        sampled_fc_out = copy_float_sample(fc_out, 0, "fc_out");
        sampled_act = copy_float_sample(act, 0, "act");
        sampled_out = copy_float_sample(out, 0, "out");
        sampled_grad_x = copy_float_sample(grad_x, 0, "grad_x");
        sampled_grad_fc_weight = copy_float_sample(grad_fc_weight, 0, "grad_fc_weight");
        sampled_grad_proj_weight = copy_float_sample(grad_proj_weight, 0, "grad_proj_weight");
        sampled_fc_weight = copy_float_sample(fc_weight, 0, "fc_weight");
        sampled_proj_weight = copy_float_sample(proj_weight, 0, "proj_weight");
    }

    auto record_abs = [&](float value) {
        if (!std::isfinite(value)) {
            return false;
        }
        const double abs_value = std::fabs(static_cast<double>(value));
        if (abs_value > max_abs_sample) {
            max_abs_sample = abs_value;
        }
        return true;
    };
    if (error.empty()) {
        const bool finite_samples =
            record_abs(sampled_fc_out) && record_abs(sampled_act) && record_abs(sampled_out) &&
            record_abs(sampled_grad_x) && record_abs(sampled_grad_fc_weight) &&
            record_abs(sampled_grad_proj_weight) && record_abs(sampled_fc_weight) &&
            record_abs(sampled_proj_weight);
        max_weight_delta = std::fabs(static_cast<double>(sampled_fc_weight) - kFcWeight);
        const double proj_weight_delta = std::fabs(static_cast<double>(sampled_proj_weight) - kProjWeight);
        if (proj_weight_delta > max_weight_delta) {
            max_weight_delta = proj_weight_delta;
        }
        passed = finite_samples && max_abs_sample > 0.0 && max_weight_delta > 0.0;
        if (!passed) {
            std::ostringstream out_message;
            out_message << "GPT-2 MLP smoke did not produce finite nonzero samples and weight updates: sample="
                        << max_abs_sample << " weight_delta=" << max_weight_delta;
            error = out_message.str();
        }
    }

    auto free_device = [&](void* ptr, const std::string& name) {
        if (ptr != nullptr && cuda_free != nullptr) {
            const int status = cuda_free(ptr);
            if (status != 0 && error.empty()) {
                error = cuda_error(status, "cudaFree " + name);
            }
        }
    };
    free_device(x, "x");
    free_device(fc_weight, "fc_weight");
    free_device(proj_weight, "proj_weight");
    free_device(grad_fc_weight, "grad_fc_weight");
    free_device(grad_proj_weight, "grad_proj_weight");
    free_device(fc_exp_avg, "fc_exp_avg");
    free_device(fc_exp_avg_sq, "fc_exp_avg_sq");
    free_device(proj_exp_avg, "proj_exp_avg");
    free_device(proj_exp_avg_sq, "proj_exp_avg_sq");
    free_device(fc_out, "fc_out");
    free_device(act, "act");
    free_device(out, "out");
    free_device(grad_out, "grad_out");
    free_device(grad_act, "grad_act");
    free_device(grad_fc_out, "grad_fc_out");
    free_device(grad_x, "grad_x");
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(cfg.model_family) << "\",\n"
        << "  \"backend\": \"tile-cuda\",\n"
        << "  \"smoke\": \"mlp_step\",\n"
        << "  \"token_shards_resolved\": false,\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"rows\": " << kRows << ",\n"
        << "  \"model_dim\": " << kDim << ",\n"
        << "  \"hidden_dim\": " << kHidden << ",\n"
        << "  \"kernels\": [\n"
        << "    \"nfn_native_tile_linear_float32\",\n"
        << "    \"nfn_native_tile_gelu_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_weight_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_input_float32\",\n"
        << "    \"nfn_native_tile_gelu_backward_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"sample_fc_out\": " << sampled_fc_out << ",\n"
        << "  \"sample_act\": " << sampled_act << ",\n"
        << "  \"sample_out\": " << sampled_out << ",\n"
        << "  \"sample_grad_x\": " << sampled_grad_x << ",\n"
        << "  \"sample_grad_fc_weight\": " << sampled_grad_fc_weight << ",\n"
        << "  \"sample_grad_proj_weight\": " << sampled_grad_proj_weight << ",\n"
        << "  \"sample_updated_fc_weight\": " << sampled_fc_weight << ",\n"
        << "  \"sample_updated_proj_weight\": " << sampled_proj_weight << ",\n"
        << "  \"max_abs_sample\": " << max_abs_sample << ",\n"
        << "  \"max_weight_delta\": " << max_weight_delta << ",\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";

    return passed ? 0 : 2;
}

int print_transformer_block_step_smoke_json(const Config& cfg, const char* program) {
    constexpr std::int64_t kBatch = 1;
    constexpr std::int64_t kHeads = 12;
    constexpr std::int64_t kSeq = 2;
    constexpr std::int64_t kDim = 768;
    constexpr std::int64_t kHeadDim = kDim / kHeads;
    constexpr std::int64_t kRows = kBatch * kSeq;
    constexpr std::int64_t kHidden = kDim * 4;
    constexpr std::int64_t kQkvDim = kDim * 3;
    constexpr std::int64_t kActivationElements = kRows * kDim;
    constexpr std::int64_t kHiddenElements = kRows * kHidden;
    constexpr std::int64_t kQkvActivationElements = kRows * kQkvDim;
    constexpr std::int64_t kQkvWeightElements = kQkvDim * kDim;
    constexpr std::int64_t kAttnProjWeightElements = kDim * kDim;
    constexpr std::int64_t kFcWeightElements = kHidden * kDim;
    constexpr std::int64_t kMlpProjWeightElements = kDim * kHidden;
    constexpr float kLnWeight = 1.0f;
    constexpr float kLnBias = 0.0f;
    constexpr float kQWeight = 0.02f;
    constexpr float kKWeight = 0.01f;
    constexpr float kVWeight = -0.015f;
    constexpr float kAttnProjWeight = 0.005f;
    constexpr float kFcWeight = 0.002f;
    constexpr float kMlpProjWeight = 0.001f;
    constexpr float kGradFinal = 0.25f;
    constexpr float kResidualScale = 1.0f;
    constexpr float kLearningRate = 0.1f;
    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.95f;
    constexpr float kEps = 1e-8f;
    constexpr float kNormEps = 1e-5f;
    constexpr float kWeightDecay = 0.1f;
    constexpr float kBiasCorrection1 = 1.0f - kBeta1;
    constexpr float kSqrtBiasCorrection2 = 0.22360679775f;
    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;

    std::vector<float> host_x(static_cast<std::size_t>(kActivationElements), 0.0f);
    for (std::int64_t i = 0; i < kActivationElements; ++i) {
        host_x[static_cast<std::size_t>(i)] = static_cast<float>((i % 29) - 14) * 0.005f;
    }
    std::vector<float> host_qkv_weight(static_cast<std::size_t>(kQkvWeightElements), 0.0f);
    for (std::int64_t row = 0; row < kQkvDim; ++row) {
        const float value = row < kDim ? kQWeight : (row < 2 * kDim ? kKWeight : kVWeight);
        const std::int64_t offset = row * kDim;
        for (std::int64_t col = 0; col < kDim; ++col) {
            host_qkv_weight[static_cast<std::size_t>(offset + col)] = value;
        }
    }

    const std::string tile_lib_path = cfg.tile_ops_lib.empty() ? default_tile_ops_lib(program) : cfg.tile_ops_lib;
    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(cfg);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool passed = false;
    double max_abs_sample = 0.0;
    double max_weight_delta = 0.0;
    std::string error;

    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using GradientAccumulateFn = int (*)(float*, const float*, std::int64_t, float, void*);
    using ResidualAddFn = int (*)(const float*, const float*, const float*, float*, std::int64_t, void*);
    using SplitQkvFn = int (*)(const float*, float*, float*, float*, std::int64_t, std::int64_t, void*);
    using MergeQkvFn = int (*)(const float*, const float*, const float*, float*, std::int64_t, std::int64_t, void*);
    using ReshapeHeadsFn = int (*)(
        const float*, float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using MergeHeadsFn = int (*)(
        const float*, float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using LayerNormFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, float, void*);
    using LayerNormBackwardInputFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, float, void*);
    using LayerNormBackwardAffineFn = int (*)(
        const float*, const float*, float*, float*, std::int64_t, std::int64_t, float, void*);
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBackwardInputFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardBiasFn = int (*)(const float*, float*, std::int64_t, std::int64_t, void*);
    using GeluFn = int (*)(const float*, float*, std::int64_t, void*);
    using GeluBackwardFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
    using AttentionFn = int (*)(
        const float*,
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
    using AttentionBackwardFn = int (*)(
        const float*,
        const float*,
        const float*,
        const float*,
        float*,
        float*,
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
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);

    void* tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    void* cuda_handle = nullptr;
    FillFn fill = nullptr;
    GradientAccumulateFn gradient_accumulate = nullptr;
    ResidualAddFn residual_add = nullptr;
    SplitQkvFn split_qkv = nullptr;
    MergeQkvFn merge_qkv = nullptr;
    ReshapeHeadsFn reshape_heads = nullptr;
    MergeHeadsFn merge_heads = nullptr;
    LayerNormFn layer_norm = nullptr;
    LayerNormBackwardInputFn layer_norm_backward_input = nullptr;
    LayerNormBackwardAffineFn layer_norm_backward_affine = nullptr;
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightFn linear_backward_weight = nullptr;
    LinearBackwardBiasFn linear_backward_bias = nullptr;
    GeluFn gelu = nullptr;
    GeluBackwardFn gelu_backward = nullptr;
    AttentionFn attention = nullptr;
    AttentionBackwardFn attention_backward = nullptr;
    AdamWFn adamw = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed with CUDA error " << code;
        if (cuda_get_error_string != nullptr) {
            const char* message = cuda_get_error_string(code);
            if (message != nullptr) {
                out << ": " << message;
            }
        }
        return out.str();
    };

    if (tile_handle == nullptr) {
        error = dl_last_error("dlopen tile ops failed");
    } else {
        tile_loaded = true;
        fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
        gradient_accumulate = load_symbol<GradientAccumulateFn>(
            tile_handle, "nfn_native_tile_gradient_accumulate_float32");
        residual_add = load_symbol<ResidualAddFn>(tile_handle, "nfn_native_tile_scaled_residual_add_float32");
        split_qkv = load_symbol<SplitQkvFn>(tile_handle, "nfn_native_tile_split_qkv_float32");
        merge_qkv = load_symbol<MergeQkvFn>(tile_handle, "nfn_native_tile_merge_qkv_float32");
        reshape_heads = load_symbol<ReshapeHeadsFn>(tile_handle, "nfn_native_tile_reshape_heads_float32");
        merge_heads = load_symbol<MergeHeadsFn>(tile_handle, "nfn_native_tile_merge_heads_float32");
        layer_norm = load_symbol<LayerNormFn>(tile_handle, "nfn_native_tile_layer_norm_float32");
        layer_norm_backward_input = load_symbol<LayerNormBackwardInputFn>(
            tile_handle, "nfn_native_tile_layer_norm_backward_input_float32");
        layer_norm_backward_affine = load_symbol<LayerNormBackwardAffineFn>(
            tile_handle, "nfn_native_tile_layer_norm_backward_affine_float32");
        linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
        linear_backward_input = load_symbol<LinearBackwardInputFn>(
            tile_handle, "nfn_native_tile_linear_backward_input_float32");
        linear_backward_weight = load_symbol<LinearBackwardWeightFn>(
            tile_handle, "nfn_native_tile_linear_backward_weight_float32");
        linear_backward_bias = load_symbol<LinearBackwardBiasFn>(
            tile_handle, "nfn_native_tile_linear_backward_bias_float32");
        gelu = load_symbol<GeluFn>(tile_handle, "nfn_native_tile_gelu_float32");
        gelu_backward = load_symbol<GeluBackwardFn>(tile_handle, "nfn_native_tile_gelu_backward_float32");
        attention = load_symbol<AttentionFn>(tile_handle, "nfn_native_tile_scaled_dot_product_attention_float32");
        attention_backward = load_symbol<AttentionBackwardFn>(
            tile_handle, "nfn_native_tile_scaled_dot_product_attention_backward_float32");
        adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
        if (fill == nullptr || gradient_accumulate == nullptr || residual_add == nullptr ||
            split_qkv == nullptr || merge_qkv == nullptr || layer_norm == nullptr ||
            reshape_heads == nullptr || merge_heads == nullptr || layer_norm_backward_input == nullptr ||
            layer_norm_backward_affine == nullptr ||
            linear == nullptr || linear_backward_input == nullptr || linear_backward_weight == nullptr ||
            linear_backward_bias == nullptr ||
            gelu == nullptr || gelu_backward == nullptr || attention == nullptr ||
            attention_backward == nullptr || adamw == nullptr) {
            error = dl_last_error("dlsym GPT-2 transformer-block kernels failed");
        }
    }

    if (error.empty()) {
        std::string runtime_error;
        for (const std::string& candidate : runtime_candidates) {
            cuda_lib_path = candidate;
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_runtime_loaded = true;
                break;
            }
            runtime_error = dl_last_error("dlopen CUDA runtime failed");
        }
        if (cuda_handle == nullptr) {
            error = runtime_error.empty() ? "could not load CUDA runtime" : runtime_error;
        }
    }

    if (error.empty()) {
        cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
        cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
        cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
        cuda_device_synchronize = load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
        cuda_get_error_string = load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
        if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr || cuda_device_synchronize == nullptr) {
            error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaDeviceSynchronize";
        }
    }

    float* x = nullptr;
    float* residual_scale = nullptr;
    float* ln1_weight = nullptr;
    float* ln1_bias = nullptr;
    float* ln2_weight = nullptr;
    float* ln2_bias = nullptr;
    float* qkv_weight = nullptr;
    float* qkv_bias = nullptr;
    float* attn_proj_weight = nullptr;
    float* attn_proj_bias = nullptr;
    float* fc_weight = nullptr;
    float* fc_bias = nullptr;
    float* mlp_proj_weight = nullptr;
    float* mlp_proj_bias = nullptr;
    float* ln1_out = nullptr;
    float* qkv = nullptr;
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* q_heads = nullptr;
    float* k_heads = nullptr;
    float* v_heads = nullptr;
    float* attn_heads = nullptr;
    float* attn_out = nullptr;
    float* attn_proj = nullptr;
    float* residual1 = nullptr;
    float* ln2_out = nullptr;
    float* fc_out = nullptr;
    float* act = nullptr;
    float* mlp_out = nullptr;
    float* residual2 = nullptr;
    float* grad_residual2 = nullptr;
    float* grad_act = nullptr;
    float* grad_fc_out = nullptr;
    float* grad_ln2 = nullptr;
    float* grad_residual1_from_mlp = nullptr;
    float* grad_residual1 = nullptr;
    float* grad_attn_out = nullptr;
    float* grad_attn_heads = nullptr;
    float* grad_q_heads = nullptr;
    float* grad_k_heads = nullptr;
    float* grad_v_heads = nullptr;
    float* grad_q = nullptr;
    float* grad_k = nullptr;
    float* grad_v = nullptr;
    float* grad_qkv = nullptr;
    float* grad_ln1 = nullptr;
    float* grad_x_from_attn = nullptr;
    float* grad_x = nullptr;
    float* grad_ln1_weight = nullptr;
    float* grad_ln1_bias = nullptr;
    float* grad_ln2_weight = nullptr;
    float* grad_ln2_bias = nullptr;
    float* grad_qkv_weight = nullptr;
    float* grad_qkv_bias = nullptr;
    float* grad_attn_proj_weight = nullptr;
    float* grad_attn_proj_bias = nullptr;
    float* grad_fc_weight = nullptr;
    float* grad_fc_bias = nullptr;
    float* grad_mlp_proj_weight = nullptr;
    float* grad_mlp_proj_bias = nullptr;
    float* ln1_weight_exp_avg = nullptr;
    float* ln1_weight_exp_avg_sq = nullptr;
    float* ln1_bias_exp_avg = nullptr;
    float* ln1_bias_exp_avg_sq = nullptr;
    float* ln2_weight_exp_avg = nullptr;
    float* ln2_weight_exp_avg_sq = nullptr;
    float* ln2_bias_exp_avg = nullptr;
    float* ln2_bias_exp_avg_sq = nullptr;
    float* qkv_exp_avg = nullptr;
    float* qkv_exp_avg_sq = nullptr;
    float* qkv_bias_exp_avg = nullptr;
    float* qkv_bias_exp_avg_sq = nullptr;
    float* attn_proj_exp_avg = nullptr;
    float* attn_proj_exp_avg_sq = nullptr;
    float* attn_proj_bias_exp_avg = nullptr;
    float* attn_proj_bias_exp_avg_sq = nullptr;
    float* fc_exp_avg = nullptr;
    float* fc_exp_avg_sq = nullptr;
    float* fc_bias_exp_avg = nullptr;
    float* fc_bias_exp_avg_sq = nullptr;
    float* mlp_proj_exp_avg = nullptr;
    float* mlp_proj_exp_avg_sq = nullptr;
    float* mlp_proj_bias_exp_avg = nullptr;
    float* mlp_proj_bias_exp_avg_sq = nullptr;

    auto allocate = [&](float** ptr, std::int64_t elements, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        const int status = cuda_malloc(
            reinterpret_cast<void**>(ptr),
            sizeof(float) * static_cast<std::size_t>(elements));
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc " + name);
        }
    };
    allocate(&x, kActivationElements, "x");
    allocate(&residual_scale, 1, "residual_scale");
    allocate(&ln1_weight, kDim, "ln1_weight");
    allocate(&ln1_bias, kDim, "ln1_bias");
    allocate(&ln2_weight, kDim, "ln2_weight");
    allocate(&ln2_bias, kDim, "ln2_bias");
    allocate(&qkv_weight, kQkvWeightElements, "qkv_weight");
    allocate(&qkv_bias, kQkvDim, "qkv_bias");
    allocate(&attn_proj_weight, kAttnProjWeightElements, "attn_proj_weight");
    allocate(&attn_proj_bias, kDim, "attn_proj_bias");
    allocate(&fc_weight, kFcWeightElements, "fc_weight");
    allocate(&fc_bias, kHidden, "fc_bias");
    allocate(&mlp_proj_weight, kMlpProjWeightElements, "mlp_proj_weight");
    allocate(&mlp_proj_bias, kDim, "mlp_proj_bias");
    allocate(&ln1_out, kActivationElements, "ln1_out");
    allocate(&qkv, kQkvActivationElements, "qkv");
    allocate(&q, kActivationElements, "q");
    allocate(&k, kActivationElements, "k");
    allocate(&v, kActivationElements, "v");
    allocate(&q_heads, kActivationElements, "q_heads");
    allocate(&k_heads, kActivationElements, "k_heads");
    allocate(&v_heads, kActivationElements, "v_heads");
    allocate(&attn_heads, kActivationElements, "attn_heads");
    allocate(&attn_out, kActivationElements, "attn_out");
    allocate(&attn_proj, kActivationElements, "attn_proj");
    allocate(&residual1, kActivationElements, "residual1");
    allocate(&ln2_out, kActivationElements, "ln2_out");
    allocate(&fc_out, kHiddenElements, "fc_out");
    allocate(&act, kHiddenElements, "act");
    allocate(&mlp_out, kActivationElements, "mlp_out");
    allocate(&residual2, kActivationElements, "residual2");
    allocate(&grad_residual2, kActivationElements, "grad_residual2");
    allocate(&grad_act, kHiddenElements, "grad_act");
    allocate(&grad_fc_out, kHiddenElements, "grad_fc_out");
    allocate(&grad_ln2, kActivationElements, "grad_ln2");
    allocate(&grad_residual1_from_mlp, kActivationElements, "grad_residual1_from_mlp");
    allocate(&grad_residual1, kActivationElements, "grad_residual1");
    allocate(&grad_attn_out, kActivationElements, "grad_attn_out");
    allocate(&grad_attn_heads, kActivationElements, "grad_attn_heads");
    allocate(&grad_q_heads, kActivationElements, "grad_q_heads");
    allocate(&grad_k_heads, kActivationElements, "grad_k_heads");
    allocate(&grad_v_heads, kActivationElements, "grad_v_heads");
    allocate(&grad_q, kActivationElements, "grad_q");
    allocate(&grad_k, kActivationElements, "grad_k");
    allocate(&grad_v, kActivationElements, "grad_v");
    allocate(&grad_qkv, kQkvActivationElements, "grad_qkv");
    allocate(&grad_ln1, kActivationElements, "grad_ln1");
    allocate(&grad_x_from_attn, kActivationElements, "grad_x_from_attn");
    allocate(&grad_x, kActivationElements, "grad_x");
    allocate(&grad_ln1_weight, kDim, "grad_ln1_weight");
    allocate(&grad_ln1_bias, kDim, "grad_ln1_bias");
    allocate(&grad_ln2_weight, kDim, "grad_ln2_weight");
    allocate(&grad_ln2_bias, kDim, "grad_ln2_bias");
    allocate(&grad_qkv_weight, kQkvWeightElements, "grad_qkv_weight");
    allocate(&grad_qkv_bias, kQkvDim, "grad_qkv_bias");
    allocate(&grad_attn_proj_weight, kAttnProjWeightElements, "grad_attn_proj_weight");
    allocate(&grad_attn_proj_bias, kDim, "grad_attn_proj_bias");
    allocate(&grad_fc_weight, kFcWeightElements, "grad_fc_weight");
    allocate(&grad_fc_bias, kHidden, "grad_fc_bias");
    allocate(&grad_mlp_proj_weight, kMlpProjWeightElements, "grad_mlp_proj_weight");
    allocate(&grad_mlp_proj_bias, kDim, "grad_mlp_proj_bias");
    allocate(&ln1_weight_exp_avg, kDim, "ln1_weight_exp_avg");
    allocate(&ln1_weight_exp_avg_sq, kDim, "ln1_weight_exp_avg_sq");
    allocate(&ln1_bias_exp_avg, kDim, "ln1_bias_exp_avg");
    allocate(&ln1_bias_exp_avg_sq, kDim, "ln1_bias_exp_avg_sq");
    allocate(&ln2_weight_exp_avg, kDim, "ln2_weight_exp_avg");
    allocate(&ln2_weight_exp_avg_sq, kDim, "ln2_weight_exp_avg_sq");
    allocate(&ln2_bias_exp_avg, kDim, "ln2_bias_exp_avg");
    allocate(&ln2_bias_exp_avg_sq, kDim, "ln2_bias_exp_avg_sq");
    allocate(&qkv_exp_avg, kQkvWeightElements, "qkv_exp_avg");
    allocate(&qkv_exp_avg_sq, kQkvWeightElements, "qkv_exp_avg_sq");
    allocate(&qkv_bias_exp_avg, kQkvDim, "qkv_bias_exp_avg");
    allocate(&qkv_bias_exp_avg_sq, kQkvDim, "qkv_bias_exp_avg_sq");
    allocate(&attn_proj_exp_avg, kAttnProjWeightElements, "attn_proj_exp_avg");
    allocate(&attn_proj_exp_avg_sq, kAttnProjWeightElements, "attn_proj_exp_avg_sq");
    allocate(&attn_proj_bias_exp_avg, kDim, "attn_proj_bias_exp_avg");
    allocate(&attn_proj_bias_exp_avg_sq, kDim, "attn_proj_bias_exp_avg_sq");
    allocate(&fc_exp_avg, kFcWeightElements, "fc_exp_avg");
    allocate(&fc_exp_avg_sq, kFcWeightElements, "fc_exp_avg_sq");
    allocate(&fc_bias_exp_avg, kHidden, "fc_bias_exp_avg");
    allocate(&fc_bias_exp_avg_sq, kHidden, "fc_bias_exp_avg_sq");
    allocate(&mlp_proj_exp_avg, kMlpProjWeightElements, "mlp_proj_exp_avg");
    allocate(&mlp_proj_exp_avg_sq, kMlpProjWeightElements, "mlp_proj_exp_avg_sq");
    allocate(&mlp_proj_bias_exp_avg, kDim, "mlp_proj_bias_exp_avg");
    allocate(&mlp_proj_bias_exp_avg_sq, kDim, "mlp_proj_bias_exp_avg_sq");

    auto fill_buffer = [&](float* ptr, std::int64_t elements, float value, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        const int status = fill(ptr, elements, value, nullptr);
        if (status != 0) {
            error = cuda_error(status, "fill " + name);
        }
    };
    auto copy_to_device = [&](float* dst, const std::vector<float>& host, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        const int status = cuda_memcpy(dst, host.data(), sizeof(float) * host.size(), kCudaMemcpyHostToDevice);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy host-to-device " + name);
        }
    };
    copy_to_device(x, host_x, "x");
    copy_to_device(qkv_weight, host_qkv_weight, "qkv_weight");
    fill_buffer(residual_scale, 1, kResidualScale, "residual_scale");
    fill_buffer(ln1_weight, kDim, kLnWeight, "ln1_weight");
    fill_buffer(ln1_bias, kDim, kLnBias, "ln1_bias");
    fill_buffer(ln2_weight, kDim, kLnWeight, "ln2_weight");
    fill_buffer(ln2_bias, kDim, kLnBias, "ln2_bias");
    fill_buffer(qkv_bias, kQkvDim, 0.0f, "qkv_bias");
    fill_buffer(attn_proj_weight, kAttnProjWeightElements, kAttnProjWeight, "attn_proj_weight");
    fill_buffer(attn_proj_bias, kDim, 0.0f, "attn_proj_bias");
    fill_buffer(fc_weight, kFcWeightElements, kFcWeight, "fc_weight");
    fill_buffer(fc_bias, kHidden, 0.0f, "fc_bias");
    fill_buffer(mlp_proj_weight, kMlpProjWeightElements, kMlpProjWeight, "mlp_proj_weight");
    fill_buffer(mlp_proj_bias, kDim, 0.0f, "mlp_proj_bias");
    fill_buffer(grad_residual2, kActivationElements, kGradFinal, "grad_residual2");
    fill_buffer(grad_ln1_weight, kDim, 0.0f, "grad_ln1_weight");
    fill_buffer(grad_ln1_bias, kDim, 0.0f, "grad_ln1_bias");
    fill_buffer(grad_ln2_weight, kDim, 0.0f, "grad_ln2_weight");
    fill_buffer(grad_ln2_bias, kDim, 0.0f, "grad_ln2_bias");
    fill_buffer(grad_qkv_weight, kQkvWeightElements, 0.0f, "grad_qkv_weight");
    fill_buffer(grad_qkv_bias, kQkvDim, 0.0f, "grad_qkv_bias");
    fill_buffer(grad_attn_proj_weight, kAttnProjWeightElements, 0.0f, "grad_attn_proj_weight");
    fill_buffer(grad_attn_proj_bias, kDim, 0.0f, "grad_attn_proj_bias");
    fill_buffer(grad_fc_weight, kFcWeightElements, 0.0f, "grad_fc_weight");
    fill_buffer(grad_fc_bias, kHidden, 0.0f, "grad_fc_bias");
    fill_buffer(grad_mlp_proj_weight, kMlpProjWeightElements, 0.0f, "grad_mlp_proj_weight");
    fill_buffer(grad_mlp_proj_bias, kDim, 0.0f, "grad_mlp_proj_bias");
    fill_buffer(ln1_weight_exp_avg, kDim, 0.0f, "ln1_weight_exp_avg");
    fill_buffer(ln1_weight_exp_avg_sq, kDim, 0.0f, "ln1_weight_exp_avg_sq");
    fill_buffer(ln1_bias_exp_avg, kDim, 0.0f, "ln1_bias_exp_avg");
    fill_buffer(ln1_bias_exp_avg_sq, kDim, 0.0f, "ln1_bias_exp_avg_sq");
    fill_buffer(ln2_weight_exp_avg, kDim, 0.0f, "ln2_weight_exp_avg");
    fill_buffer(ln2_weight_exp_avg_sq, kDim, 0.0f, "ln2_weight_exp_avg_sq");
    fill_buffer(ln2_bias_exp_avg, kDim, 0.0f, "ln2_bias_exp_avg");
    fill_buffer(ln2_bias_exp_avg_sq, kDim, 0.0f, "ln2_bias_exp_avg_sq");
    fill_buffer(qkv_exp_avg, kQkvWeightElements, 0.0f, "qkv_exp_avg");
    fill_buffer(qkv_exp_avg_sq, kQkvWeightElements, 0.0f, "qkv_exp_avg_sq");
    fill_buffer(qkv_bias_exp_avg, kQkvDim, 0.0f, "qkv_bias_exp_avg");
    fill_buffer(qkv_bias_exp_avg_sq, kQkvDim, 0.0f, "qkv_bias_exp_avg_sq");
    fill_buffer(attn_proj_exp_avg, kAttnProjWeightElements, 0.0f, "attn_proj_exp_avg");
    fill_buffer(attn_proj_exp_avg_sq, kAttnProjWeightElements, 0.0f, "attn_proj_exp_avg_sq");
    fill_buffer(attn_proj_bias_exp_avg, kDim, 0.0f, "attn_proj_bias_exp_avg");
    fill_buffer(attn_proj_bias_exp_avg_sq, kDim, 0.0f, "attn_proj_bias_exp_avg_sq");
    fill_buffer(fc_exp_avg, kFcWeightElements, 0.0f, "fc_exp_avg");
    fill_buffer(fc_exp_avg_sq, kFcWeightElements, 0.0f, "fc_exp_avg_sq");
    fill_buffer(fc_bias_exp_avg, kHidden, 0.0f, "fc_bias_exp_avg");
    fill_buffer(fc_bias_exp_avg_sq, kHidden, 0.0f, "fc_bias_exp_avg_sq");
    fill_buffer(mlp_proj_exp_avg, kMlpProjWeightElements, 0.0f, "mlp_proj_exp_avg");
    fill_buffer(mlp_proj_exp_avg_sq, kMlpProjWeightElements, 0.0f, "mlp_proj_exp_avg_sq");
    fill_buffer(mlp_proj_bias_exp_avg, kDim, 0.0f, "mlp_proj_bias_exp_avg");
    fill_buffer(mlp_proj_bias_exp_avg_sq, kDim, 0.0f, "mlp_proj_bias_exp_avg_sq");

    auto run = [&](int status, const std::string& name) {
        if (status != 0 && error.empty()) {
            error = cuda_error(status, name);
        }
    };
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    if (error.empty()) {
        run(layer_norm(x, ln1_weight, ln1_bias, ln1_out, kRows, kDim, kNormEps, nullptr), "ln1.forward");
    }
    if (error.empty()) {
        run(linear(ln1_out, qkv_weight, qkv_bias, qkv, kRows, kDim, kQkvDim, true, nullptr), "attn.qkv.forward");
    }
    if (error.empty()) {
        run(split_qkv(qkv, q, k, v, kRows, kDim, nullptr), "attn.qkv.split");
    }
    if (error.empty()) {
        run(reshape_heads(q, q_heads, kBatch, kSeq, kHeads, kHeadDim, nullptr), "attn.q.reshape_heads");
        run(reshape_heads(k, k_heads, kBatch, kSeq, kHeads, kHeadDim, nullptr), "attn.k.reshape_heads");
        run(reshape_heads(v, v_heads, kBatch, kSeq, kHeads, kHeadDim, nullptr), "attn.v.reshape_heads");
    }
    if (error.empty()) {
        run(attention(
                q_heads,
                k_heads,
                v_heads,
                attn_heads,
                kActivationElements,
                kHeads,
                kHeads,
                kSeq,
                kSeq,
                kHeadDim,
                kHeadDim,
                attention_scale,
                true,
                false,
                false,
                0,
                0,
                0,
                0,
                nullptr),
            "attn.sdpa.forward");
    }
    if (error.empty()) {
        run(merge_heads(attn_heads, attn_out, kBatch, kHeads, kSeq, kHeadDim, nullptr), "attn.merge_heads");
    }
    if (error.empty()) {
        run(linear(attn_out, attn_proj_weight, attn_proj_bias, attn_proj, kRows, kDim, kDim, true, nullptr),
            "attn.out.forward");
    }
    if (error.empty()) {
        run(residual_add(x, attn_proj, residual_scale, residual1, kActivationElements, nullptr),
            "attn.residual.forward");
    }
    if (error.empty()) {
        run(layer_norm(residual1, ln2_weight, ln2_bias, ln2_out, kRows, kDim, kNormEps, nullptr),
            "ln2.forward");
    }
    if (error.empty()) {
        run(linear(ln2_out, fc_weight, fc_bias, fc_out, kRows, kDim, kHidden, true, nullptr),
            "mlp.c_fc.forward");
    }
    if (error.empty()) {
        run(gelu(fc_out, act, kHiddenElements, nullptr), "mlp.gelu.forward");
    }
    if (error.empty()) {
        run(linear(act, mlp_proj_weight, mlp_proj_bias, mlp_out, kRows, kHidden, kDim, true, nullptr),
            "mlp.c_proj.forward");
    }
    if (error.empty()) {
        run(residual_add(residual1, mlp_out, residual_scale, residual2, kActivationElements, nullptr),
            "mlp.residual.forward");
    }
    if (error.empty()) {
        run(linear_backward_weight(act, grad_residual2, grad_mlp_proj_weight, kRows, kHidden, kDim, nullptr),
            "mlp.c_proj.backward_weight");
    }
    if (error.empty()) {
        run(linear_backward_bias(grad_residual2, grad_mlp_proj_bias, kRows, kDim, nullptr),
            "mlp.c_proj.backward_bias");
    }
    if (error.empty()) {
        run(linear_backward_input(grad_residual2, mlp_proj_weight, grad_act, kRows, kHidden, kDim, nullptr),
            "mlp.c_proj.backward_input");
    }
    if (error.empty()) {
        run(gelu_backward(fc_out, grad_act, grad_fc_out, kHiddenElements, nullptr), "mlp.gelu.backward");
    }
    if (error.empty()) {
        run(linear_backward_weight(ln2_out, grad_fc_out, grad_fc_weight, kRows, kDim, kHidden, nullptr),
            "mlp.c_fc.backward_weight");
    }
    if (error.empty()) {
        run(linear_backward_bias(grad_fc_out, grad_fc_bias, kRows, kHidden, nullptr),
            "mlp.c_fc.backward_bias");
    }
    if (error.empty()) {
        run(linear_backward_input(grad_fc_out, fc_weight, grad_ln2, kRows, kDim, kHidden, nullptr),
            "mlp.c_fc.backward_input");
    }
    if (error.empty()) {
        run(layer_norm_backward_affine(residual1, grad_ln2, grad_ln2_weight, grad_ln2_bias, kRows, kDim, kNormEps, nullptr),
            "ln2.backward_affine");
    }
    if (error.empty()) {
        run(layer_norm_backward_input(residual1, grad_ln2, ln2_weight, grad_residual1_from_mlp, kRows, kDim, kNormEps, nullptr),
            "ln2.backward_input");
    }
    fill_buffer(grad_residual1, kActivationElements, kGradFinal, "grad_residual1");
    if (error.empty()) {
        run(gradient_accumulate(grad_residual1, grad_residual1_from_mlp, kActivationElements, 1.0f, nullptr),
            "mlp.residual.backward_accumulate");
    }
    if (error.empty()) {
        run(linear_backward_weight(attn_out, grad_residual1, grad_attn_proj_weight, kRows, kDim, kDim, nullptr),
            "attn.out.backward_weight");
    }
    if (error.empty()) {
        run(linear_backward_bias(grad_residual1, grad_attn_proj_bias, kRows, kDim, nullptr),
            "attn.out.backward_bias");
    }
    if (error.empty()) {
        run(linear_backward_input(grad_residual1, attn_proj_weight, grad_attn_out, kRows, kDim, kDim, nullptr),
            "attn.out.backward_input");
    }
    if (error.empty()) {
        run(reshape_heads(grad_attn_out, grad_attn_heads, kBatch, kSeq, kHeads, kHeadDim, nullptr),
            "attn.grad_out.reshape_heads");
    }
    if (error.empty()) {
        run(attention_backward(
                q_heads,
                k_heads,
                v_heads,
                grad_attn_heads,
                grad_q_heads,
                grad_k_heads,
                grad_v_heads,
                kBatch,
                kHeads,
                kHeads,
                kSeq,
                kSeq,
                kHeadDim,
                kHeadDim,
                attention_scale,
                true,
                false,
                false,
                0,
                0,
                0,
                0,
                nullptr),
            "attn.sdpa.backward");
    }
    if (error.empty()) {
        run(merge_heads(grad_q_heads, grad_q, kBatch, kHeads, kSeq, kHeadDim, nullptr), "attn.grad_q.merge_heads");
        run(merge_heads(grad_k_heads, grad_k, kBatch, kHeads, kSeq, kHeadDim, nullptr), "attn.grad_k.merge_heads");
        run(merge_heads(grad_v_heads, grad_v, kBatch, kHeads, kSeq, kHeadDim, nullptr), "attn.grad_v.merge_heads");
    }
    if (error.empty()) {
        run(merge_qkv(grad_q, grad_k, grad_v, grad_qkv, kRows, kDim, nullptr), "attn.qkv.merge_grad");
    }
    if (error.empty()) {
        run(linear_backward_weight(ln1_out, grad_qkv, grad_qkv_weight, kRows, kDim, kQkvDim, nullptr),
            "attn.qkv.backward_weight");
    }
    if (error.empty()) {
        run(linear_backward_bias(grad_qkv, grad_qkv_bias, kRows, kQkvDim, nullptr),
            "attn.qkv.backward_bias");
    }
    if (error.empty()) {
        run(linear_backward_input(grad_qkv, qkv_weight, grad_ln1, kRows, kDim, kQkvDim, nullptr),
            "attn.qkv.backward_input");
    }
    if (error.empty()) {
        run(layer_norm_backward_affine(x, grad_ln1, grad_ln1_weight, grad_ln1_bias, kRows, kDim, kNormEps, nullptr),
            "ln1.backward_affine");
    }
    if (error.empty()) {
        run(layer_norm_backward_input(x, grad_ln1, ln1_weight, grad_x_from_attn, kRows, kDim, kNormEps, nullptr),
            "ln1.backward_input");
    }
    fill_buffer(grad_x, kActivationElements, kGradFinal, "grad_x");
    if (error.empty()) {
        run(gradient_accumulate(grad_x, grad_x_from_attn, kActivationElements, 1.0f, nullptr),
            "attn.residual.backward_accumulate");
    }
    if (error.empty()) {
        run(gradient_accumulate(grad_x, grad_residual1_from_mlp, kActivationElements, 1.0f, nullptr),
            "mlp.direct_residual.backward_accumulate");
    }
    if (error.empty()) {
        run(adamw(
                ln1_weight,
                grad_ln1_weight,
                ln1_weight_exp_avg,
                ln1_weight_exp_avg_sq,
                kDim,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                0.0f,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr),
            "ln1.weight.adamw");
    }
    if (error.empty()) {
        run(adamw(
                ln1_bias,
                grad_ln1_bias,
                ln1_bias_exp_avg,
                ln1_bias_exp_avg_sq,
                kDim,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                0.0f,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr),
            "ln1.bias.adamw");
    }
    if (error.empty()) {
        run(adamw(
                qkv_weight,
                grad_qkv_weight,
                qkv_exp_avg,
                qkv_exp_avg_sq,
                kQkvWeightElements,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                kWeightDecay,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr),
            "attn.qkv.adamw");
    }
    if (error.empty()) {
        run(adamw(
                qkv_bias,
                grad_qkv_bias,
                qkv_bias_exp_avg,
                qkv_bias_exp_avg_sq,
                kQkvDim,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                0.0f,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr),
            "attn.qkv.bias.adamw");
    }
    if (error.empty()) {
        run(adamw(
                attn_proj_weight,
                grad_attn_proj_weight,
                attn_proj_exp_avg,
                attn_proj_exp_avg_sq,
                kAttnProjWeightElements,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                kWeightDecay,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr),
            "attn.out.adamw");
    }
    if (error.empty()) {
        run(adamw(
                attn_proj_bias,
                grad_attn_proj_bias,
                attn_proj_bias_exp_avg,
                attn_proj_bias_exp_avg_sq,
                kDim,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                0.0f,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr),
            "attn.out.bias.adamw");
    }
    if (error.empty()) {
        run(adamw(
                ln2_weight,
                grad_ln2_weight,
                ln2_weight_exp_avg,
                ln2_weight_exp_avg_sq,
                kDim,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                0.0f,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr),
            "ln2.weight.adamw");
    }
    if (error.empty()) {
        run(adamw(
                ln2_bias,
                grad_ln2_bias,
                ln2_bias_exp_avg,
                ln2_bias_exp_avg_sq,
                kDim,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                0.0f,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr),
            "ln2.bias.adamw");
    }
    if (error.empty()) {
        run(adamw(
                fc_weight,
                grad_fc_weight,
                fc_exp_avg,
                fc_exp_avg_sq,
                kFcWeightElements,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                kWeightDecay,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr),
            "mlp.c_fc.adamw");
    }
    if (error.empty()) {
        run(adamw(
                fc_bias,
                grad_fc_bias,
                fc_bias_exp_avg,
                fc_bias_exp_avg_sq,
                kHidden,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                0.0f,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr),
            "mlp.c_fc.bias.adamw");
    }
    if (error.empty()) {
        run(adamw(
                mlp_proj_weight,
                grad_mlp_proj_weight,
                mlp_proj_exp_avg,
                mlp_proj_exp_avg_sq,
                kMlpProjWeightElements,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                kWeightDecay,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr),
            "mlp.c_proj.adamw");
    }
    if (error.empty()) {
        run(adamw(
                mlp_proj_bias,
                grad_mlp_proj_bias,
                mlp_proj_bias_exp_avg,
                mlp_proj_bias_exp_avg_sq,
                kDim,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                0.0f,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr),
            "mlp.c_proj.bias.adamw");
    }
    if (error.empty()) {
        run(cuda_device_synchronize(), "cudaDeviceSynchronize");
    }

    auto copy_float_sample = [&](const float* device_ptr, std::int64_t offset, const std::string& name) -> float {
        float value = 0.0f;
        if (!error.empty()) {
            return value;
        }
        const int status = cuda_memcpy(&value, device_ptr + offset, sizeof(float), kCudaMemcpyDeviceToHost);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy sample " + name);
        }
        return value;
    };
    float sampled_residual2 = 0.0f;
    float sampled_grad_x = 0.0f;
    float sampled_grad_qkv_weight = 0.0f;
    float sampled_grad_qkv_bias = 0.0f;
    float sampled_grad_attn_proj_weight = 0.0f;
    float sampled_grad_attn_proj_bias = 0.0f;
    float sampled_grad_fc_weight = 0.0f;
    float sampled_grad_fc_bias = 0.0f;
    float sampled_grad_mlp_proj_weight = 0.0f;
    float sampled_grad_mlp_proj_bias = 0.0f;
    float sampled_ln1_weight = 0.0f;
    float sampled_ln1_bias = 0.0f;
    float sampled_ln2_weight = 0.0f;
    float sampled_ln2_bias = 0.0f;
    float sampled_qkv_weight = 0.0f;
    float sampled_qkv_bias = 0.0f;
    float sampled_attn_proj_weight = 0.0f;
    float sampled_attn_proj_bias = 0.0f;
    float sampled_fc_weight = 0.0f;
    float sampled_fc_bias = 0.0f;
    float sampled_mlp_proj_weight = 0.0f;
    float sampled_mlp_proj_bias = 0.0f;
    if (error.empty()) {
        sampled_residual2 = copy_float_sample(residual2, 17, "residual2");
        sampled_grad_x = copy_float_sample(grad_x, 19, "grad_x");
        sampled_grad_qkv_weight = copy_float_sample(grad_qkv_weight, 23, "grad_qkv_weight");
        sampled_grad_qkv_bias = copy_float_sample(grad_qkv_bias, 0, "grad_qkv_bias");
        sampled_grad_attn_proj_weight = copy_float_sample(grad_attn_proj_weight, 29, "grad_attn_proj_weight");
        sampled_grad_attn_proj_bias = copy_float_sample(grad_attn_proj_bias, 0, "grad_attn_proj_bias");
        sampled_grad_fc_weight = copy_float_sample(grad_fc_weight, 31, "grad_fc_weight");
        sampled_grad_fc_bias = copy_float_sample(grad_fc_bias, 0, "grad_fc_bias");
        sampled_grad_mlp_proj_weight = copy_float_sample(grad_mlp_proj_weight, 37, "grad_mlp_proj_weight");
        sampled_grad_mlp_proj_bias = copy_float_sample(grad_mlp_proj_bias, 0, "grad_mlp_proj_bias");
        sampled_ln1_weight = copy_float_sample(ln1_weight, 0, "ln1_weight");
        sampled_ln1_bias = copy_float_sample(ln1_bias, 0, "ln1_bias");
        sampled_ln2_weight = copy_float_sample(ln2_weight, 0, "ln2_weight");
        sampled_ln2_bias = copy_float_sample(ln2_bias, 0, "ln2_bias");
        sampled_qkv_weight = copy_float_sample(qkv_weight, 0, "qkv_weight");
        sampled_qkv_bias = copy_float_sample(qkv_bias, 0, "qkv_bias");
        sampled_attn_proj_weight = copy_float_sample(attn_proj_weight, 0, "attn_proj_weight");
        sampled_attn_proj_bias = copy_float_sample(attn_proj_bias, 0, "attn_proj_bias");
        sampled_fc_weight = copy_float_sample(fc_weight, 0, "fc_weight");
        sampled_fc_bias = copy_float_sample(fc_bias, 0, "fc_bias");
        sampled_mlp_proj_weight = copy_float_sample(mlp_proj_weight, 0, "mlp_proj_weight");
        sampled_mlp_proj_bias = copy_float_sample(mlp_proj_bias, 0, "mlp_proj_bias");
    }

    auto record_abs = [&](float value) {
        if (!std::isfinite(value)) {
            return false;
        }
        const double abs_value = std::fabs(static_cast<double>(value));
        if (abs_value > max_abs_sample) {
            max_abs_sample = abs_value;
        }
        return true;
    };
    if (error.empty()) {
        const bool finite_samples =
            record_abs(sampled_residual2) && record_abs(sampled_grad_x) &&
            record_abs(sampled_grad_qkv_weight) && record_abs(sampled_grad_attn_proj_weight) &&
            record_abs(sampled_grad_fc_weight) && record_abs(sampled_grad_mlp_proj_weight) &&
            record_abs(sampled_grad_qkv_bias) && record_abs(sampled_grad_attn_proj_bias) &&
            record_abs(sampled_grad_fc_bias) && record_abs(sampled_grad_mlp_proj_bias) &&
            record_abs(sampled_ln1_weight) && record_abs(sampled_ln1_bias) &&
            record_abs(sampled_ln2_weight) && record_abs(sampled_ln2_bias) &&
            record_abs(sampled_qkv_weight) && record_abs(sampled_attn_proj_weight) &&
            record_abs(sampled_fc_weight) && record_abs(sampled_mlp_proj_weight) &&
            record_abs(sampled_qkv_bias) && record_abs(sampled_attn_proj_bias) &&
            record_abs(sampled_fc_bias) && record_abs(sampled_mlp_proj_bias);
        max_weight_delta = std::fabs(static_cast<double>(sampled_ln1_weight) - kLnWeight);
        const double ln1_bias_delta = std::fabs(static_cast<double>(sampled_ln1_bias) - kLnBias);
        const double ln2_weight_delta = std::fabs(static_cast<double>(sampled_ln2_weight) - kLnWeight);
        const double ln2_bias_delta = std::fabs(static_cast<double>(sampled_ln2_bias) - kLnBias);
        const double qkv_delta = std::fabs(static_cast<double>(sampled_qkv_weight) - kQWeight);
        const double qkv_bias_delta = std::fabs(static_cast<double>(sampled_qkv_bias));
        const double attn_proj_delta = std::fabs(static_cast<double>(sampled_attn_proj_weight) - kAttnProjWeight);
        const double attn_proj_bias_delta = std::fabs(static_cast<double>(sampled_attn_proj_bias));
        const double fc_delta = std::fabs(static_cast<double>(sampled_fc_weight) - kFcWeight);
        const double fc_bias_delta = std::fabs(static_cast<double>(sampled_fc_bias));
        const double mlp_proj_delta = std::fabs(static_cast<double>(sampled_mlp_proj_weight) - kMlpProjWeight);
        const double mlp_proj_bias_delta = std::fabs(static_cast<double>(sampled_mlp_proj_bias));
        for (double candidate : {
                 ln1_bias_delta,
                 ln2_weight_delta,
                 ln2_bias_delta,
                 qkv_delta,
                 qkv_bias_delta,
                 attn_proj_delta,
                 attn_proj_bias_delta,
                 fc_delta,
                 fc_bias_delta,
                 mlp_proj_delta,
                 mlp_proj_bias_delta,
             }) {
            if (candidate > max_weight_delta) {
                max_weight_delta = candidate;
            }
        }
        passed = finite_samples && max_abs_sample > 0.0 && max_weight_delta > 0.0;
        if (!passed) {
            std::ostringstream out_message;
            out_message << "GPT-2 transformer-block smoke did not produce finite nonzero samples and weight updates: sample="
                        << max_abs_sample << " weight_delta=" << max_weight_delta;
            error = out_message.str();
        }
    }

    auto free_device = [&](void* ptr, const std::string& name) {
        if (ptr != nullptr && cuda_free != nullptr) {
            const int status = cuda_free(ptr);
            if (status != 0 && error.empty()) {
                error = cuda_error(status, "cudaFree " + name);
            }
        }
    };
    free_device(x, "x");
    free_device(residual_scale, "residual_scale");
    free_device(ln1_weight, "ln1_weight");
    free_device(ln1_bias, "ln1_bias");
    free_device(ln2_weight, "ln2_weight");
    free_device(ln2_bias, "ln2_bias");
    free_device(qkv_weight, "qkv_weight");
    free_device(qkv_bias, "qkv_bias");
    free_device(attn_proj_weight, "attn_proj_weight");
    free_device(attn_proj_bias, "attn_proj_bias");
    free_device(fc_weight, "fc_weight");
    free_device(fc_bias, "fc_bias");
    free_device(mlp_proj_weight, "mlp_proj_weight");
    free_device(mlp_proj_bias, "mlp_proj_bias");
    free_device(ln1_out, "ln1_out");
    free_device(qkv, "qkv");
    free_device(q, "q");
    free_device(k, "k");
    free_device(v, "v");
    free_device(attn_out, "attn_out");
    free_device(attn_proj, "attn_proj");
    free_device(residual1, "residual1");
    free_device(ln2_out, "ln2_out");
    free_device(fc_out, "fc_out");
    free_device(act, "act");
    free_device(mlp_out, "mlp_out");
    free_device(residual2, "residual2");
    free_device(grad_residual2, "grad_residual2");
    free_device(grad_act, "grad_act");
    free_device(grad_fc_out, "grad_fc_out");
    free_device(grad_ln2, "grad_ln2");
    free_device(grad_residual1_from_mlp, "grad_residual1_from_mlp");
    free_device(grad_residual1, "grad_residual1");
    free_device(grad_attn_out, "grad_attn_out");
    free_device(grad_attn_heads, "grad_attn_heads");
    free_device(grad_q_heads, "grad_q_heads");
    free_device(grad_k_heads, "grad_k_heads");
    free_device(grad_v_heads, "grad_v_heads");
    free_device(grad_q, "grad_q");
    free_device(grad_k, "grad_k");
    free_device(grad_v, "grad_v");
    free_device(grad_qkv, "grad_qkv");
    free_device(grad_ln1, "grad_ln1");
    free_device(grad_x_from_attn, "grad_x_from_attn");
    free_device(grad_x, "grad_x");
    free_device(grad_ln1_weight, "grad_ln1_weight");
    free_device(grad_ln1_bias, "grad_ln1_bias");
    free_device(grad_ln2_weight, "grad_ln2_weight");
    free_device(grad_ln2_bias, "grad_ln2_bias");
    free_device(grad_qkv_weight, "grad_qkv_weight");
    free_device(grad_qkv_bias, "grad_qkv_bias");
    free_device(grad_attn_proj_weight, "grad_attn_proj_weight");
    free_device(grad_attn_proj_bias, "grad_attn_proj_bias");
    free_device(grad_fc_weight, "grad_fc_weight");
    free_device(grad_fc_bias, "grad_fc_bias");
    free_device(grad_mlp_proj_weight, "grad_mlp_proj_weight");
    free_device(grad_mlp_proj_bias, "grad_mlp_proj_bias");
    free_device(ln1_weight_exp_avg, "ln1_weight_exp_avg");
    free_device(ln1_weight_exp_avg_sq, "ln1_weight_exp_avg_sq");
    free_device(ln1_bias_exp_avg, "ln1_bias_exp_avg");
    free_device(ln1_bias_exp_avg_sq, "ln1_bias_exp_avg_sq");
    free_device(ln2_weight_exp_avg, "ln2_weight_exp_avg");
    free_device(ln2_weight_exp_avg_sq, "ln2_weight_exp_avg_sq");
    free_device(ln2_bias_exp_avg, "ln2_bias_exp_avg");
    free_device(ln2_bias_exp_avg_sq, "ln2_bias_exp_avg_sq");
    free_device(qkv_exp_avg, "qkv_exp_avg");
    free_device(qkv_exp_avg_sq, "qkv_exp_avg_sq");
    free_device(qkv_bias_exp_avg, "qkv_bias_exp_avg");
    free_device(qkv_bias_exp_avg_sq, "qkv_bias_exp_avg_sq");
    free_device(attn_proj_exp_avg, "attn_proj_exp_avg");
    free_device(attn_proj_exp_avg_sq, "attn_proj_exp_avg_sq");
    free_device(attn_proj_bias_exp_avg, "attn_proj_bias_exp_avg");
    free_device(attn_proj_bias_exp_avg_sq, "attn_proj_bias_exp_avg_sq");
    free_device(fc_exp_avg, "fc_exp_avg");
    free_device(fc_exp_avg_sq, "fc_exp_avg_sq");
    free_device(fc_bias_exp_avg, "fc_bias_exp_avg");
    free_device(fc_bias_exp_avg_sq, "fc_bias_exp_avg_sq");
    free_device(mlp_proj_exp_avg, "mlp_proj_exp_avg");
    free_device(mlp_proj_exp_avg_sq, "mlp_proj_exp_avg_sq");
    free_device(mlp_proj_bias_exp_avg, "mlp_proj_bias_exp_avg");
    free_device(mlp_proj_bias_exp_avg_sq, "mlp_proj_bias_exp_avg_sq");
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(cfg.model_family) << "\",\n"
        << "  \"backend\": \"tile-cuda\",\n"
        << "  \"smoke\": \"transformer_block_step\",\n"
        << "  \"token_shards_resolved\": false,\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"batch\": " << kBatch << ",\n"
        << "  \"heads\": " << kHeads << ",\n"
        << "  \"head_dim\": " << kHeadDim << ",\n"
        << "  \"seq\": " << kSeq << ",\n"
        << "  \"model_dim\": " << kDim << ",\n"
        << "  \"hidden_dim\": " << kHidden << ",\n"
        << "  \"weight_update_count\": 12,\n"
        << "  \"kernels\": [\n"
        << "    \"nfn_native_tile_layer_norm_float32\",\n"
        << "    \"nfn_native_tile_linear_float32\",\n"
        << "    \"nfn_native_tile_split_qkv_float32\",\n"
        << "    \"nfn_native_tile_reshape_heads_float32\",\n"
        << "    \"nfn_native_tile_scaled_dot_product_attention_float32\",\n"
        << "    \"nfn_native_tile_merge_heads_float32\",\n"
        << "    \"nfn_native_tile_scaled_residual_add_float32\",\n"
        << "    \"nfn_native_tile_gelu_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_weight_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_bias_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_input_float32\",\n"
        << "    \"nfn_native_tile_gelu_backward_float32\",\n"
        << "    \"nfn_native_tile_layer_norm_backward_affine_float32\",\n"
        << "    \"nfn_native_tile_layer_norm_backward_input_float32\",\n"
        << "    \"nfn_native_tile_gradient_accumulate_float32\",\n"
        << "    \"nfn_native_tile_scaled_dot_product_attention_backward_float32\",\n"
        << "    \"nfn_native_tile_merge_qkv_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"sample_residual2\": " << sampled_residual2 << ",\n"
        << "  \"sample_grad_x\": " << sampled_grad_x << ",\n"
        << "  \"sample_grad_qkv_weight\": " << sampled_grad_qkv_weight << ",\n"
        << "  \"sample_grad_qkv_bias\": " << sampled_grad_qkv_bias << ",\n"
        << "  \"sample_grad_attn_proj_weight\": " << sampled_grad_attn_proj_weight << ",\n"
        << "  \"sample_grad_attn_proj_bias\": " << sampled_grad_attn_proj_bias << ",\n"
        << "  \"sample_grad_fc_weight\": " << sampled_grad_fc_weight << ",\n"
        << "  \"sample_grad_fc_bias\": " << sampled_grad_fc_bias << ",\n"
        << "  \"sample_grad_mlp_proj_weight\": " << sampled_grad_mlp_proj_weight << ",\n"
        << "  \"sample_grad_mlp_proj_bias\": " << sampled_grad_mlp_proj_bias << ",\n"
        << "  \"sample_updated_ln1_weight\": " << sampled_ln1_weight << ",\n"
        << "  \"sample_updated_ln1_bias\": " << sampled_ln1_bias << ",\n"
        << "  \"sample_updated_ln2_weight\": " << sampled_ln2_weight << ",\n"
        << "  \"sample_updated_ln2_bias\": " << sampled_ln2_bias << ",\n"
        << "  \"sample_updated_qkv_weight\": " << sampled_qkv_weight << ",\n"
        << "  \"sample_updated_qkv_bias\": " << sampled_qkv_bias << ",\n"
        << "  \"sample_updated_attn_proj_weight\": " << sampled_attn_proj_weight << ",\n"
        << "  \"sample_updated_attn_proj_bias\": " << sampled_attn_proj_bias << ",\n"
        << "  \"sample_updated_fc_weight\": " << sampled_fc_weight << ",\n"
        << "  \"sample_updated_fc_bias\": " << sampled_fc_bias << ",\n"
        << "  \"sample_updated_mlp_proj_weight\": " << sampled_mlp_proj_weight << ",\n"
        << "  \"sample_updated_mlp_proj_bias\": " << sampled_mlp_proj_bias << ",\n"
        << "  \"max_abs_sample\": " << max_abs_sample << ",\n"
        << "  \"max_weight_delta\": " << max_weight_delta << ",\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";

    return passed ? 0 : 2;
}

int print_transformer_lm_step_smoke_json(
    const Config& cfg,
    const neuralfn::native_train::TokenShardDataset& dataset,
    const char* program) {
    constexpr std::int64_t kBatch = 1;
    constexpr std::int64_t kHeads = 1;
    constexpr std::int64_t kSeq = 2;
    constexpr std::int64_t kVocab = 50257;
    constexpr std::int64_t kPaddedVocab = 50304;
    constexpr std::int64_t kDim = 4;
    constexpr std::int64_t kHeadDim = kDim / kHeads;
    constexpr std::int64_t kRows = kBatch * kSeq;
    constexpr std::int64_t kHidden = kDim * 2;
    constexpr std::int64_t kQkvDim = kDim * 3;
    constexpr std::int64_t kActivationElements = kRows * kDim;
    constexpr std::int64_t kHiddenElements = kRows * kHidden;
    constexpr std::int64_t kQkvActivationElements = kRows * kQkvDim;
    constexpr std::int64_t kTokenWeightElements = kPaddedVocab * kDim;
    constexpr std::int64_t kPositionWeightElements = kSeq * kDim;
    constexpr std::int64_t kQkvWeightElements = kQkvDim * kDim;
    constexpr std::int64_t kAttnProjWeightElements = kDim * kDim;
    constexpr std::int64_t kFcWeightElements = kHidden * kDim;
    constexpr std::int64_t kMlpProjWeightElements = kDim * kHidden;
    constexpr std::int64_t kLogitElements = kRows * kPaddedVocab;
    constexpr float kInitialPositionWeight = 0.02f;
    constexpr float kLnWeight = 1.0f;
    constexpr float kLnBias = 0.0f;
    constexpr float kQkvWeight = 0.015f;
    constexpr float kAttnProjWeight = 0.01f;
    constexpr float kFcWeight = 0.02f;
    constexpr float kMlpProjWeight = 0.01f;
    constexpr float kResidualScale = 1.0f;
    constexpr float kLearningRate = 0.1f;
    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.95f;
    constexpr float kEps = 1e-8f;
    constexpr float kNormEps = 1e-5f;
    constexpr float kWeightDecay = 0.1f;
    constexpr float kBiasCorrection1 = 1.0f - kBeta1;
    constexpr float kSqrtBiasCorrection2 = 0.22360679775f;
    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;

    neuralfn::native_train::BatchPlan batch_plan;
    neuralfn::native_train::TokenBatch batch;
    bool have_batch = false;
    std::string error;
    try {
        batch_plan = neuralfn::native_train::build_batch_plan(dataset, kSeq, kBatch, kRows);
        neuralfn::native_train::SequentialTokenBatchSampler sampler(dataset.train_shards, kSeq, kBatch);
        have_batch = sampler.next(batch);
        if (!have_batch) {
            error = "not enough train tokens to build one GPT-2 transformer/LM smoke batch";
        }
    } catch (const std::exception& exc) {
        error = exc.what();
    }

    std::int64_t host_tokens[kRows] = {0, 1};
    std::int64_t host_targets[kRows] = {1, 2};
    if (error.empty()) {
        if (batch.tokens.size() != static_cast<std::size_t>(kRows) ||
            batch.targets.size() != static_cast<std::size_t>(kRows)) {
            error = "sampled GPT-2 transformer/LM smoke batch has unexpected size";
        } else {
            for (std::int64_t i = 0; i < kRows; ++i) {
                const std::int64_t token = static_cast<std::int64_t>(batch.tokens[static_cast<std::size_t>(i)]);
                const std::int64_t target = static_cast<std::int64_t>(batch.targets[static_cast<std::size_t>(i)]);
                if (token < 0 || token >= kVocab || target < 0 || target >= kVocab) {
                    std::ostringstream out;
                    out << "sampled GPT-2 transformer/LM smoke token id out of range at row " << i
                        << ": token=" << token << " target=" << target << " vocab=" << kVocab;
                    error = out.str();
                    break;
                }
                host_tokens[i] = token;
                host_targets[i] = target;
            }
        }
    }

    std::vector<float> host_token_weight(static_cast<std::size_t>(kTokenWeightElements), 0.0f);
    for (std::int64_t i = 0; i < kTokenWeightElements; ++i) {
        host_token_weight[static_cast<std::size_t>(i)] = static_cast<float>((i % 17) - 8) * 0.01f;
    }
    const float initial_token_weight_sample = host_token_weight[0];

    const std::string tile_lib_path = cfg.tile_ops_lib.empty() ? default_tile_ops_lib(program) : cfg.tile_ops_lib;
    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(cfg);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool passed = false;
    double max_abs_sample = 0.0;
    double max_weight_delta = 0.0;

    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using GradientAccumulateFn = int (*)(float*, const float*, std::int64_t, float, void*);
    using TokenEmbeddingFn = int (*)(const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using TokenEmbeddingBackwardWeightFn = int (*)(
        const std::int64_t*, const float*, float*, std::int64_t, std::int64_t, void*);
    using PositionEmbeddingFn = int (*)(const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using PositionEmbeddingBackwardFn = int (*)(const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using ResidualAddFn = int (*)(const float*, const float*, const float*, float*, std::int64_t, void*);
    using SplitQkvFn = int (*)(const float*, float*, float*, float*, std::int64_t, std::int64_t, void*);
    using MergeQkvFn = int (*)(const float*, const float*, const float*, float*, std::int64_t, std::int64_t, void*);
    using ReshapeHeadsFn = int (*)(
        const float*, float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using MergeHeadsFn = int (*)(
        const float*, float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using LayerNormFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, float, void*);
    using LayerNormBackwardInputFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, float, void*);
    using LayerNormBackwardAffineFn = int (*)(
        const float*, const float*, float*, float*, std::int64_t, std::int64_t, float, void*);
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBackwardInputFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardBiasFn = int (*)(const float*, float*, std::int64_t, std::int64_t, void*);
    using GeluFn = int (*)(const float*, float*, std::int64_t, void*);
    using GeluBackwardFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
    using AttentionFn = int (*)(
        const float*, const float*, const float*, float*,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        std::int64_t, std::int64_t, std::int64_t, float, bool, bool, bool,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using AttentionBackwardFn = int (*)(
        const float*, const float*, const float*, const float*, float*, float*, float*,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        std::int64_t, std::int64_t, std::int64_t, float, bool, bool, bool,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyPartialsFn = int (*)(
        const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyBackwardWorkspaceFn = int (*)(
        const float*, const std::int64_t*, float*, float*, float*,
        std::int64_t, std::int64_t, float, void*);
    using AdamWFn = int (*)(
        float*, const float*, float*, float*, std::int64_t,
        float, float, float, float, float, float, float, void*);
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);

    void* tile_handle = nullptr;
    void* cuda_handle = nullptr;
    FillFn fill = nullptr;
    GradientAccumulateFn gradient_accumulate = nullptr;
    TokenEmbeddingFn token_embedding = nullptr;
    TokenEmbeddingBackwardWeightFn token_embedding_backward_weight = nullptr;
    PositionEmbeddingFn position_embedding = nullptr;
    PositionEmbeddingBackwardFn position_embedding_backward = nullptr;
    ResidualAddFn residual_add = nullptr;
    SplitQkvFn split_qkv = nullptr;
    MergeQkvFn merge_qkv = nullptr;
    ReshapeHeadsFn reshape_heads = nullptr;
    MergeHeadsFn merge_heads = nullptr;
    LayerNormFn layer_norm = nullptr;
    LayerNormBackwardInputFn layer_norm_backward_input = nullptr;
    LayerNormBackwardAffineFn layer_norm_backward_affine = nullptr;
    LinearFn linear = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardWeightFn linear_backward_weight = nullptr;
    LinearBackwardBiasFn linear_backward_bias = nullptr;
    GeluFn gelu = nullptr;
    GeluBackwardFn gelu_backward = nullptr;
    AttentionFn attention = nullptr;
    AttentionBackwardFn attention_backward = nullptr;
    TokenCrossEntropyPartialsFn ce_partials = nullptr;
    TokenCrossEntropyBackwardWorkspaceFn ce_backward_workspace = nullptr;
    AdamWFn adamw = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed with CUDA error " << code;
        if (cuda_get_error_string != nullptr) {
            const char* message = cuda_get_error_string(code);
            if (message != nullptr) {
                out << ": " << message;
            }
        }
        return out.str();
    };

    if (error.empty()) {
        tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (tile_handle == nullptr) {
            error = dl_last_error("dlopen tile ops failed");
        } else {
            tile_loaded = true;
            fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
            gradient_accumulate = load_symbol<GradientAccumulateFn>(
                tile_handle, "nfn_native_tile_gradient_accumulate_float32");
            token_embedding = load_symbol<TokenEmbeddingFn>(tile_handle, "nfn_native_tile_token_embedding_float32");
            token_embedding_backward_weight = load_symbol<TokenEmbeddingBackwardWeightFn>(
                tile_handle, "nfn_native_tile_token_embedding_backward_weight_float32");
            position_embedding = load_symbol<PositionEmbeddingFn>(
                tile_handle, "nfn_native_tile_absolute_position_embedding_float32");
            position_embedding_backward = load_symbol<PositionEmbeddingBackwardFn>(
                tile_handle, "nfn_native_tile_absolute_position_embedding_backward_float32");
            residual_add = load_symbol<ResidualAddFn>(tile_handle, "nfn_native_tile_scaled_residual_add_float32");
            split_qkv = load_symbol<SplitQkvFn>(tile_handle, "nfn_native_tile_split_qkv_float32");
            merge_qkv = load_symbol<MergeQkvFn>(tile_handle, "nfn_native_tile_merge_qkv_float32");
            reshape_heads = load_symbol<ReshapeHeadsFn>(tile_handle, "nfn_native_tile_reshape_heads_float32");
            merge_heads = load_symbol<MergeHeadsFn>(tile_handle, "nfn_native_tile_merge_heads_float32");
            layer_norm = load_symbol<LayerNormFn>(tile_handle, "nfn_native_tile_layer_norm_float32");
            layer_norm_backward_input = load_symbol<LayerNormBackwardInputFn>(
                tile_handle, "nfn_native_tile_layer_norm_backward_input_float32");
            layer_norm_backward_affine = load_symbol<LayerNormBackwardAffineFn>(
                tile_handle, "nfn_native_tile_layer_norm_backward_affine_float32");
            linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
            linear_backward_input = load_symbol<LinearBackwardInputFn>(
                tile_handle, "nfn_native_tile_linear_backward_input_float32");
            linear_backward_weight = load_symbol<LinearBackwardWeightFn>(
                tile_handle, "nfn_native_tile_linear_backward_weight_float32");
            linear_backward_bias = load_symbol<LinearBackwardBiasFn>(
                tile_handle, "nfn_native_tile_linear_backward_bias_float32");
            gelu = load_symbol<GeluFn>(tile_handle, "nfn_native_tile_gelu_float32");
            gelu_backward = load_symbol<GeluBackwardFn>(tile_handle, "nfn_native_tile_gelu_backward_float32");
            attention = load_symbol<AttentionFn>(tile_handle, "nfn_native_tile_scaled_dot_product_attention_float32");
            attention_backward = load_symbol<AttentionBackwardFn>(
                tile_handle, "nfn_native_tile_scaled_dot_product_attention_backward_float32");
            ce_partials = load_symbol<TokenCrossEntropyPartialsFn>(
                tile_handle, "nfn_native_tile_token_cross_entropy_partials_float32");
            ce_backward_workspace = load_symbol<TokenCrossEntropyBackwardWorkspaceFn>(
                tile_handle, "nfn_native_tile_token_cross_entropy_backward_with_workspace_float32");
            adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
            if (fill == nullptr || gradient_accumulate == nullptr ||
                token_embedding == nullptr || token_embedding_backward_weight == nullptr ||
                position_embedding == nullptr || position_embedding_backward == nullptr ||
                residual_add == nullptr || split_qkv == nullptr || merge_qkv == nullptr ||
                reshape_heads == nullptr || merge_heads == nullptr ||
                layer_norm == nullptr || layer_norm_backward_input == nullptr ||
                layer_norm_backward_affine == nullptr || linear == nullptr ||
                linear_backward_input == nullptr || linear_backward_weight == nullptr ||
                linear_backward_bias == nullptr || gelu == nullptr || gelu_backward == nullptr ||
                attention == nullptr || attention_backward == nullptr ||
                ce_partials == nullptr || ce_backward_workspace == nullptr || adamw == nullptr) {
                error = dl_last_error("dlsym GPT-2 transformer/LM-step kernels failed");
            }
        }
    }

    if (error.empty()) {
        std::string runtime_error;
        for (const std::string& candidate : runtime_candidates) {
            cuda_lib_path = candidate;
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_runtime_loaded = true;
                break;
            }
            runtime_error = dl_last_error("dlopen CUDA runtime failed");
        }
        if (cuda_handle == nullptr) {
            error = runtime_error.empty() ? "could not load CUDA runtime" : runtime_error;
        }
    }

    if (error.empty()) {
        cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
        cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
        cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
        cuda_device_synchronize = load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
        cuda_get_error_string = load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
        if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr || cuda_device_synchronize == nullptr) {
            error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaDeviceSynchronize";
        }
    }

    std::vector<float*> device_ptrs;
    auto allocate = [&](float** ptr, std::int64_t elements, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        const int status = cuda_malloc(
            reinterpret_cast<void**>(ptr),
            sizeof(float) * static_cast<std::size_t>(elements));
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc " + name);
        } else {
            device_ptrs.push_back(*ptr);
        }
    };
    auto allocate_i64 = [&](std::int64_t** ptr, std::int64_t elements, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        const int status = cuda_malloc(
            reinterpret_cast<void**>(ptr),
            sizeof(std::int64_t) * static_cast<std::size_t>(elements));
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc " + name);
        }
    };

    float *token_weight = nullptr, *position_weight = nullptr, *residual_scale = nullptr;
    float *ln1_weight = nullptr, *ln1_bias = nullptr, *ln2_weight = nullptr, *ln2_bias = nullptr;
    float *lnf_weight = nullptr, *lnf_bias = nullptr;
    float *qkv_weight = nullptr, *qkv_bias = nullptr, *attn_proj_weight = nullptr, *attn_proj_bias = nullptr;
    float *fc_weight = nullptr, *fc_bias = nullptr, *mlp_proj_weight = nullptr, *mlp_proj_bias = nullptr;
    float *grad_token_weight = nullptr, *grad_position_weight = nullptr;
    float *grad_ln1_weight = nullptr, *grad_ln1_bias = nullptr, *grad_ln2_weight = nullptr, *grad_ln2_bias = nullptr;
    float *grad_lnf_weight = nullptr, *grad_lnf_bias = nullptr;
    float *grad_qkv_weight = nullptr, *grad_qkv_bias = nullptr, *grad_attn_proj_weight = nullptr, *grad_attn_proj_bias = nullptr;
    float *grad_fc_weight = nullptr, *grad_fc_bias = nullptr, *grad_mlp_proj_weight = nullptr, *grad_mlp_proj_bias = nullptr;
    float *token_avg = nullptr, *token_avg_sq = nullptr, *position_avg = nullptr, *position_avg_sq = nullptr;
    float *ln1_weight_avg = nullptr, *ln1_weight_avg_sq = nullptr, *ln1_bias_avg = nullptr, *ln1_bias_avg_sq = nullptr;
    float *ln2_weight_avg = nullptr, *ln2_weight_avg_sq = nullptr, *ln2_bias_avg = nullptr, *ln2_bias_avg_sq = nullptr;
    float *lnf_weight_avg = nullptr, *lnf_weight_avg_sq = nullptr, *lnf_bias_avg = nullptr, *lnf_bias_avg_sq = nullptr;
    float *qkv_avg = nullptr, *qkv_avg_sq = nullptr, *qkv_bias_avg = nullptr, *qkv_bias_avg_sq = nullptr;
    float *attn_proj_avg = nullptr, *attn_proj_avg_sq = nullptr, *attn_proj_bias_avg = nullptr, *attn_proj_bias_avg_sq = nullptr;
    float *fc_avg = nullptr, *fc_avg_sq = nullptr, *fc_bias_avg = nullptr, *fc_bias_avg_sq = nullptr;
    float *mlp_proj_avg = nullptr, *mlp_proj_avg_sq = nullptr, *mlp_proj_bias_avg = nullptr, *mlp_proj_bias_avg_sq = nullptr;
    float *token_out = nullptr, *position_out = nullptr, *x = nullptr, *ln1_out = nullptr, *qkv = nullptr;
    float *q = nullptr, *k = nullptr, *v = nullptr, *q_heads = nullptr, *k_heads = nullptr, *v_heads = nullptr;
    float *attn_heads = nullptr, *attn_out = nullptr, *attn_proj = nullptr, *residual1 = nullptr;
    float *ln2_out = nullptr, *fc_out = nullptr, *act = nullptr, *mlp_out = nullptr, *residual2 = nullptr;
    float *lnf_out = nullptr, *logits = nullptr, *loss_partials = nullptr, *row_max = nullptr, *row_denom = nullptr;
    float *grad_logits = nullptr, *grad_lnf = nullptr, *grad_residual2 = nullptr, *grad_act = nullptr;
    float *grad_fc_out = nullptr, *grad_ln2 = nullptr, *grad_residual1_from_mlp = nullptr, *grad_residual1 = nullptr;
    float *grad_attn_out = nullptr, *grad_attn_heads = nullptr, *grad_q_heads = nullptr, *grad_k_heads = nullptr;
    float *grad_v_heads = nullptr, *grad_q = nullptr, *grad_k = nullptr, *grad_v = nullptr, *grad_qkv = nullptr;
    float *grad_ln1 = nullptr, *grad_x_from_attn = nullptr, *grad_x = nullptr;
    std::int64_t *token_ids = nullptr, *targets = nullptr;

    allocate(&token_weight, kTokenWeightElements, "token_weight");
    allocate(&position_weight, kPositionWeightElements, "position_weight");
    allocate(&residual_scale, 1, "residual_scale");
    allocate(&ln1_weight, kDim, "ln1_weight");
    allocate(&ln1_bias, kDim, "ln1_bias");
    allocate(&ln2_weight, kDim, "ln2_weight");
    allocate(&ln2_bias, kDim, "ln2_bias");
    allocate(&lnf_weight, kDim, "lnf_weight");
    allocate(&lnf_bias, kDim, "lnf_bias");
    allocate(&qkv_weight, kQkvWeightElements, "qkv_weight");
    allocate(&qkv_bias, kQkvDim, "qkv_bias");
    allocate(&attn_proj_weight, kAttnProjWeightElements, "attn_proj_weight");
    allocate(&attn_proj_bias, kDim, "attn_proj_bias");
    allocate(&fc_weight, kFcWeightElements, "fc_weight");
    allocate(&fc_bias, kHidden, "fc_bias");
    allocate(&mlp_proj_weight, kMlpProjWeightElements, "mlp_proj_weight");
    allocate(&mlp_proj_bias, kDim, "mlp_proj_bias");
    for (auto item : {
             std::pair<float**, std::int64_t>{&grad_token_weight, kTokenWeightElements},
             {&grad_position_weight, kPositionWeightElements},
             {&grad_ln1_weight, kDim},
             {&grad_ln1_bias, kDim},
             {&grad_ln2_weight, kDim},
             {&grad_ln2_bias, kDim},
             {&grad_lnf_weight, kDim},
             {&grad_lnf_bias, kDim},
             {&grad_qkv_weight, kQkvWeightElements},
             {&grad_qkv_bias, kQkvDim},
             {&grad_attn_proj_weight, kAttnProjWeightElements},
             {&grad_attn_proj_bias, kDim},
             {&grad_fc_weight, kFcWeightElements},
             {&grad_fc_bias, kHidden},
             {&grad_mlp_proj_weight, kMlpProjWeightElements},
             {&grad_mlp_proj_bias, kDim},
             {&token_avg, kTokenWeightElements},
             {&token_avg_sq, kTokenWeightElements},
             {&position_avg, kPositionWeightElements},
             {&position_avg_sq, kPositionWeightElements},
             {&ln1_weight_avg, kDim},
             {&ln1_weight_avg_sq, kDim},
             {&ln1_bias_avg, kDim},
             {&ln1_bias_avg_sq, kDim},
             {&ln2_weight_avg, kDim},
             {&ln2_weight_avg_sq, kDim},
             {&ln2_bias_avg, kDim},
             {&ln2_bias_avg_sq, kDim},
             {&lnf_weight_avg, kDim},
             {&lnf_weight_avg_sq, kDim},
             {&lnf_bias_avg, kDim},
             {&lnf_bias_avg_sq, kDim},
             {&qkv_avg, kQkvWeightElements},
             {&qkv_avg_sq, kQkvWeightElements},
             {&qkv_bias_avg, kQkvDim},
             {&qkv_bias_avg_sq, kQkvDim},
             {&attn_proj_avg, kAttnProjWeightElements},
             {&attn_proj_avg_sq, kAttnProjWeightElements},
             {&attn_proj_bias_avg, kDim},
             {&attn_proj_bias_avg_sq, kDim},
             {&fc_avg, kFcWeightElements},
             {&fc_avg_sq, kFcWeightElements},
             {&fc_bias_avg, kHidden},
             {&fc_bias_avg_sq, kHidden},
             {&mlp_proj_avg, kMlpProjWeightElements},
             {&mlp_proj_avg_sq, kMlpProjWeightElements},
             {&mlp_proj_bias_avg, kDim},
             {&mlp_proj_bias_avg_sq, kDim},
             {&token_out, kActivationElements},
             {&position_out, kActivationElements},
             {&x, kActivationElements},
             {&ln1_out, kActivationElements},
             {&qkv, kQkvActivationElements},
             {&q, kActivationElements},
             {&k, kActivationElements},
             {&v, kActivationElements},
             {&q_heads, kActivationElements},
             {&k_heads, kActivationElements},
             {&v_heads, kActivationElements},
             {&attn_heads, kActivationElements},
             {&attn_out, kActivationElements},
             {&attn_proj, kActivationElements},
             {&residual1, kActivationElements},
             {&ln2_out, kActivationElements},
             {&fc_out, kHiddenElements},
             {&act, kHiddenElements},
             {&mlp_out, kActivationElements},
             {&residual2, kActivationElements},
             {&lnf_out, kActivationElements},
             {&logits, kLogitElements},
             {&loss_partials, 1},
             {&row_max, kRows},
             {&row_denom, kRows},
             {&grad_logits, kLogitElements},
             {&grad_lnf, kActivationElements},
             {&grad_residual2, kActivationElements},
             {&grad_act, kHiddenElements},
             {&grad_fc_out, kHiddenElements},
             {&grad_ln2, kActivationElements},
             {&grad_residual1_from_mlp, kActivationElements},
             {&grad_residual1, kActivationElements},
             {&grad_attn_out, kActivationElements},
             {&grad_attn_heads, kActivationElements},
             {&grad_q_heads, kActivationElements},
             {&grad_k_heads, kActivationElements},
             {&grad_v_heads, kActivationElements},
             {&grad_q, kActivationElements},
             {&grad_k, kActivationElements},
             {&grad_v, kActivationElements},
             {&grad_qkv, kQkvActivationElements},
             {&grad_ln1, kActivationElements},
             {&grad_x_from_attn, kActivationElements},
             {&grad_x, kActivationElements},
         }) {
        allocate(item.first, item.second, "transformer_lm_buffer");
    }
    allocate_i64(&token_ids, kRows, "token_ids");
    allocate_i64(&targets, kRows, "targets");

    auto copy_to_device = [&](void* dst, const void* src, std::size_t bytes, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        const int status = cuda_memcpy(dst, src, bytes, kCudaMemcpyHostToDevice);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy host-to-device " + name);
        }
    };
    auto fill_buffer = [&](float* ptr, std::int64_t elements, float value, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        const int status = fill(ptr, elements, value, nullptr);
        if (status != 0) {
            error = cuda_error(status, "fill " + name);
        }
    };
    copy_to_device(token_weight, host_token_weight.data(), sizeof(float) * host_token_weight.size(), "token_weight");
    copy_to_device(token_ids, host_tokens, sizeof(host_tokens), "token_ids");
    copy_to_device(targets, host_targets, sizeof(host_targets), "targets");
    fill_buffer(position_weight, kPositionWeightElements, kInitialPositionWeight, "position_weight");
    fill_buffer(residual_scale, 1, kResidualScale, "residual_scale");
    fill_buffer(ln1_weight, kDim, kLnWeight, "ln1_weight");
    fill_buffer(ln1_bias, kDim, kLnBias, "ln1_bias");
    fill_buffer(ln2_weight, kDim, kLnWeight, "ln2_weight");
    fill_buffer(ln2_bias, kDim, kLnBias, "ln2_bias");
    fill_buffer(lnf_weight, kDim, kLnWeight, "lnf_weight");
    fill_buffer(lnf_bias, kDim, kLnBias, "lnf_bias");
    fill_buffer(qkv_weight, kQkvWeightElements, kQkvWeight, "qkv_weight");
    fill_buffer(qkv_bias, kQkvDim, 0.0f, "qkv_bias");
    fill_buffer(attn_proj_weight, kAttnProjWeightElements, kAttnProjWeight, "attn_proj_weight");
    fill_buffer(attn_proj_bias, kDim, 0.0f, "attn_proj_bias");
    fill_buffer(fc_weight, kFcWeightElements, kFcWeight, "fc_weight");
    fill_buffer(fc_bias, kHidden, 0.0f, "fc_bias");
    fill_buffer(mlp_proj_weight, kMlpProjWeightElements, kMlpProjWeight, "mlp_proj_weight");
    fill_buffer(mlp_proj_bias, kDim, 0.0f, "mlp_proj_bias");
    fill_buffer(grad_token_weight, kTokenWeightElements, 0.0f, "grad_token_weight");
    fill_buffer(grad_position_weight, kPositionWeightElements, 0.0f, "grad_position_weight");
    fill_buffer(grad_ln1_weight, kDim, 0.0f, "grad_ln1_weight");
    fill_buffer(grad_ln1_bias, kDim, 0.0f, "grad_ln1_bias");
    fill_buffer(grad_ln2_weight, kDim, 0.0f, "grad_ln2_weight");
    fill_buffer(grad_ln2_bias, kDim, 0.0f, "grad_ln2_bias");
    fill_buffer(grad_lnf_weight, kDim, 0.0f, "grad_lnf_weight");
    fill_buffer(grad_lnf_bias, kDim, 0.0f, "grad_lnf_bias");
    fill_buffer(grad_qkv_weight, kQkvWeightElements, 0.0f, "grad_qkv_weight");
    fill_buffer(grad_qkv_bias, kQkvDim, 0.0f, "grad_qkv_bias");
    fill_buffer(grad_attn_proj_weight, kAttnProjWeightElements, 0.0f, "grad_attn_proj_weight");
    fill_buffer(grad_attn_proj_bias, kDim, 0.0f, "grad_attn_proj_bias");
    fill_buffer(grad_fc_weight, kFcWeightElements, 0.0f, "grad_fc_weight");
    fill_buffer(grad_fc_bias, kHidden, 0.0f, "grad_fc_bias");
    fill_buffer(grad_mlp_proj_weight, kMlpProjWeightElements, 0.0f, "grad_mlp_proj_weight");
    fill_buffer(grad_mlp_proj_bias, kDim, 0.0f, "grad_mlp_proj_bias");
    for (auto item : {
             std::pair<float*, std::int64_t>{token_avg, kTokenWeightElements},
             {token_avg_sq, kTokenWeightElements},
             {position_avg, kPositionWeightElements},
             {position_avg_sq, kPositionWeightElements},
             {ln1_weight_avg, kDim},
             {ln1_weight_avg_sq, kDim},
             {ln1_bias_avg, kDim},
             {ln1_bias_avg_sq, kDim},
             {ln2_weight_avg, kDim},
             {ln2_weight_avg_sq, kDim},
             {ln2_bias_avg, kDim},
             {ln2_bias_avg_sq, kDim},
             {lnf_weight_avg, kDim},
             {lnf_weight_avg_sq, kDim},
             {lnf_bias_avg, kDim},
             {lnf_bias_avg_sq, kDim},
             {qkv_avg, kQkvWeightElements},
             {qkv_avg_sq, kQkvWeightElements},
             {qkv_bias_avg, kQkvDim},
             {qkv_bias_avg_sq, kQkvDim},
             {attn_proj_avg, kAttnProjWeightElements},
             {attn_proj_avg_sq, kAttnProjWeightElements},
             {attn_proj_bias_avg, kDim},
             {attn_proj_bias_avg_sq, kDim},
             {fc_avg, kFcWeightElements},
             {fc_avg_sq, kFcWeightElements},
             {fc_bias_avg, kHidden},
             {fc_bias_avg_sq, kHidden},
             {mlp_proj_avg, kMlpProjWeightElements},
             {mlp_proj_avg_sq, kMlpProjWeightElements},
             {mlp_proj_bias_avg, kDim},
             {mlp_proj_bias_avg_sq, kDim},
         }) {
        fill_buffer(item.first, item.second, 0.0f, "adamw_state");
    }

    auto run = [&](int status, const std::string& name) {
        if (status != 0 && error.empty()) {
            error = cuda_error(status, name);
        }
    };
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    if (error.empty()) run(token_embedding(token_weight, token_ids, token_out, kRows, kDim, nullptr), "wte.forward");
    if (error.empty()) run(position_embedding(position_weight, position_out, kBatch, kSeq, kDim, nullptr), "wpe.forward");
    if (error.empty()) run(residual_add(token_out, position_out, residual_scale, x, kActivationElements, nullptr), "embedding.residual");
    if (error.empty()) run(layer_norm(x, ln1_weight, ln1_bias, ln1_out, kRows, kDim, kNormEps, nullptr), "ln1.forward");
    if (error.empty()) run(linear(ln1_out, qkv_weight, qkv_bias, qkv, kRows, kDim, kQkvDim, true, nullptr), "attn.qkv.forward");
    if (error.empty()) run(split_qkv(qkv, q, k, v, kRows, kDim, nullptr), "attn.qkv.split");
    if (error.empty()) run(reshape_heads(q, q_heads, kBatch, kSeq, kHeads, kHeadDim, nullptr), "attn.q.reshape");
    if (error.empty()) run(reshape_heads(k, k_heads, kBatch, kSeq, kHeads, kHeadDim, nullptr), "attn.k.reshape");
    if (error.empty()) run(reshape_heads(v, v_heads, kBatch, kSeq, kHeads, kHeadDim, nullptr), "attn.v.reshape");
    if (error.empty()) {
        run(attention(q_heads, k_heads, v_heads, attn_heads, kActivationElements, kHeads, kHeads, kSeq, kSeq, kHeadDim, kHeadDim, attention_scale, true, false, false, 0, 0, 0, 0, nullptr), "attn.sdpa.forward");
    }
    if (error.empty()) run(merge_heads(attn_heads, attn_out, kBatch, kHeads, kSeq, kHeadDim, nullptr), "attn.merge_heads");
    if (error.empty()) run(linear(attn_out, attn_proj_weight, attn_proj_bias, attn_proj, kRows, kDim, kDim, true, nullptr), "attn.out.forward");
    if (error.empty()) run(residual_add(x, attn_proj, residual_scale, residual1, kActivationElements, nullptr), "attn.residual");
    if (error.empty()) run(layer_norm(residual1, ln2_weight, ln2_bias, ln2_out, kRows, kDim, kNormEps, nullptr), "ln2.forward");
    if (error.empty()) run(linear(ln2_out, fc_weight, fc_bias, fc_out, kRows, kDim, kHidden, true, nullptr), "mlp.fc.forward");
    if (error.empty()) run(gelu(fc_out, act, kHiddenElements, nullptr), "mlp.gelu.forward");
    if (error.empty()) run(linear(act, mlp_proj_weight, mlp_proj_bias, mlp_out, kRows, kHidden, kDim, true, nullptr), "mlp.proj.forward");
    if (error.empty()) run(residual_add(residual1, mlp_out, residual_scale, residual2, kActivationElements, nullptr), "mlp.residual");
    if (error.empty()) run(layer_norm(residual2, lnf_weight, lnf_bias, lnf_out, kRows, kDim, kNormEps, nullptr), "ln_f.forward");
    if (error.empty()) run(linear(lnf_out, token_weight, nullptr, logits, kRows, kDim, kPaddedVocab, false, nullptr), "lm_head.forward");
    if (error.empty()) run(ce_partials(logits, targets, loss_partials, kRows, kPaddedVocab, nullptr), "ce.forward");
    if (error.empty()) run(ce_backward_workspace(logits, targets, row_max, row_denom, grad_logits, kRows, kPaddedVocab, 1.0f / static_cast<float>(kRows), nullptr), "ce.backward");
    if (error.empty()) run(linear_backward_input(grad_logits, token_weight, grad_lnf, kRows, kDim, kPaddedVocab, nullptr), "lm_head.backward_input");
    if (error.empty()) run(linear_backward_weight(lnf_out, grad_logits, grad_token_weight, kRows, kDim, kPaddedVocab, nullptr), "lm_head.backward_weight");
    if (error.empty()) run(layer_norm_backward_affine(residual2, grad_lnf, grad_lnf_weight, grad_lnf_bias, kRows, kDim, kNormEps, nullptr), "ln_f.backward_affine");
    if (error.empty()) run(layer_norm_backward_input(residual2, grad_lnf, lnf_weight, grad_residual2, kRows, kDim, kNormEps, nullptr), "ln_f.backward_input");
    if (error.empty()) run(linear_backward_weight(act, grad_residual2, grad_mlp_proj_weight, kRows, kHidden, kDim, nullptr), "mlp.proj.backward_weight");
    if (error.empty()) run(linear_backward_bias(grad_residual2, grad_mlp_proj_bias, kRows, kDim, nullptr), "mlp.proj.backward_bias");
    if (error.empty()) run(linear_backward_input(grad_residual2, mlp_proj_weight, grad_act, kRows, kHidden, kDim, nullptr), "mlp.proj.backward_input");
    if (error.empty()) run(gelu_backward(fc_out, grad_act, grad_fc_out, kHiddenElements, nullptr), "mlp.gelu.backward");
    if (error.empty()) run(linear_backward_weight(ln2_out, grad_fc_out, grad_fc_weight, kRows, kDim, kHidden, nullptr), "mlp.fc.backward_weight");
    if (error.empty()) run(linear_backward_bias(grad_fc_out, grad_fc_bias, kRows, kHidden, nullptr), "mlp.fc.backward_bias");
    if (error.empty()) run(linear_backward_input(grad_fc_out, fc_weight, grad_ln2, kRows, kDim, kHidden, nullptr), "mlp.fc.backward_input");
    if (error.empty()) run(layer_norm_backward_affine(residual1, grad_ln2, grad_ln2_weight, grad_ln2_bias, kRows, kDim, kNormEps, nullptr), "ln2.backward_affine");
    if (error.empty()) run(layer_norm_backward_input(residual1, grad_ln2, ln2_weight, grad_residual1_from_mlp, kRows, kDim, kNormEps, nullptr), "ln2.backward_input");
    fill_buffer(grad_residual1, kActivationElements, 0.0f, "grad_residual1");
    if (error.empty()) run(gradient_accumulate(grad_residual1, grad_residual2, kActivationElements, 1.0f, nullptr), "mlp.residual.direct");
    if (error.empty()) run(gradient_accumulate(grad_residual1, grad_residual1_from_mlp, kActivationElements, 1.0f, nullptr), "mlp.residual.backward");
    if (error.empty()) run(linear_backward_weight(attn_out, grad_residual1, grad_attn_proj_weight, kRows, kDim, kDim, nullptr), "attn.out.backward_weight");
    if (error.empty()) run(linear_backward_bias(grad_residual1, grad_attn_proj_bias, kRows, kDim, nullptr), "attn.out.backward_bias");
    if (error.empty()) run(linear_backward_input(grad_residual1, attn_proj_weight, grad_attn_out, kRows, kDim, kDim, nullptr), "attn.out.backward_input");
    if (error.empty()) run(reshape_heads(grad_attn_out, grad_attn_heads, kBatch, kSeq, kHeads, kHeadDim, nullptr), "attn.grad.reshape");
    if (error.empty()) {
        run(attention_backward(q_heads, k_heads, v_heads, grad_attn_heads, grad_q_heads, grad_k_heads, grad_v_heads, kBatch, kHeads, kHeads, kSeq, kSeq, kHeadDim, kHeadDim, attention_scale, true, false, false, 0, 0, 0, 0, nullptr), "attn.sdpa.backward");
    }
    if (error.empty()) run(merge_heads(grad_q_heads, grad_q, kBatch, kHeads, kSeq, kHeadDim, nullptr), "attn.grad_q.merge");
    if (error.empty()) run(merge_heads(grad_k_heads, grad_k, kBatch, kHeads, kSeq, kHeadDim, nullptr), "attn.grad_k.merge");
    if (error.empty()) run(merge_heads(grad_v_heads, grad_v, kBatch, kHeads, kSeq, kHeadDim, nullptr), "attn.grad_v.merge");
    if (error.empty()) run(merge_qkv(grad_q, grad_k, grad_v, grad_qkv, kRows, kDim, nullptr), "attn.qkv.merge_grad");
    if (error.empty()) run(linear_backward_weight(ln1_out, grad_qkv, grad_qkv_weight, kRows, kDim, kQkvDim, nullptr), "attn.qkv.backward_weight");
    if (error.empty()) run(linear_backward_bias(grad_qkv, grad_qkv_bias, kRows, kQkvDim, nullptr), "attn.qkv.backward_bias");
    if (error.empty()) run(linear_backward_input(grad_qkv, qkv_weight, grad_ln1, kRows, kDim, kQkvDim, nullptr), "attn.qkv.backward_input");
    if (error.empty()) run(layer_norm_backward_affine(x, grad_ln1, grad_ln1_weight, grad_ln1_bias, kRows, kDim, kNormEps, nullptr), "ln1.backward_affine");
    if (error.empty()) run(layer_norm_backward_input(x, grad_ln1, ln1_weight, grad_x_from_attn, kRows, kDim, kNormEps, nullptr), "ln1.backward_input");
    fill_buffer(grad_x, kActivationElements, 0.0f, "grad_x");
    if (error.empty()) run(gradient_accumulate(grad_x, grad_residual1, kActivationElements, 1.0f, nullptr), "attn.residual.direct");
    if (error.empty()) run(gradient_accumulate(grad_x, grad_x_from_attn, kActivationElements, 1.0f, nullptr), "attn.residual.backward");
    if (error.empty()) run(token_embedding_backward_weight(token_ids, grad_x, grad_token_weight, kRows, kDim, nullptr), "wte.backward_weight");
    if (error.empty()) run(position_embedding_backward(grad_x, grad_position_weight, kBatch, kSeq, kDim, nullptr), "wpe.backward_weight");

    auto step = [&](float* param, float* grad, float* avg, float* avg_sq, std::int64_t elements, float decay, const std::string& name) {
        if (error.empty()) {
            run(adamw(param, grad, avg, avg_sq, elements, kLearningRate, kBeta1, kBeta2, kEps, decay, kBiasCorrection1, kSqrtBiasCorrection2, nullptr), name);
        }
    };
    step(token_weight, grad_token_weight, token_avg, token_avg_sq, kTokenWeightElements, kWeightDecay, "wte.adamw");
    step(position_weight, grad_position_weight, position_avg, position_avg_sq, kPositionWeightElements, kWeightDecay, "wpe.adamw");
    step(ln1_weight, grad_ln1_weight, ln1_weight_avg, ln1_weight_avg_sq, kDim, 0.0f, "ln1.weight.adamw");
    step(ln1_bias, grad_ln1_bias, ln1_bias_avg, ln1_bias_avg_sq, kDim, 0.0f, "ln1.bias.adamw");
    step(qkv_weight, grad_qkv_weight, qkv_avg, qkv_avg_sq, kQkvWeightElements, kWeightDecay, "attn.qkv.weight.adamw");
    step(qkv_bias, grad_qkv_bias, qkv_bias_avg, qkv_bias_avg_sq, kQkvDim, 0.0f, "attn.qkv.bias.adamw");
    step(attn_proj_weight, grad_attn_proj_weight, attn_proj_avg, attn_proj_avg_sq, kAttnProjWeightElements, kWeightDecay, "attn.out.weight.adamw");
    step(attn_proj_bias, grad_attn_proj_bias, attn_proj_bias_avg, attn_proj_bias_avg_sq, kDim, 0.0f, "attn.out.bias.adamw");
    step(ln2_weight, grad_ln2_weight, ln2_weight_avg, ln2_weight_avg_sq, kDim, 0.0f, "ln2.weight.adamw");
    step(ln2_bias, grad_ln2_bias, ln2_bias_avg, ln2_bias_avg_sq, kDim, 0.0f, "ln2.bias.adamw");
    step(fc_weight, grad_fc_weight, fc_avg, fc_avg_sq, kFcWeightElements, kWeightDecay, "mlp.fc.weight.adamw");
    step(fc_bias, grad_fc_bias, fc_bias_avg, fc_bias_avg_sq, kHidden, 0.0f, "mlp.fc.bias.adamw");
    step(mlp_proj_weight, grad_mlp_proj_weight, mlp_proj_avg, mlp_proj_avg_sq, kMlpProjWeightElements, kWeightDecay, "mlp.proj.weight.adamw");
    step(mlp_proj_bias, grad_mlp_proj_bias, mlp_proj_bias_avg, mlp_proj_bias_avg_sq, kDim, 0.0f, "mlp.proj.bias.adamw");
    step(lnf_weight, grad_lnf_weight, lnf_weight_avg, lnf_weight_avg_sq, kDim, 0.0f, "ln_f.weight.adamw");
    step(lnf_bias, grad_lnf_bias, lnf_bias_avg, lnf_bias_avg_sq, kDim, 0.0f, "ln_f.bias.adamw");
    if (error.empty()) run(cuda_device_synchronize(), "cudaDeviceSynchronize");

    auto copy_float_sample = [&](const float* device_ptr, std::int64_t offset, const std::string& name) -> float {
        float value = 0.0f;
        if (!error.empty()) {
            return value;
        }
        const int status = cuda_memcpy(&value, device_ptr + offset, sizeof(float), kCudaMemcpyDeviceToHost);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy sample " + name);
        }
        return value;
    };
    float sampled_loss = 0.0f;
    float sampled_residual2 = 0.0f;
    float sampled_grad_x = 0.0f;
    float sampled_grad_qkv_weight = 0.0f;
    float sampled_grad_token_weight = 0.0f;
    float sampled_token_weight = 0.0f;
    float sampled_qkv_weight = 0.0f;
    float sampled_lnf_weight = 0.0f;
    if (error.empty()) {
        sampled_loss = copy_float_sample(loss_partials, 0, "loss");
        sampled_residual2 = copy_float_sample(residual2, 0, "residual2");
        sampled_grad_x = copy_float_sample(grad_x, 0, "grad_x");
        sampled_grad_qkv_weight = copy_float_sample(grad_qkv_weight, 0, "grad_qkv_weight");
        sampled_grad_token_weight = copy_float_sample(grad_token_weight, host_targets[0] * kDim, "grad_token_weight");
        sampled_token_weight = copy_float_sample(token_weight, 0, "token_weight");
        sampled_qkv_weight = copy_float_sample(qkv_weight, 0, "qkv_weight");
        sampled_lnf_weight = copy_float_sample(lnf_weight, 0, "lnf_weight");
    }

    auto record_abs = [&](float value) {
        if (!std::isfinite(value)) {
            return false;
        }
        const double abs_value = std::fabs(static_cast<double>(value));
        if (abs_value > max_abs_sample) {
            max_abs_sample = abs_value;
        }
        return true;
    };
    if (error.empty()) {
        const bool finite_samples =
            record_abs(sampled_loss) && record_abs(sampled_residual2) &&
            record_abs(sampled_grad_x) && record_abs(sampled_grad_qkv_weight) &&
            record_abs(sampled_grad_token_weight) && record_abs(sampled_token_weight) &&
            record_abs(sampled_qkv_weight) && record_abs(sampled_lnf_weight);
        max_weight_delta = std::fabs(static_cast<double>(sampled_token_weight) - initial_token_weight_sample);
        const double qkv_delta = std::fabs(static_cast<double>(sampled_qkv_weight) - kQkvWeight);
        const double lnf_delta = std::fabs(static_cast<double>(sampled_lnf_weight) - kLnWeight);
        if (qkv_delta > max_weight_delta) {
            max_weight_delta = qkv_delta;
        }
        if (lnf_delta > max_weight_delta) {
            max_weight_delta = lnf_delta;
        }
        passed = finite_samples && sampled_loss > 0.0f && max_abs_sample > 0.0 && max_weight_delta > 0.0;
        if (!passed) {
            std::ostringstream out_message;
            out_message << "GPT-2 transformer/LM smoke did not produce finite loss and weight updates: sample="
                        << max_abs_sample << " loss=" << sampled_loss
                        << " weight_delta=" << max_weight_delta;
            error = out_message.str();
        }
    }

    for (float* ptr : device_ptrs) {
        if (ptr != nullptr && cuda_free != nullptr) {
            const int status = cuda_free(ptr);
            if (status != 0 && error.empty()) {
                error = cuda_error(status, "cudaFree transformer_lm_buffer");
            }
        }
    }
    if (token_ids != nullptr && cuda_free != nullptr) {
        const int status = cuda_free(token_ids);
        if (status != 0 && error.empty()) {
            error = cuda_error(status, "cudaFree token_ids");
        }
    }
    if (targets != nullptr && cuda_free != nullptr) {
        const int status = cuda_free(targets);
        if (status != 0 && error.empty()) {
            error = cuda_error(status, "cudaFree targets");
        }
    }
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(cfg.model_family) << "\",\n"
        << "  \"backend\": \"tile-cuda\",\n"
        << "  \"smoke\": \"transformer_lm_step\",\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"dataset_loaded\": true,\n"
        << "  \"batch_loaded\": " << (have_batch ? "true" : "false") << ",\n"
        << "  \"token_shards\": " << neuralfn::native_train::token_shard_dataset_json(dataset, &batch_plan) << ",\n"
        << "  \"sample_batch\": ";
    if (have_batch) {
        std::cout << neuralfn::native_train::token_batch_json(batch);
    } else {
        std::cout << "null";
    }
    std::cout
        << ",\n"
        << "  \"batch\": " << kBatch << ",\n"
        << "  \"heads\": " << kHeads << ",\n"
        << "  \"head_dim\": " << kHeadDim << ",\n"
        << "  \"seq\": " << kSeq << ",\n"
        << "  \"rows\": " << kRows << ",\n"
        << "  \"vocab\": " << kVocab << ",\n"
        << "  \"padded_vocab\": " << kPaddedVocab << ",\n"
        << "  \"model_dim\": " << kDim << ",\n"
        << "  \"hidden_dim\": " << kHidden << ",\n"
        << "  \"weight_update_count\": 16,\n"
        << "  \"kernels\": [\n"
        << "    \"nfn_native_tile_token_embedding_float32\",\n"
        << "    \"nfn_native_tile_absolute_position_embedding_float32\",\n"
        << "    \"nfn_native_tile_layer_norm_float32\",\n"
        << "    \"nfn_native_tile_linear_float32\",\n"
        << "    \"nfn_native_tile_split_qkv_float32\",\n"
        << "    \"nfn_native_tile_reshape_heads_float32\",\n"
        << "    \"nfn_native_tile_scaled_dot_product_attention_float32\",\n"
        << "    \"nfn_native_tile_merge_heads_float32\",\n"
        << "    \"nfn_native_tile_scaled_residual_add_float32\",\n"
        << "    \"nfn_native_tile_gelu_float32\",\n"
        << "    \"nfn_native_tile_token_cross_entropy_partials_float32\",\n"
        << "    \"nfn_native_tile_token_cross_entropy_backward_with_workspace_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_weight_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_bias_float32\",\n"
        << "    \"nfn_native_tile_linear_backward_input_float32\",\n"
        << "    \"nfn_native_tile_layer_norm_backward_affine_float32\",\n"
        << "    \"nfn_native_tile_layer_norm_backward_input_float32\",\n"
        << "    \"nfn_native_tile_gradient_accumulate_float32\",\n"
        << "    \"nfn_native_tile_scaled_dot_product_attention_backward_float32\",\n"
        << "    \"nfn_native_tile_merge_qkv_float32\",\n"
        << "    \"nfn_native_tile_token_embedding_backward_weight_float32\",\n"
        << "    \"nfn_native_tile_absolute_position_embedding_backward_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"sample_loss\": " << sampled_loss << ",\n"
        << "  \"sample_residual2\": " << sampled_residual2 << ",\n"
        << "  \"sample_grad_x\": " << sampled_grad_x << ",\n"
        << "  \"sample_grad_qkv_weight\": " << sampled_grad_qkv_weight << ",\n"
        << "  \"sample_grad_token_weight\": " << sampled_grad_token_weight << ",\n"
        << "  \"sample_updated_token_weight\": " << sampled_token_weight << ",\n"
        << "  \"sample_updated_qkv_weight\": " << sampled_qkv_weight << ",\n"
        << "  \"sample_updated_lnf_weight\": " << sampled_lnf_weight << ",\n"
        << "  \"max_abs_sample\": " << max_abs_sample << ",\n"
        << "  \"max_weight_delta\": " << max_weight_delta << ",\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";

    return passed ? 0 : 2;
}

int write_checkpoint_metadata_smoke_json(
    const Config& cfg,
    const neuralfn::native_train::TokenShardDataset& dataset) {
    constexpr std::int64_t kVocab = 50257;
    constexpr std::int64_t kPaddedVocab = 50304;
    constexpr std::int64_t kHeads = 12;
    constexpr std::int64_t kDim = 768;
    constexpr std::int64_t kVersion = 5;
    constexpr std::int64_t kBytesPerParam = 2;
    constexpr std::int64_t kHeaderBytes = 256 * 4;

    const std::int64_t max_seq_len = cfg.seq_len > 0 ? cfg.seq_len : 1024;
    const std::int64_t target_layers = cfg.num_layers > 0 ? cfg.num_layers : 12;
    const std::int64_t checkpoint_step = cfg.max_steps > 0 ? cfg.max_steps : 0;
    const fs::path output_dir = cfg.output_dir.empty() ? fs::path(default_output_dir()) : fs::path(cfg.output_dir);
    std::ostringstream checkpoint_name;
    checkpoint_name << "model_" << std::setw(8) << std::setfill('0') << checkpoint_step << ".bin";
    const fs::path checkpoint_path = output_dir / checkpoint_name.str();
    const fs::path done_marker = output_dir / ("DONE_" + checkpoint_name.str().substr(6, 8));

    std::string error;
    const bool checkpoint_written = write_sparse_native_gpt2_checkpoint(
        checkpoint_path,
        max_seq_len,
        kVocab,
        target_layers,
        kHeads,
        kDim,
        kPaddedVocab,
        &error);
    bool done_written = false;
    if (checkpoint_written) {
        std::ofstream done(done_marker, std::ios::trunc);
        done.close();
        done_written = static_cast<bool>(done);
        if (!done_written) {
            error = "failed to write DONE marker: " + done_marker.string();
        }
    }

    const std::int64_t parameter_count =
        native_gpt2_parameter_count(max_seq_len, kPaddedVocab, target_layers, kDim);
    const std::int64_t parameter_bytes = parameter_count * kBytesPerParam;
    const std::int64_t expected_file_size = kHeaderBytes + parameter_bytes;
    const std::int64_t actual_file_size =
        checkpoint_written && fs::exists(checkpoint_path)
            ? static_cast<std::int64_t>(fs::file_size(checkpoint_path))
            : 0;
    const bool passed = checkpoint_written && done_written && actual_file_size == expected_file_size;

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(cfg.model_family) << "\",\n"
        << "  \"backend\": \"tile-cuda\",\n"
        << "  \"status\": \"" << (passed ? "native-checkpoint-metadata-written" : "native-checkpoint-metadata-failed") << "\",\n"
        << "  \"checkpoint_metadata_smoke\": true,\n"
        << "  \"dataset_path\": \"" << json_escape(dataset.dataset_path.string()) << "\",\n"
        << "  \"output_dir\": \"" << json_escape(output_dir.string()) << "\",\n"
        << "  \"checkpoint_path\": \"" << json_escape(checkpoint_path.string()) << "\",\n"
        << "  \"done_marker\": \"" << json_escape(done_marker.string()) << "\",\n"
        << "  \"checkpoint_step\": " << checkpoint_step << ",\n"
        << "  \"version\": " << kVersion << ",\n"
        << "  \"precision\": \"bf16\",\n"
        << "  \"metadata_only\": true,\n"
        << "  \"trained_layers\": 0,\n"
        << "  \"target_layers\": " << target_layers << ",\n"
        << "  \"num_layers\": " << target_layers << ",\n"
        << "  \"num_heads\": " << kHeads << ",\n"
        << "  \"vocab\": " << kVocab << ",\n"
        << "  \"padded_vocab\": " << kPaddedVocab << ",\n"
        << "  \"model_dim\": " << kDim << ",\n"
        << "  \"max_seq_len\": " << max_seq_len << ",\n"
        << "  \"parameter_count\": " << parameter_count << ",\n"
        << "  \"parameter_bytes\": " << parameter_bytes << ",\n"
        << "  \"expected_file_size\": " << expected_file_size << ",\n"
        << "  \"actual_file_size\": " << actual_file_size << ",\n"
        << "  \"size_matches\": " << (actual_file_size == expected_file_size ? "true" : "false") << ",\n"
        << "  \"sparse_body\": true,\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";
    return passed ? 0 : 2;
}

int run_transformer_lm_training_json(
    const Config& cfg,
    const neuralfn::native_train::TokenShardDataset& dataset,
    const char* program) {
    constexpr std::int64_t kHeads = 12;
    constexpr std::int64_t kDefaultTargetLayers = 12;
    constexpr std::int64_t kActivationTapeCount = 1;
    constexpr std::int64_t kVocab = 50257;
    constexpr std::int64_t kPaddedVocab = 50304;
    constexpr std::int64_t kDim = 768;
    constexpr std::int64_t kHeadDim = kDim / kHeads;
    constexpr std::int64_t kHidden = kDim * 4;
    constexpr std::int64_t kQkvDim = kDim * 3;
    constexpr std::int64_t kTokenWeightElements = kPaddedVocab * kDim;
    constexpr std::int64_t kQkvWeightElements = kQkvDim * kDim;
    constexpr std::int64_t kAttnProjWeightElements = kDim * kDim;
    constexpr std::int64_t kFcWeightElements = kHidden * kDim;
    constexpr std::int64_t kMlpProjWeightElements = kDim * kHidden;
    constexpr float kInitialPositionWeight = 0.02f;
    constexpr float kInitialTokenWeightSample = -0.08f;
    constexpr float kLnWeight = 1.0f;
    constexpr float kQkvWeight = 0.015f;
    constexpr float kAttnProjWeight = 0.01f;
    constexpr float kFcWeight = 0.02f;
    constexpr float kMlpProjWeight = 0.01f;
    constexpr float kResidualScale = 1.0f;
    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.95f;
    constexpr float kEps = 1e-8f;
    constexpr float kNormEps = 1e-5f;
    constexpr float kWeightDecay = 0.1f;
    constexpr float kGradClipNorm = 1.0f;
    constexpr float kClipEps = 1e-6f;
    constexpr std::int64_t kTileSize = 1024;
    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;
    using Clock = std::chrono::steady_clock;
    const auto total_start_time = Clock::now();
    auto elapsed_ms = [](Clock::time_point start, Clock::time_point end) -> double {
        return std::chrono::duration<double, std::milli>(end - start).count();
    };
    struct SetupTimingRecord {
        std::string name;
        double total_ms = 0.0;
        std::int64_t count = 0;
    };
    std::vector<SetupTimingRecord> setup_timing_records;
    auto record_setup_timing = [&](const std::string& name, double ms) {
        for (SetupTimingRecord& record : setup_timing_records) {
            if (record.name == name) {
                record.total_ms += ms;
                record.count += 1;
                return;
            }
        }
        setup_timing_records.push_back(SetupTimingRecord{name, ms, 1});
    };
    auto run_setup_timed = [&](const std::string& name, const auto& fn) {
        const auto start = Clock::now();
        fn();
        record_setup_timing(name, elapsed_ms(start, Clock::now()));
    };
    double setup_wall_ms = 0.0;
    double train_loop_wall_ms = 0.0;
    double validation_wall_ms = 0.0;
    double checkpoint_wall_ms = 0.0;
    double total_wall_ms = 0.0;

    const std::int64_t batch_size = cfg.batch_size;
    const std::int64_t seq_len = cfg.seq_len;
    const std::int64_t target_layers = cfg.num_layers > 0 ? cfg.num_layers : kDefaultTargetLayers;
    const std::int64_t trained_layers = target_layers;
    const std::int64_t persistent_block_output_count = trained_layers > 0 ? trained_layers - 1 : 0;
    const std::int64_t backward_recompute_block_count = trained_layers > 0 ? trained_layers - 1 : 0;
    const std::int64_t rows = batch_size * seq_len;
    const std::int64_t activation_elements = rows * kDim;
    const std::int64_t hidden_elements = rows * kHidden;
    const std::int64_t qkv_activation_elements = rows * kQkvDim;
    const std::int64_t position_weight_elements = seq_len * kDim;
    auto partial_count_for = [](std::int64_t elements) {
        return (elements + kTileSize - 1) / kTileSize;
    };
    const std::int64_t requested_lm_head_chunk_rows =
        cfg.lm_head_row_chunk_size > 0 ? cfg.lm_head_row_chunk_size : 1;
    const std::int64_t lm_head_chunk_rows =
        rows < requested_lm_head_chunk_rows ? rows : requested_lm_head_chunk_rows;
    const std::int64_t lm_head_chunk_count =
        (rows + lm_head_chunk_rows - 1) / lm_head_chunk_rows;
    const std::int64_t logit_elements = lm_head_chunk_rows * kPaddedVocab;
    const std::int64_t loss_partial_count = partial_count_for(lm_head_chunk_rows);
    constexpr std::int64_t kPerBlockParameterBuffers = 12;
    constexpr std::int64_t kPerBlockGradientBuffers = 0;
    constexpr std::int64_t kPerBlockDirectAccumGradientBuffers = 12;
    constexpr std::int64_t kPerBlockAdamWStateBuffers = 24;
    constexpr std::int64_t kGlobalParameterBuffers = 4;
    const std::int64_t global_gradient_partial_count =
        partial_count_for(kTokenWeightElements) +
        partial_count_for(position_weight_elements) +
        partial_count_for(kDim) * 2;
    const std::int64_t per_block_gradient_partial_count =
        partial_count_for(kDim) * 4 +
        partial_count_for(kQkvWeightElements) +
        partial_count_for(kQkvDim) +
        partial_count_for(kAttnProjWeightElements) +
        partial_count_for(kDim) +
        partial_count_for(kFcWeightElements) +
        partial_count_for(kHidden) +
        partial_count_for(kMlpProjWeightElements) +
        partial_count_for(kDim);
    const std::int64_t gradient_partial_count =
        global_gradient_partial_count +
        per_block_gradient_partial_count * trained_layers;
    const std::int64_t microbatch_tokens = rows;
    const std::int64_t requested_train_batch_tokens =
        std::max<std::int64_t>(microbatch_tokens, static_cast<std::int64_t>(cfg.train_batch_tokens));
    const std::int64_t grad_accum_steps = microbatch_tokens > 0
        ? std::max<std::int64_t>(
              1,
              (requested_train_batch_tokens + microbatch_tokens - 1) / microbatch_tokens)
        : 1;
    const std::int64_t effective_train_batch_tokens = grad_accum_steps * microbatch_tokens;
    const std::string tile_lib_path = cfg.tile_ops_lib.empty() ? default_tile_ops_lib(program) : cfg.tile_ops_lib;
    neuralfn::native_train::BatchPlan batch_plan;
    std::string error;
    try {
        batch_plan = neuralfn::native_train::build_batch_plan(
            dataset, seq_len, batch_size, cfg.train_batch_tokens);
    } catch (const std::exception& exc) {
        error = exc.what();
    }
    if (error.empty() && (
            batch_size <= 0 || seq_len <= 0 || cfg.max_steps <= 0 ||
            target_layers <= 0 || cfg.lm_head_row_chunk_size <= 0)) {
        error = "batch size, seq_len, num_layers, max_steps, and lm_head_row_chunk_size must be positive";
    }

    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool passed = false;
    std::int64_t steps_completed = 0;
    std::int64_t epochs_completed = 0;
    std::int64_t tokens_processed = 0;
    std::int64_t train_microbatches_completed = 0;
    double final_loss_sum = 0.0;
    double final_loss_mean = 0.0;
    std::int64_t train_loss_eval_count = 0;
    std::int64_t train_loss_last_step = 0;
    bool cuda_runtime_preflight_checked = false;
    bool checkpoint_written = false;
    std::int64_t attention_forward_row_launches = 0;
    std::int64_t attention_forward_tk_launches = 0;
    std::int64_t attention_backward_tk_launches = 0;
    std::int64_t attention_tk_workspace_allocations = 0;
    std::int64_t attention_tk_workspace_element_capacity = 0;
    std::int64_t attention_tk_workspace_row_capacity = 0;
    std::int64_t attention_forward_row_fallbacks = 0;
    std::int64_t attention_forward_scalar_launches = 0;
    std::int64_t attention_forward_row_successes = 0;
    int attention_forward_row_last_error_code = 0;
    int attention_forward_row_prelaunch_clear_error_code = 0;
    int attention_forward_row_prelaunch_peek_error_code = 0;
    std::int64_t attention_forward_row_grid_x = 0;
    std::int64_t attention_forward_row_grid_y = 0;
    std::int64_t attention_forward_row_grid_z = 0;
    std::int64_t attention_forward_row_block_x = 0;
    int attention_forward_row_attr_status_code = -1;
    int attention_forward_row_attr_max_threads_per_block = 0;
    int attention_forward_row_attr_num_regs = 0;
    std::int64_t attention_forward_row_attr_shared_size_bytes = 0;
    std::int64_t attention_forward_row_attr_const_size_bytes = 0;
    std::int64_t attention_forward_row_attr_local_size_bytes = 0;
    std::int64_t linear_bf16_gemm_count = 0;
    std::int64_t linear_tk_gemm_count = 0;
    std::int64_t linear_tk_float_out_gemm_count = 0;
    std::int64_t linear_cublaslt_gemm_count = 0;
    std::int64_t linear_sgemm_count = 0;
    std::int64_t linear_bf16_a_pack_count = 0;
    std::int64_t linear_bf16_a_cache_hit_count = 0;
    std::int64_t linear_bf16_cache_reset_count = 0;
    std::int64_t linear_bf16_workspace_allocation_count = 0;
    std::int64_t linear_bf16_workspace_a_capacity = 0;
    std::int64_t linear_bf16_workspace_b_capacity = 0;
    std::int64_t linear_bf16_cached_a_capacity = 0;
    std::int64_t linear_bf16_cache_entry_count = 0;
    const bool linear_cublaslt_descriptor_cache_enabled =
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_TILE_CUDA_CUBLASLT_DESCRIPTOR_CACHE",
                              "NFN_NATIVE_LINEAR_CUBLASLT_DESCRIPTOR_CACHE"}),
            true);
    struct LinearShapeStat {
        int path = 0;
        int m = 0;
        int n = 0;
        int k = 0;
        int op_a = 0;
        int op_b = 0;
        std::int64_t calls = 0;
    };
    std::vector<LinearShapeStat> linear_shape_stats;
    std::string checkpoint_path_json;
    std::string done_marker_json;
    std::int64_t checkpoint_step = 0;
    std::int64_t checkpoint_expected_file_size = 0;
    std::int64_t checkpoint_actual_file_size = 0;
    std::int64_t checkpoint_tensor_count = 0;
    std::int64_t checkpoint_payload_elements = 0;
    std::int64_t checkpoint_device_pack_kernel_launches = 0;
    std::int64_t checkpoint_bf16_param_sync_kernel_launches = 0;
    std::int64_t checkpoint_d2h_copy_count = 0;
    std::int64_t checkpoint_d2h_bytes = 0;
    std::int64_t checkpoint_float32_d2h_bytes_elided = 0;
    bool token_weight_bf16_initial_refresh_elided = false;
    void* tile_handle = nullptr;
    void* cuda_handle = nullptr;
    std::vector<std::string> missing_symbols;
    const std::vector<std::string> required_symbols = {
        "nfn_native_tile_fill_float32",
        "nfn_native_tile_fill_many_float32",
        "nfn_native_tile_fill_many_values_float32",
        "nfn_native_tile_fill_many_values_bf16_bits_float32",
        "nfn_native_tile_init_gpt2_token_weight_float32",
        "nfn_native_tile_init_gpt2_token_weight_fast_float32",
        "nfn_native_tile_init_gpt2_token_weight_with_bf16_shadow_float32",
        "nfn_native_tile_init_gpt2_token_weight_fast_with_bf16_shadow_float32",
        "nfn_native_tile_copy_float32",
        "nfn_native_tile_uint16_to_int64",
        "nfn_native_tile_float32_to_bf16_bits",
        "nfn_native_tile_bf16_bits_to_float32",
        "nfn_native_tile_store_mlp_activations_bf16_float32",
        "nfn_native_tile_restore_mlp_activations_bf16_float32",
        "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_float32",
        "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_float32",
        "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_float32",
        "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_float32_beta",
        "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta",
        "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32_beta",
        "nfn_native_tile_linear_backward_weight_bias_accumulate_float32_bf16_bits_beta",
        "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_to_bf16_bits_float32",
        "nfn_native_tile_gelu_backward_inplace_bf16_bits_float32",
        "nfn_native_tile_float32_to_bf16_bits_many",
        "nfn_native_tile_gradient_accumulate_float32",
        "nfn_native_tile_sum_partials_float32",
        "nfn_native_tile_sumsq_partials_float32",
        "nfn_native_tile_sumsq_partials_many_float32",
        "nfn_native_tile_sumsq_partials_many_bf16_bits_float32",
        "nfn_native_tile_global_norm_clip_scale_float32",
        "nfn_native_tile_scale_inplace_by_device_float32",
        "nfn_native_tile_token_embedding_float32",
        "nfn_native_tile_token_embedding_u16_float32",
        "nfn_native_tile_token_embedding_backward_weight_float32",
        "nfn_native_tile_token_embedding_backward_weight_u16_float32",
        "nfn_native_tile_absolute_position_embedding_float32",
        "nfn_native_tile_absolute_position_embedding_backward_accumulate_float32",
        "nfn_native_tile_scaled_residual_add_float32",
        "nfn_native_tile_linear_bias_residual_add_float32",
        "nfn_native_tile_linear_bias_residual_add_bf16_linear_float32",
        "nfn_native_tile_linear_bias_residual_layer_norm_float32",
        "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_float32",
        "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_linear_float32",
        "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_residual_float32",
        "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_float32",
        "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_residual_bf16_norm_float32",
        "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_bf16_norm_float32",
        "nfn_native_tile_split_qkv_float32",
        "nfn_native_tile_split_qkv_to_heads_float32",
        "nfn_native_tile_split_qkv_to_heads_add_bias_float32",
        "nfn_native_tile_merge_qkv_float32",
        "nfn_native_tile_merge_heads_to_qkv_float32",
        "nfn_native_tile_reshape_heads_float32",
        "nfn_native_tile_merge_heads_float32",
        "nfn_native_tile_layer_norm_float32",
        "nfn_native_tile_layer_norm_with_stats_float32",
        "nfn_native_tile_layer_norm_with_stats_bf16_out_float32",
        "nfn_native_tile_layer_norm_apply_stats_bf16_out_float32",
        "nfn_native_tile_layer_norm_backward_input_float32",
        "nfn_native_tile_layer_norm_backward_input_with_stats_float32",
        "nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_float32",
        "nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_bf16_bits_float32",
        "nfn_native_tile_layer_norm_backward_affine_accumulate_float32",
        "nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_float32",
        "nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_bf16_bits_float32",
        "nfn_native_tile_layer_norm_backward_affine_residual_add_accumulate_with_stats_float32",
        "nfn_native_tile_layer_norm_backward_affine_residual_add_accumulate_with_stats_bf16_bits_float32",
        "nfn_native_tile_linear_float32",
        "nfn_native_tile_linear_bf16_float32",
        "nfn_native_tile_linear_weight_bf16_float32",
        "nfn_native_tile_linear_bf16_output_float32",
        "nfn_native_tile_linear_weight_bf16_output_float32",
        "nfn_native_tile_linear_bf16_input_weight_bf16_output_float32",
        "nfn_native_tile_linear_bf16_input_float_weight_bf16_output_float32",
        "nfn_native_tile_bf16_bits_add_bias_inplace_float32",
        "nfn_native_tile_linear_bf16_input_bits_float32",
        "nfn_native_tile_linear_bf16_input_weight_bf16_float32",
        "nfn_native_tile_linear_bf16_gelu_bf16_float32",
        "nfn_native_tile_linear_weight_bf16_gelu_bf16_float32",
        "nfn_native_tile_linear_bf16_input_weight_bf16_gelu_bf16_float32",
        "nfn_native_tile_linear_backward_input_float32",
        "nfn_native_tile_linear_backward_input_bf16_float32",
        "nfn_native_tile_linear_backward_input_weight_bf16_float32",
        "nfn_native_tile_linear_backward_input_weight_bf16_to_bf16_bits_float32",
        "nfn_native_tile_linear_backward_input_bf16_bits_float32",
        "nfn_native_tile_linear_backward_input_dgelu_bf16_bits_float32",
        "nfn_native_tile_linear_backward_input_dgelu_weight_bf16_bits_float32",
        "nfn_native_tile_linear_backward_input_dgelu_weight_bf16_bits_only_float32",
        "nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_float32",
        "nfn_native_tile_linear_backward_weight_accumulate_float32",
        "nfn_native_tile_linear_backward_weight_accumulate_bf16_float32",
        "nfn_native_tile_linear_backward_bias_accumulate_float32",
        "nfn_native_tile_linear_backward_weight_accumulate_float32_bf16_bits",
        "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32",
        "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta",
        "nfn_native_tile_linear_backward_weight_bias_accumulate_float32_bf16_bits",
        "nfn_native_tile_linear_backward_weight_bias_accumulate_float32_bf16_bits_beta",
        "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_float32_beta",
        "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32",
        "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32_beta",
        "nfn_native_tile_gelu_float32",
        "nfn_native_tile_gelu_add_bias_float32",
        "nfn_native_tile_gelu_add_bias_bf16_act_float32",
        "nfn_native_tile_gelu_backward_inplace_float32",
        "nfn_native_tile_scaled_dot_product_attention_float32",
        "nfn_native_tile_attention_forward_tk_launch_count",
        "nfn_native_tile_attention_backward_tk_launch_count",
        "nfn_native_tile_attention_tk_workspace_allocation_count",
        "nfn_native_tile_attention_tk_workspace_element_capacity",
        "nfn_native_tile_attention_tk_workspace_row_capacity",
        "nfn_native_tile_trainer_linear_stats_reset",
        "nfn_native_tile_trainer_linear_bf16_cache_reset",
        "nfn_native_tile_trainer_linear_bf16_gemm_count",
        "nfn_native_tile_trainer_linear_tk_gemm_count",
        "nfn_native_tile_trainer_linear_tk_float_out_gemm_count",
        "nfn_native_tile_trainer_linear_cublaslt_gemm_count",
        "nfn_native_tile_trainer_linear_sgemm_count",
        "nfn_native_tile_trainer_linear_bf16_a_pack_count",
        "nfn_native_tile_trainer_linear_bf16_a_cache_hit_count",
        "nfn_native_tile_trainer_linear_bf16_cache_reset_count",
        "nfn_native_tile_trainer_linear_bf16_workspace_allocation_count",
        "nfn_native_tile_trainer_linear_bf16_workspace_a_capacity",
        "nfn_native_tile_trainer_linear_bf16_workspace_b_capacity",
        "nfn_native_tile_trainer_linear_bf16_cached_a_capacity",
        "nfn_native_tile_trainer_linear_bf16_cache_entry_count",
        "nfn_native_tile_trainer_linear_shape_stats_count",
        "nfn_native_tile_trainer_linear_shape_stats_entry",
        "nfn_native_tile_scaled_dot_product_attention_backward_float32",
        "nfn_native_tile_scaled_dot_product_attention_backward_from_merged_grad_float32",
        "nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_reuse_forward_from_merged_grad_float32",
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32",
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_store_lse_bf16_float32",
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_merged_grad_float32",
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_saved_lse_bf16_from_merged_grad_float32",
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_merged_grad_float32",
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_merged_grad_float32",
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_bf16_merged_grad_float32",
        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_bf16_merged_grad_float32",
        "nfn_native_tile_scaled_dot_product_attention_store_tk_bf16_float32",
        "nfn_native_tile_attention_tk_store_forward_workspace_bf16",
        "nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_saved_tk_bf16_from_merged_grad_float32",
        "nfn_native_tile_token_cross_entropy_partials_float32",
        "nfn_native_tile_token_cross_entropy_partials_bf16_bits",
        "nfn_native_tile_token_cross_entropy_partials_strided_float32",
        "nfn_native_tile_token_cross_entropy_partials_strided_bf16_bits",
        "nfn_native_tile_token_cross_entropy_partials_strided_bf16_bits_u16_targets",
        "nfn_native_tile_token_cross_entropy_backward_inplace_with_workspace_float32",
        "nfn_native_tile_token_cross_entropy_backward_inplace_bf16_bits_with_workspace",
        "nfn_native_tile_token_cross_entropy_backward_inplace_strided_with_workspace_float32",
        "nfn_native_tile_token_cross_entropy_backward_inplace_strided_bf16_bits_with_workspace",
        "nfn_native_tile_token_cross_entropy_backward_inplace_strided_bf16_bits_u16_targets_with_workspace",
        "nfn_native_tile_adamw_step_float32",
        "nfn_native_tile_adamw_step_with_device_scale_float32",
        "nfn_native_tile_adamw_step_many_with_device_scale_float32",
        "nfn_native_tile_adamw_step_many_with_device_scale_bf16_shadow_float32",
        "nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_float32",
        "nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32",
    };

    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using FillManyValuesFn = int (*)(float* const*, const std::int64_t*, const float*, std::int64_t, std::int64_t, void*);
    using FillManyValuesBf16BitsFn =
        int (*)(std::uint16_t* const*, const std::int64_t*, const float*, std::int64_t, std::int64_t, void*);
    using Uint16ToInt64Fn = int (*)(const std::uint16_t*, std::int64_t*, std::int64_t, void*);
    using Bf16BitsToFloat32Fn = int (*)(const std::uint16_t*, float*, std::int64_t, void*);
    using Float32ToBf16BitsFn = int (*)(const float*, std::uint16_t*, std::int64_t, void*);
    using StoreMlpActivationsBf16Fn =
        int (*)(const float*, const float*, const float*, std::uint16_t*, std::int64_t, std::int64_t, void*);
    using Float32ToBf16BitsManyFn =
        int (*)(const float* const*, const std::int64_t*, const std::int64_t*, std::uint16_t*,
                std::int64_t, std::int64_t, void*);
    using GradientAccumulateFn = int (*)(float*, const float*, std::int64_t, float, void*);
    using SumPartialsFn = int (*)(const float*, float*, std::int64_t, void*);
    using SumsqPartialsManyFn =
        int (*)(const float* const*, const std::int64_t*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using SumsqPartialsManyBf16BitsFn =
        int (*)(const std::uint16_t* const*, const std::int64_t*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using ClipScaleFn = int (*)(const float*, float*, std::int64_t, float, float, void*);
    using TokenEmbeddingFn = int (*)(const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using TokenEmbeddingU16Fn =
        int (*)(const float*, const std::uint16_t*, float*, std::int64_t, std::int64_t, void*);
    using TokenEmbeddingBackwardWeightFn = int (*)(
        const std::int64_t*, const float*, float*, std::int64_t, std::int64_t, void*);
    using TokenEmbeddingBackwardWeightU16Fn = int (*)(
        const std::uint16_t*, const float*, float*, std::int64_t, std::int64_t, void*);
    using PositionEmbeddingFn = int (*)(const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using PositionEmbeddingBackwardAccumulateFn =
        int (*)(const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using ResidualAddFn = int (*)(const float*, const float*, const float*, float*, std::int64_t, void*);
    using LinearBiasResidualAddFn =
        int (*)(const float*, const float*, const float*, const float*, float*, std::int64_t, std::int64_t, void*);
    using LinearBiasResidualAddBf16LinearFn =
        int (*)(const float*, const std::uint16_t*, const float*, const float*, float*,
                std::int64_t, std::int64_t, void*);
    using LinearBiasResidualLayerNormWithStatsFn =
        int (*)(const float*, const float*, const float*, const float*, const float*, const float*,
                float*, float*, float*, float*, std::int64_t, std::int64_t, float, void*);
    using LinearBiasResidualLayerNormWithStatsBf16LinearFn =
        int (*)(const float*, const std::uint16_t*, const float*, const float*, const float*, const float*,
                float*, float*, float*, float*, std::int64_t, std::int64_t, float, void*);
    using LinearBiasResidualLayerNormWithStatsBf16ResidualFn =
        int (*)(const float*, const float*, const float*, const float*, const float*, const float*,
                float*, float*, float*, float*, std::uint16_t*, std::int64_t, std::int64_t, float, void*);
    using LinearBiasResidualLayerNormWithStatsBf16LinearBf16ResidualFn =
        int (*)(const float*, const std::uint16_t*, const float*, const float*, const float*, const float*,
                float*, float*, float*, float*, std::uint16_t*, std::int64_t, std::int64_t, float, void*);
    using LinearBiasResidualLayerNormWithStatsBf16ResidualBf16NormFn =
        int (*)(const float*, const float*, const float*, const float*, const float*, const float*,
                float*, float*, float*, float*, std::uint16_t*, std::uint16_t*,
                std::int64_t, std::int64_t, float, void*);
    using LinearBiasResidualLayerNormWithStatsBf16LinearBf16ResidualBf16NormFn =
        int (*)(const float*, const std::uint16_t*, const float*, const float*, const float*, const float*,
                float*, float*, float*, float*, std::uint16_t*, std::uint16_t*,
                std::int64_t, std::int64_t, float, void*);
    using SplitQkvToHeadsAddBiasFn = int (*)(
        const float*, const float*, float*, float*, float*,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using MergeHeadsFn = int (*)(
        const float*, float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using LayerNormFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, float, void*);
    using LayerNormWithStatsFn = int (*)(
        const float*, const float*, const float*, float*, float*, float*, std::int64_t, std::int64_t, float, void*);
    using LayerNormWithStatsBf16OutFn = int (*)(
        const float*, const float*, const float*, float*, float*, float*, std::uint16_t*,
        std::int64_t, std::int64_t, float, void*);
    using LayerNormApplyStatsBf16OutFn = int (*)(
        const float*, const float*, const float*, const float*, const float*, std::uint16_t*,
        std::int64_t, std::int64_t, void*);
    using LayerNormBackwardInputFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, float, void*);
    using LayerNormBackwardInputWithStatsFn = int (*)(
        const float*, const float*, const float*, const float*, const float*, float*, std::int64_t, std::int64_t, void*);
    using LayerNormBackwardInputResidualAddWithStatsFn = int (*)(
        const float*, const float*, const float*, const float*, const float*,
        const float*, const float*, float*, std::int64_t, std::int64_t, void*);
    using LayerNormBackwardInputResidualAddWithStatsBf16BitsFn = int (*)(
        const std::uint16_t*, const float*, const float*, const float*, const float*,
        const float*, const float*, float*, std::int64_t, std::int64_t, void*);
    using LayerNormBackwardAffineAccumulateFn = int (*)(
        const float*, const float*, float*, float*, std::int64_t, std::int64_t, float, void*);
    using LayerNormBackwardAffineAccumulateWithStatsFn = int (*)(
        const float*, const float*, const float*, const float*, float*, float*, std::int64_t, std::int64_t, void*);
    using LayerNormBackwardAffineAccumulateWithStatsBf16BitsFn = int (*)(
        const std::uint16_t*, const float*, const float*, const float*, float*, float*, std::int64_t, std::int64_t, void*);
    using LayerNormBackwardAffineResidualAddAccumulateWithStatsFn = int (*)(
        const float*, const float*, const float*, const float*, const float*,
        const float*, const float*, float*, float*, float*, std::int64_t, std::int64_t, void*);
    using LayerNormBackwardAffineResidualAddAccumulateWithStatsBf16BitsFn = int (*)(
        const std::uint16_t*, const float*, const float*, const float*, const float*,
        const float*, const float*, float*, float*, float*, std::int64_t, std::int64_t, void*);
    using LinearFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBf16OutputFn = int (*)(
        const float*, const float*, const float*, std::uint16_t*,
        std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearWeightBf16OutputFn = int (*)(
        const float*, const std::uint16_t*, const float*, std::uint16_t*,
        std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBf16InputWeightBf16OutputFn = int (*)(
        const std::uint16_t*, const std::uint16_t*, const float*, std::uint16_t*,
        std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearBf16InputFloatWeightBf16OutputFn = int (*)(
        const std::uint16_t*, const float*, const float*, std::uint16_t*,
        std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearWeightBf16Fn = int (*)(
        const float*, const std::uint16_t*, const float*, float*,
        std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using Bf16BitsAddBiasInplaceFn =
        int (*)(std::uint16_t*, const float*, std::int64_t, std::int64_t, void*);
    using LinearBf16InputWeightBf16Fn = int (*)(
        const std::uint16_t*, const std::uint16_t*, const float*, float*,
        std::int64_t, std::int64_t, std::int64_t, bool, void*);
    using LinearWeightBf16GeluBf16Fn = int (*)(
        const float*, const std::uint16_t*, const float*, std::uint16_t*, std::uint16_t*,
        std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBf16InputWeightBf16GeluBf16Fn = int (*)(
        const std::uint16_t*, const std::uint16_t*, const float*, std::uint16_t*, std::uint16_t*,
        std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardInputFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardInputBf16BitsFn = int (*)(
        const std::uint16_t*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardInputWeightBf16Fn = int (*)(
        const float*, const std::uint16_t*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardInputWeightBf16ToBf16BitsFn = int (*)(
        const float*, const std::uint16_t*, std::uint16_t*,
        std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardInputBf16BitsWeightBf16Fn = int (*)(
        const std::uint16_t*, const std::uint16_t*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardInputDgeluWeightBf16BitsFn = int (*)(
        const float*, const std::uint16_t*, const std::uint16_t*, std::uint16_t*, float*,
        std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardInputDgeluWeightBf16BitsOnlyFn = int (*)(
        const float*, const std::uint16_t*, const std::uint16_t*, std::uint16_t*, float*,
        std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightAccumulateFn = int (*)(
        const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightBiasAccumulateFn = int (*)(
        const float*, const float*, float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightBiasAccumulateBf16BitsFn = int (*)(
        const std::uint16_t*, const float*, float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightBiasAccumulateBf16BitsBetaFn = int (*)(
        const std::uint16_t*, const float*, float*, float*,
        std::int64_t, std::int64_t, std::int64_t, float, void*);
    using LinearBackwardWeightBiasAccumulateBf16BitsBf16BitsFn = int (*)(
        const std::uint16_t*, const std::uint16_t*, float*, float*,
        std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightBiasAccumulateBf16BitsBf16BitsBetaFn = int (*)(
        const std::uint16_t*, const std::uint16_t*, float*, float*,
        std::int64_t, std::int64_t, std::int64_t, float, void*);
    using LinearBackwardWeightBiasAccumulateBf16BitsBf16BitsToBf16BitsFn = int (*)(
        const std::uint16_t*, const std::uint16_t*, std::uint16_t*, float*,
        std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightAccumulateFloat32Bf16BitsFn = int (*)(
        const float*, const std::uint16_t*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightAccumulateBf16BitsBf16BitsFn = int (*)(
        const std::uint16_t*, const std::uint16_t*, float*,
        std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightAccumulateBf16BitsBf16BitsBetaFn = int (*)(
        const std::uint16_t*, const std::uint16_t*, float*,
        std::int64_t, std::int64_t, std::int64_t, float, void*);
    using LinearBackwardWeightBiasAccumulateFloat32Bf16BitsFn = int (*)(
        const float*, const std::uint16_t*, float*, float*,
        std::int64_t, std::int64_t, std::int64_t, void*);
    using LinearBackwardWeightBiasAccumulateFloat32Bf16BitsBetaFn = int (*)(
        const float*, const std::uint16_t*, float*, float*,
        std::int64_t, std::int64_t, std::int64_t, float, void*);
    using GeluAddBiasBf16ActFn =
        int (*)(const float*, const float*, float*, float*, std::uint16_t*, std::int64_t, std::int64_t, void*);
    using GeluBackwardInplaceFn = int (*)(const float*, float*, std::int64_t, void*);
    using GeluBackwardInplaceBf16BitsFn = int (*)(const std::uint16_t*, float*, std::int64_t, void*);
    using AttentionFn = int (*)(
        const float*, const float*, const float*, float*,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        std::int64_t, std::int64_t, std::int64_t, float, bool, bool, bool,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using AttentionStatsResetFn = void (*)();
    using AttentionStatsCountFn = std::int64_t (*)();
    using AttentionStatsErrorFn = int (*)();
    using TrainerLinearStatsResetFn = void (*)();
    using TrainerLinearStatsCountFn = std::int64_t (*)();
    using TrainerLinearShapeStatsEntryFn = bool (*)(
        std::int64_t, int*, int*, int*, int*, int*, int*, std::int64_t*);
    using AttentionBackwardToQkvReuseForwardFn = int (*)(
        const float*, float*,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        std::int64_t, std::int64_t, std::int64_t, float, bool, bool, bool,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using PackedAttentionForwardFn = int (*)(
        const std::uint16_t*, std::uint16_t*,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        std::int64_t, std::int64_t, std::int64_t, float, bool, bool, bool,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using PackedAttentionForwardStoreLseFn = int (*)(
        const std::uint16_t*, std::uint16_t*, float*,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        std::int64_t, std::int64_t, std::int64_t, float, bool, bool, bool,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using PackedAttentionBackwardToQkvFn = int (*)(
        const std::uint16_t*, const std::uint16_t*, const float*, float*,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        std::int64_t, std::int64_t, std::int64_t, float, bool, bool, bool,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using PackedAttentionBackwardToQkvSavedLseFn = int (*)(
        const std::uint16_t*, const std::uint16_t*, const float*, const float*, float*,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        std::int64_t, std::int64_t, std::int64_t, float, bool, bool, bool,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using PackedAttentionBackwardToQkvBf16BitsFn = int (*)(
        const std::uint16_t*, const std::uint16_t*, const float*, std::uint16_t*,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        std::int64_t, std::int64_t, std::int64_t, float, bool, bool, bool,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using PackedAttentionBackwardToQkvBf16BitsSavedLseFn = int (*)(
        const std::uint16_t*, const std::uint16_t*, const float*, const float*, std::uint16_t*,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        std::int64_t, std::int64_t, std::int64_t, float, bool, bool, bool,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using PackedAttentionBackwardToQkvBf16BitsFromBf16GradFn = int (*)(
        const std::uint16_t*, const std::uint16_t*, const std::uint16_t*, std::uint16_t*,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        std::int64_t, std::int64_t, std::int64_t, float, bool, bool, bool,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using PackedAttentionBackwardToQkvBf16BitsSavedLseFromBf16GradFn = int (*)(
        const std::uint16_t*, const std::uint16_t*, const float*, const std::uint16_t*, std::uint16_t*,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        std::int64_t, std::int64_t, std::int64_t, float, bool, bool, bool,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using AttentionStoreForwardTkFn = int (*)(
        const float*, const float*, const float*, float*,
        std::uint16_t*, std::uint16_t*, std::uint16_t*, std::uint16_t*, float*,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        std::int64_t, std::int64_t, std::int64_t, float, bool, bool, bool,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using StoreAttentionTkWorkspaceFn = int (*)(
        std::uint16_t*, std::uint16_t*, std::uint16_t*, std::uint16_t*, float*,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using AttentionBackwardToQkvFromSavedTkFn = int (*)(
        const std::uint16_t*, const std::uint16_t*, const std::uint16_t*, const std::uint16_t*,
        const float*, const float*, float*,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        std::int64_t, std::int64_t, std::int64_t, float, bool, bool, bool,
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyPartialsFn = int (*)(
        const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyPartialsBf16BitsFn = int (*)(
        const std::uint16_t*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyPartialsStridedFn = int (*)(
        const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyPartialsStridedBf16BitsFn = int (*)(
        const std::uint16_t*, const std::int64_t*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyPartialsStridedBf16BitsU16TargetsFn = int (*)(
        const std::uint16_t*, const std::uint16_t*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using TokenCrossEntropyBackwardInplaceWorkspaceFn = int (*)(
        float*, const std::int64_t*, float*, float*,
        std::int64_t, std::int64_t, float, void*);
    using TokenCrossEntropyBackwardInplaceBf16BitsWorkspaceFn = int (*)(
        std::uint16_t*, const std::int64_t*, float*, float*,
        std::int64_t, std::int64_t, float, void*);
    using TokenCrossEntropyBackwardInplaceStridedWorkspaceFn = int (*)(
        float*, const std::int64_t*, float*, float*,
        std::int64_t, std::int64_t, std::int64_t, float, void*);
    using TokenCrossEntropyBackwardInplaceStridedBf16BitsWorkspaceFn = int (*)(
        std::uint16_t*, const std::int64_t*, float*, float*,
        std::int64_t, std::int64_t, std::int64_t, float, void*);
    using TokenCrossEntropyBackwardInplaceStridedBf16BitsU16TargetsWorkspaceFn = int (*)(
        std::uint16_t*, const std::uint16_t*, float*, float*,
        std::int64_t, std::int64_t, std::int64_t, float, void*);
    using AdamWManyWithDeviceScaleFn = int (*)(
        float* const*, const float* const*, const float*, float* const*, float* const*,
        const std::int64_t*, const float*, std::int64_t, std::int64_t,
        float, float, float, float, float, float, void*);
    using AdamWManyWithDeviceScaleBf16ShadowFn = int (*)(
        float* const*, const float* const*, const float*, float* const*, float* const*,
        const std::int64_t*, const float*, const std::int64_t*, std::uint16_t*,
        std::int64_t, std::int64_t, float, float, float, float, float, float, void*);
    using AdamWManyWithDeviceScaleBf16ParamFn = int (*)(
        std::uint16_t* const*, const float* const*, const float*, float* const*, float* const*,
        const std::int64_t*, const float*, std::int64_t, std::int64_t,
        float, float, float, float, float, float, void*);
    using AdamWManyWithDeviceScaleBf16ParamBf16GradFn = int (*)(
        std::uint16_t* const*, const std::uint16_t* const*, const float*, float* const*, float* const*,
        const std::int64_t*, const float*, std::int64_t, std::int64_t,
        float, float, float, float, float, float, void*);
    using FillManyFn = int (*)(float* const*, const std::int64_t*, std::int64_t, std::int64_t, float, void*);
    using InitGpt2TokenWeightFn = int (*)(float*, std::int64_t, void*);
    using InitGpt2TokenWeightWithBf16ShadowFn = int (*)(
        float*, std::uint16_t*, std::int64_t, void*);
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaMemcpyAsyncFn = int (*)(void*, const void*, std::size_t, int, void*);
    using CudaMemsetAsyncFn = int (*)(void*, int, std::size_t, void*);
    using CudaMallocAsyncFn = int (*)(void**, std::size_t, void*);
    using CudaFreeAsyncFn = int (*)(void*, void*);
    using CudaHostAllocFn = int (*)(void**, std::size_t, unsigned int);
    using CudaFreeHostFn = int (*)(void*);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);
    using CudaVersionFn = int (*)(int*);
    using CudaEventCreateWithFlagsFn = int (*)(void**, unsigned int);
    using CudaEventRecordFn = int (*)(void*, void*);
    using CudaEventElapsedTimeFn = int (*)(float*, void*, void*);
    using CudaEventDestroyFn = int (*)(void*);

    FillFn fill = nullptr;
    FillManyValuesFn fill_many_values = nullptr;
    FillManyValuesBf16BitsFn fill_many_values_bf16_bits = nullptr;
    InitGpt2TokenWeightFn init_gpt2_token_weight = nullptr;
    InitGpt2TokenWeightFn init_gpt2_token_weight_fast = nullptr;
    InitGpt2TokenWeightWithBf16ShadowFn init_gpt2_token_weight_with_bf16_shadow = nullptr;
    InitGpt2TokenWeightWithBf16ShadowFn init_gpt2_token_weight_fast_with_bf16_shadow = nullptr;
    Uint16ToInt64Fn uint16_to_int64 = nullptr;
    Bf16BitsToFloat32Fn bf16_bits_to_float32 = nullptr;
    Float32ToBf16BitsFn float32_to_bf16_bits = nullptr;
    StoreMlpActivationsBf16Fn store_mlp_activations_bf16 = nullptr;
    Float32ToBf16BitsManyFn float32_to_bf16_bits_many = nullptr;
    GradientAccumulateFn gradient_accumulate = nullptr;
    SumPartialsFn sum_partials = nullptr;
    SumsqPartialsManyFn sumsq_partials_many = nullptr;
    SumsqPartialsManyBf16BitsFn sumsq_partials_many_bf16_bits = nullptr;
    ClipScaleFn clip_scale = nullptr;
    TokenEmbeddingFn token_embedding = nullptr;
    TokenEmbeddingU16Fn token_embedding_u16 = nullptr;
    TokenEmbeddingBackwardWeightFn token_embedding_backward_weight = nullptr;
    TokenEmbeddingBackwardWeightU16Fn token_embedding_backward_weight_u16 = nullptr;
    PositionEmbeddingFn position_embedding = nullptr;
    PositionEmbeddingBackwardAccumulateFn position_embedding_backward_accumulate = nullptr;
    ResidualAddFn residual_add = nullptr;
    LinearBiasResidualAddFn linear_bias_residual_add = nullptr;
    LinearBiasResidualAddBf16LinearFn linear_bias_residual_add_bf16_linear = nullptr;
    LinearBiasResidualLayerNormWithStatsFn linear_bias_residual_layer_norm_with_stats = nullptr;
    LinearBiasResidualLayerNormWithStatsBf16LinearFn
        linear_bias_residual_layer_norm_with_stats_bf16_linear = nullptr;
    LinearBiasResidualLayerNormWithStatsBf16ResidualFn
        linear_bias_residual_layer_norm_with_stats_bf16_residual = nullptr;
    LinearBiasResidualLayerNormWithStatsBf16LinearBf16ResidualFn
        linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual = nullptr;
    LinearBiasResidualLayerNormWithStatsBf16ResidualBf16NormFn
        linear_bias_residual_layer_norm_with_stats_bf16_residual_bf16_norm = nullptr;
    LinearBiasResidualLayerNormWithStatsBf16LinearBf16ResidualBf16NormFn
        linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_bf16_norm = nullptr;
    SplitQkvToHeadsAddBiasFn split_qkv_to_heads_add_bias = nullptr;
    MergeHeadsFn merge_heads = nullptr;
    LayerNormFn layer_norm = nullptr;
    LayerNormWithStatsFn layer_norm_with_stats = nullptr;
    LayerNormWithStatsBf16OutFn layer_norm_with_stats_bf16_out = nullptr;
    LayerNormApplyStatsBf16OutFn layer_norm_apply_stats_bf16_out = nullptr;
    LayerNormBackwardInputFn layer_norm_backward_input = nullptr;
    LayerNormBackwardInputWithStatsFn layer_norm_backward_input_with_stats = nullptr;
    LayerNormBackwardInputResidualAddWithStatsFn layer_norm_backward_input_residual_add_with_stats = nullptr;
    LayerNormBackwardInputResidualAddWithStatsBf16BitsFn
        layer_norm_backward_input_residual_add_with_stats_bf16_bits = nullptr;
    LayerNormBackwardAffineAccumulateFn layer_norm_backward_affine_accumulate = nullptr;
    LayerNormBackwardAffineAccumulateWithStatsFn layer_norm_backward_affine_accumulate_with_stats = nullptr;
    LayerNormBackwardAffineAccumulateWithStatsBf16BitsFn
        layer_norm_backward_affine_accumulate_with_stats_bf16_bits = nullptr;
    LayerNormBackwardAffineResidualAddAccumulateWithStatsFn
        layer_norm_backward_affine_residual_add_accumulate_with_stats = nullptr;
    LayerNormBackwardAffineResidualAddAccumulateWithStatsBf16BitsFn
        layer_norm_backward_affine_residual_add_accumulate_with_stats_bf16_bits = nullptr;
    LinearFn linear = nullptr;
    LinearBf16OutputFn linear_bf16_output = nullptr;
    LinearWeightBf16Fn linear_weight_bf16 = nullptr;
    LinearWeightBf16OutputFn linear_weight_bf16_output = nullptr;
    LinearBf16InputWeightBf16OutputFn linear_bf16_input_weight_bf16_output = nullptr;
    LinearBf16InputFloatWeightBf16OutputFn linear_bf16_input_float_weight_bf16_output = nullptr;
    Bf16BitsAddBiasInplaceFn bf16_bits_add_bias_inplace = nullptr;
    LinearBf16InputWeightBf16Fn linear_bf16_input_weight_bf16 = nullptr;
    LinearWeightBf16GeluBf16Fn linear_weight_bf16_gelu_bf16 = nullptr;
    LinearBf16InputWeightBf16GeluBf16Fn linear_bf16_input_weight_bf16_gelu_bf16 = nullptr;
    LinearBackwardInputFn linear_backward_input = nullptr;
    LinearBackwardInputBf16BitsFn linear_backward_input_bf16_bits = nullptr;
    LinearBackwardInputWeightBf16Fn linear_backward_input_weight_bf16 = nullptr;
    LinearBackwardInputWeightBf16ToBf16BitsFn linear_backward_input_weight_bf16_to_bf16_bits = nullptr;
    LinearBackwardInputBf16BitsWeightBf16Fn linear_backward_input_bf16_bits_weight_bf16 = nullptr;
    LinearBackwardInputDgeluWeightBf16BitsFn linear_backward_input_dgelu_weight_bf16_bits = nullptr;
    LinearBackwardInputDgeluWeightBf16BitsOnlyFn
        linear_backward_input_dgelu_weight_bf16_bits_only = nullptr;
    LinearBackwardWeightAccumulateFn linear_backward_weight_accumulate = nullptr;
    LinearBackwardWeightBiasAccumulateFn linear_backward_weight_bias_accumulate_bf16 = nullptr;
    LinearBackwardWeightBiasAccumulateBf16BitsFn linear_backward_weight_bias_accumulate_bf16_bits = nullptr;
    LinearBackwardWeightBiasAccumulateBf16BitsBetaFn linear_backward_weight_bias_accumulate_bf16_bits_beta = nullptr;
    LinearBackwardWeightBiasAccumulateBf16BitsBf16BitsFn
        linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits = nullptr;
    LinearBackwardWeightBiasAccumulateBf16BitsBf16BitsBetaFn
        linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_beta = nullptr;
    LinearBackwardWeightBiasAccumulateBf16BitsBf16BitsToBf16BitsFn
        linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_to_bf16_bits = nullptr;
    LinearBackwardWeightAccumulateFloat32Bf16BitsFn linear_backward_weight_accumulate_float32_bf16_bits = nullptr;
    LinearBackwardWeightAccumulateBf16BitsBf16BitsFn
        linear_backward_weight_accumulate_bf16_bits_bf16_bits = nullptr;
    LinearBackwardWeightAccumulateBf16BitsBf16BitsBetaFn
        linear_backward_weight_accumulate_bf16_bits_bf16_bits_beta = nullptr;
    LinearBackwardWeightBiasAccumulateFloat32Bf16BitsFn
        linear_backward_weight_bias_accumulate_float32_bf16_bits = nullptr;
    LinearBackwardWeightBiasAccumulateFloat32Bf16BitsBetaFn
        linear_backward_weight_bias_accumulate_float32_bf16_bits_beta = nullptr;
    GeluAddBiasBf16ActFn gelu_add_bias_bf16_act = nullptr;
    GeluBackwardInplaceFn gelu_backward_inplace = nullptr;
    GeluBackwardInplaceBf16BitsFn gelu_backward_inplace_bf16_bits = nullptr;
    AttentionFn attention = nullptr;
    AttentionStatsResetFn attention_stats_reset = nullptr;
    AttentionStatsCountFn attention_row_launch_count = nullptr;
    AttentionStatsCountFn attention_forward_tk_launch_count = nullptr;
    AttentionStatsCountFn attention_backward_tk_launch_count = nullptr;
    AttentionStatsCountFn attention_tk_workspace_allocation_count_fn = nullptr;
    AttentionStatsCountFn attention_tk_workspace_element_capacity_fn = nullptr;
    AttentionStatsCountFn attention_tk_workspace_row_capacity_fn = nullptr;
    AttentionStatsCountFn attention_row_fallback_count = nullptr;
    AttentionStatsCountFn attention_scalar_launch_count = nullptr;
    AttentionStatsErrorFn attention_row_last_error = nullptr;
    AttentionStatsErrorFn attention_row_prelaunch_clear_error = nullptr;
    AttentionStatsErrorFn attention_row_prelaunch_peek_error = nullptr;
    AttentionStatsCountFn attention_row_grid_x = nullptr;
    AttentionStatsCountFn attention_row_grid_y = nullptr;
    AttentionStatsCountFn attention_row_grid_z = nullptr;
    AttentionStatsCountFn attention_row_block_x = nullptr;
    AttentionStatsErrorFn attention_row_attr_status = nullptr;
    AttentionStatsErrorFn attention_row_attr_max_threads_per_block = nullptr;
    AttentionStatsErrorFn attention_row_attr_num_regs = nullptr;
    AttentionStatsCountFn attention_row_attr_shared_size_bytes = nullptr;
    AttentionStatsCountFn attention_row_attr_const_size_bytes = nullptr;
    AttentionStatsCountFn attention_row_attr_local_size_bytes = nullptr;
    TrainerLinearStatsResetFn trainer_linear_stats_reset = nullptr;
    TrainerLinearStatsResetFn trainer_linear_bf16_cache_reset = nullptr;
    TrainerLinearStatsCountFn trainer_linear_bf16_gemm_count_fn = nullptr;
    TrainerLinearStatsCountFn trainer_linear_tk_gemm_count_fn = nullptr;
    TrainerLinearStatsCountFn trainer_linear_tk_float_out_gemm_count_fn = nullptr;
    TrainerLinearStatsCountFn trainer_linear_cublaslt_gemm_count_fn = nullptr;
    TrainerLinearStatsCountFn trainer_linear_sgemm_count_fn = nullptr;
    TrainerLinearStatsCountFn trainer_linear_bf16_a_pack_count_fn = nullptr;
    TrainerLinearStatsCountFn trainer_linear_bf16_a_cache_hit_count_fn = nullptr;
    TrainerLinearStatsCountFn trainer_linear_bf16_cache_reset_count_fn = nullptr;
    TrainerLinearStatsCountFn trainer_linear_bf16_workspace_allocation_count_fn = nullptr;
    TrainerLinearStatsCountFn trainer_linear_bf16_workspace_a_capacity_fn = nullptr;
    TrainerLinearStatsCountFn trainer_linear_bf16_workspace_b_capacity_fn = nullptr;
    TrainerLinearStatsCountFn trainer_linear_bf16_cached_a_capacity_fn = nullptr;
    TrainerLinearStatsCountFn trainer_linear_bf16_cache_entry_count_fn = nullptr;
    TrainerLinearStatsCountFn trainer_linear_shape_stats_count_fn = nullptr;
    TrainerLinearShapeStatsEntryFn trainer_linear_shape_stats_entry_fn = nullptr;
    AttentionBackwardToQkvReuseForwardFn attention_backward_to_qkv_reuse_forward = nullptr;
    PackedAttentionForwardFn packed_attention_forward = nullptr;
    PackedAttentionForwardStoreLseFn packed_attention_forward_store_lse = nullptr;
    PackedAttentionBackwardToQkvFn packed_attention_backward_to_qkv = nullptr;
    PackedAttentionBackwardToQkvSavedLseFn packed_attention_backward_to_qkv_saved_lse = nullptr;
    PackedAttentionBackwardToQkvBf16BitsFn packed_attention_backward_to_qkv_bf16_bits = nullptr;
    PackedAttentionBackwardToQkvBf16BitsSavedLseFn packed_attention_backward_to_qkv_bf16_bits_saved_lse = nullptr;
    PackedAttentionBackwardToQkvBf16BitsFromBf16GradFn
        packed_attention_backward_to_qkv_bf16_bits_from_bf16_grad = nullptr;
    PackedAttentionBackwardToQkvBf16BitsSavedLseFromBf16GradFn
        packed_attention_backward_to_qkv_bf16_bits_saved_lse_from_bf16_grad = nullptr;
    AttentionStoreForwardTkFn attention_store_forward_tk = nullptr;
    StoreAttentionTkWorkspaceFn store_attention_tk_workspace = nullptr;
    AttentionBackwardToQkvFromSavedTkFn attention_backward_to_qkv_from_saved_tk = nullptr;
    TokenCrossEntropyPartialsFn ce_partials = nullptr;
    TokenCrossEntropyPartialsBf16BitsFn ce_partials_bf16_bits = nullptr;
    TokenCrossEntropyPartialsStridedFn ce_partials_strided = nullptr;
    TokenCrossEntropyPartialsStridedBf16BitsFn ce_partials_strided_bf16_bits = nullptr;
    TokenCrossEntropyPartialsStridedBf16BitsU16TargetsFn
        ce_partials_strided_bf16_bits_u16_targets = nullptr;
    TokenCrossEntropyBackwardInplaceWorkspaceFn ce_backward_inplace_workspace = nullptr;
    TokenCrossEntropyBackwardInplaceBf16BitsWorkspaceFn ce_backward_inplace_bf16_bits_workspace = nullptr;
    TokenCrossEntropyBackwardInplaceStridedWorkspaceFn ce_backward_inplace_strided_workspace = nullptr;
    TokenCrossEntropyBackwardInplaceStridedBf16BitsWorkspaceFn ce_backward_inplace_strided_bf16_bits_workspace = nullptr;
    TokenCrossEntropyBackwardInplaceStridedBf16BitsU16TargetsWorkspaceFn
        ce_backward_inplace_strided_bf16_bits_u16_targets_workspace = nullptr;
    FillManyFn fill_many = nullptr;
    AdamWManyWithDeviceScaleFn adamw_many_with_device_scale = nullptr;
    AdamWManyWithDeviceScaleBf16ShadowFn adamw_many_with_device_scale_bf16_shadow = nullptr;
    AdamWManyWithDeviceScaleBf16ParamFn adamw_many_with_device_scale_bf16_param = nullptr;
    AdamWManyWithDeviceScaleBf16ParamBf16GradFn adamw_many_with_device_scale_bf16_param_bf16_grad = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaMemcpyAsyncFn cuda_memcpy_async = nullptr;
    CudaMemsetAsyncFn cuda_memset_async = nullptr;
    CudaMallocAsyncFn cuda_malloc_async = nullptr;
    CudaFreeAsyncFn cuda_free_async = nullptr;
    CudaHostAllocFn cuda_host_alloc = nullptr;
    CudaFreeHostFn cuda_free_host = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;
    CudaVersionFn cuda_runtime_get_version = nullptr;
    CudaVersionFn cuda_driver_get_version = nullptr;
    CudaEventCreateWithFlagsFn cuda_event_create_with_flags = nullptr;
    CudaEventRecordFn cuda_event_record = nullptr;
    CudaEventElapsedTimeFn cuda_event_elapsed_time = nullptr;
    CudaEventDestroyFn cuda_event_destroy = nullptr;
    int cuda_runtime_version = 0;
    int cuda_driver_version = 0;
    int cuda_runtime_version_status = -1;
    int cuda_driver_version_status = -1;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed with CUDA error " << code;
        if (cuda_get_error_string != nullptr) {
            const char* message = cuda_get_error_string(code);
            if (message != nullptr) {
                out << ": " << message;
            }
        }
        return out.str();
    };

    if (error.empty()) {
        tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (tile_handle == nullptr) {
            error = dl_last_error("dlopen tile ops failed");
        } else {
            tile_loaded = true;
            for (const std::string& symbol : required_symbols) {
                void* ptr = dlsym(tile_handle, symbol.c_str());
                if (ptr == nullptr) {
                    missing_symbols.push_back(symbol);
                }
            }
            if (!missing_symbols.empty()) {
                error = "missing required GPT-2 transformer/LM training Tile ABI symbols";
            } else {
                fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
                fill_many = load_symbol<FillManyFn>(tile_handle, "nfn_native_tile_fill_many_float32");
                adamw_many_with_device_scale_bf16_param_bf16_grad =
                    load_symbol<AdamWManyWithDeviceScaleBf16ParamBf16GradFn>(
                        tile_handle,
                        "nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32");
                fill_many_values = load_symbol<FillManyValuesFn>(
                    tile_handle, "nfn_native_tile_fill_many_values_float32");
                fill_many_values_bf16_bits = load_symbol<FillManyValuesBf16BitsFn>(
                    tile_handle, "nfn_native_tile_fill_many_values_bf16_bits_float32");
                init_gpt2_token_weight =
                    load_symbol<InitGpt2TokenWeightFn>(tile_handle, "nfn_native_tile_init_gpt2_token_weight_float32");
                init_gpt2_token_weight_fast =
                    load_symbol<InitGpt2TokenWeightFn>(
                        tile_handle,
                        "nfn_native_tile_init_gpt2_token_weight_fast_float32");
                init_gpt2_token_weight_with_bf16_shadow =
                    load_symbol<InitGpt2TokenWeightWithBf16ShadowFn>(
                        tile_handle,
                        "nfn_native_tile_init_gpt2_token_weight_with_bf16_shadow_float32");
                init_gpt2_token_weight_fast_with_bf16_shadow =
                    load_symbol<InitGpt2TokenWeightWithBf16ShadowFn>(
                        tile_handle,
                        "nfn_native_tile_init_gpt2_token_weight_fast_with_bf16_shadow_float32");
                uint16_to_int64 = load_symbol<Uint16ToInt64Fn>(tile_handle, "nfn_native_tile_uint16_to_int64");
                bf16_bits_to_float32 =
                    load_symbol<Bf16BitsToFloat32Fn>(tile_handle, "nfn_native_tile_bf16_bits_to_float32");
                float32_to_bf16_bits =
                    load_symbol<Float32ToBf16BitsFn>(tile_handle, "nfn_native_tile_float32_to_bf16_bits");
                store_mlp_activations_bf16 = load_symbol<StoreMlpActivationsBf16Fn>(
                    tile_handle, "nfn_native_tile_store_mlp_activations_bf16_float32");
                float32_to_bf16_bits_many = load_symbol<Float32ToBf16BitsManyFn>(
                    tile_handle, "nfn_native_tile_float32_to_bf16_bits_many");
                gradient_accumulate = load_symbol<GradientAccumulateFn>(
                    tile_handle, "nfn_native_tile_gradient_accumulate_float32");
                sum_partials = load_symbol<SumPartialsFn>(
                    tile_handle, "nfn_native_tile_sum_partials_float32");
                sumsq_partials_many = load_symbol<SumsqPartialsManyFn>(
                    tile_handle, "nfn_native_tile_sumsq_partials_many_float32");
                sumsq_partials_many_bf16_bits = load_symbol<SumsqPartialsManyBf16BitsFn>(
                    tile_handle, "nfn_native_tile_sumsq_partials_many_bf16_bits_float32");
                clip_scale = load_symbol<ClipScaleFn>(
                    tile_handle, "nfn_native_tile_global_norm_clip_scale_float32");
                token_embedding = load_symbol<TokenEmbeddingFn>(tile_handle, "nfn_native_tile_token_embedding_float32");
                token_embedding_u16 =
                    load_symbol<TokenEmbeddingU16Fn>(tile_handle, "nfn_native_tile_token_embedding_u16_float32");
                token_embedding_backward_weight = load_symbol<TokenEmbeddingBackwardWeightFn>(
                    tile_handle, "nfn_native_tile_token_embedding_backward_weight_float32");
                token_embedding_backward_weight_u16 = load_symbol<TokenEmbeddingBackwardWeightU16Fn>(
                    tile_handle, "nfn_native_tile_token_embedding_backward_weight_u16_float32");
                position_embedding = load_symbol<PositionEmbeddingFn>(
                    tile_handle, "nfn_native_tile_absolute_position_embedding_float32");
                position_embedding_backward_accumulate = load_symbol<PositionEmbeddingBackwardAccumulateFn>(
                    tile_handle, "nfn_native_tile_absolute_position_embedding_backward_accumulate_float32");
                residual_add = load_symbol<ResidualAddFn>(tile_handle, "nfn_native_tile_scaled_residual_add_float32");
                linear_bias_residual_add = load_symbol<LinearBiasResidualAddFn>(
                    tile_handle, "nfn_native_tile_linear_bias_residual_add_float32");
                linear_bias_residual_add_bf16_linear = load_symbol<LinearBiasResidualAddBf16LinearFn>(
                    tile_handle, "nfn_native_tile_linear_bias_residual_add_bf16_linear_float32");
                linear_bias_residual_layer_norm_with_stats = load_symbol<LinearBiasResidualLayerNormWithStatsFn>(
                    tile_handle, "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_float32");
                linear_bias_residual_layer_norm_with_stats_bf16_linear =
                    load_symbol<LinearBiasResidualLayerNormWithStatsBf16LinearFn>(
                        tile_handle,
                        "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_linear_float32");
                linear_bias_residual_layer_norm_with_stats_bf16_residual =
                    load_symbol<LinearBiasResidualLayerNormWithStatsBf16ResidualFn>(
                        tile_handle,
                        "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_residual_float32");
                linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual =
                    load_symbol<LinearBiasResidualLayerNormWithStatsBf16LinearBf16ResidualFn>(
                        tile_handle,
                        "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_float32");
                linear_bias_residual_layer_norm_with_stats_bf16_residual_bf16_norm =
                    load_symbol<LinearBiasResidualLayerNormWithStatsBf16ResidualBf16NormFn>(
                        tile_handle,
                        "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_residual_bf16_norm_float32");
                linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_bf16_norm =
                    load_symbol<LinearBiasResidualLayerNormWithStatsBf16LinearBf16ResidualBf16NormFn>(
                        tile_handle,
                        "nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_bf16_norm_float32");
                split_qkv_to_heads_add_bias = load_symbol<SplitQkvToHeadsAddBiasFn>(
                    tile_handle, "nfn_native_tile_split_qkv_to_heads_add_bias_float32");
                merge_heads = load_symbol<MergeHeadsFn>(tile_handle, "nfn_native_tile_merge_heads_float32");
                layer_norm = load_symbol<LayerNormFn>(tile_handle, "nfn_native_tile_layer_norm_float32");
                layer_norm_with_stats = load_symbol<LayerNormWithStatsFn>(
                    tile_handle, "nfn_native_tile_layer_norm_with_stats_float32");
                layer_norm_with_stats_bf16_out = load_symbol<LayerNormWithStatsBf16OutFn>(
                    tile_handle, "nfn_native_tile_layer_norm_with_stats_bf16_out_float32");
                layer_norm_apply_stats_bf16_out = load_symbol<LayerNormApplyStatsBf16OutFn>(
                    tile_handle, "nfn_native_tile_layer_norm_apply_stats_bf16_out_float32");
                layer_norm_backward_input = load_symbol<LayerNormBackwardInputFn>(
                    tile_handle, "nfn_native_tile_layer_norm_backward_input_float32");
                layer_norm_backward_input_with_stats = load_symbol<LayerNormBackwardInputWithStatsFn>(
                    tile_handle, "nfn_native_tile_layer_norm_backward_input_with_stats_float32");
                layer_norm_backward_input_residual_add_with_stats =
                    load_symbol<LayerNormBackwardInputResidualAddWithStatsFn>(
                        tile_handle, "nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_float32");
                layer_norm_backward_input_residual_add_with_stats_bf16_bits =
                    load_symbol<LayerNormBackwardInputResidualAddWithStatsBf16BitsFn>(
                        tile_handle,
                        "nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_bf16_bits_float32");
                layer_norm_backward_affine_accumulate = load_symbol<LayerNormBackwardAffineAccumulateFn>(
                    tile_handle, "nfn_native_tile_layer_norm_backward_affine_accumulate_float32");
                layer_norm_backward_affine_accumulate_with_stats =
                    load_symbol<LayerNormBackwardAffineAccumulateWithStatsFn>(
                        tile_handle, "nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_float32");
                layer_norm_backward_affine_accumulate_with_stats_bf16_bits =
                    load_symbol<LayerNormBackwardAffineAccumulateWithStatsBf16BitsFn>(
                        tile_handle,
                        "nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_bf16_bits_float32");
                layer_norm_backward_affine_residual_add_accumulate_with_stats =
                    load_symbol<LayerNormBackwardAffineResidualAddAccumulateWithStatsFn>(
                        tile_handle,
                        "nfn_native_tile_layer_norm_backward_affine_residual_add_accumulate_with_stats_float32");
                layer_norm_backward_affine_residual_add_accumulate_with_stats_bf16_bits =
                    load_symbol<LayerNormBackwardAffineResidualAddAccumulateWithStatsBf16BitsFn>(
                        tile_handle,
                        "nfn_native_tile_layer_norm_backward_affine_residual_add_accumulate_with_stats_bf16_bits_float32");
                linear = load_symbol<LinearFn>(tile_handle, "nfn_native_tile_linear_float32");
                linear_weight_bf16 =
                    load_symbol<LinearWeightBf16Fn>(tile_handle, "nfn_native_tile_linear_weight_bf16_float32");
                linear_bf16_output =
                    load_symbol<LinearBf16OutputFn>(tile_handle, "nfn_native_tile_linear_bf16_output_float32");
                linear_weight_bf16_output =
                    load_symbol<LinearWeightBf16OutputFn>(
                        tile_handle, "nfn_native_tile_linear_weight_bf16_output_float32");
                linear_bf16_input_weight_bf16_output =
                    load_symbol<LinearBf16InputWeightBf16OutputFn>(
                        tile_handle, "nfn_native_tile_linear_bf16_input_weight_bf16_output_float32");
                linear_bf16_input_float_weight_bf16_output =
                    load_symbol<LinearBf16InputFloatWeightBf16OutputFn>(
                        tile_handle, "nfn_native_tile_linear_bf16_input_float_weight_bf16_output_float32");
                bf16_bits_add_bias_inplace = load_symbol<Bf16BitsAddBiasInplaceFn>(
                    tile_handle, "nfn_native_tile_bf16_bits_add_bias_inplace_float32");
                linear_bf16_input_weight_bf16 =
                    load_symbol<LinearBf16InputWeightBf16Fn>(
                        tile_handle, "nfn_native_tile_linear_bf16_input_weight_bf16_float32");
                linear_weight_bf16_gelu_bf16 =
                    load_symbol<LinearWeightBf16GeluBf16Fn>(
                        tile_handle, "nfn_native_tile_linear_weight_bf16_gelu_bf16_float32");
                linear_bf16_input_weight_bf16_gelu_bf16 =
                    load_symbol<LinearBf16InputWeightBf16GeluBf16Fn>(
                        tile_handle, "nfn_native_tile_linear_bf16_input_weight_bf16_gelu_bf16_float32");
                linear_backward_input = load_symbol<LinearBackwardInputFn>(
                    tile_handle, "nfn_native_tile_linear_backward_input_float32");
                linear_backward_input_weight_bf16 =
                    load_symbol<LinearBackwardInputWeightBf16Fn>(
                        tile_handle, "nfn_native_tile_linear_backward_input_weight_bf16_float32");
                linear_backward_input_weight_bf16_to_bf16_bits =
                    load_symbol<LinearBackwardInputWeightBf16ToBf16BitsFn>(
                        tile_handle, "nfn_native_tile_linear_backward_input_weight_bf16_to_bf16_bits_float32");
                linear_backward_input_bf16_bits_weight_bf16 =
                    load_symbol<LinearBackwardInputBf16BitsWeightBf16Fn>(
                        tile_handle, "nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_float32");
                linear_backward_input_bf16_bits = load_symbol<LinearBackwardInputBf16BitsFn>(
                    tile_handle, "nfn_native_tile_linear_backward_input_bf16_bits_float32");
                linear_backward_input_dgelu_weight_bf16_bits =
                    load_symbol<LinearBackwardInputDgeluWeightBf16BitsFn>(
                        tile_handle, "nfn_native_tile_linear_backward_input_dgelu_weight_bf16_bits_float32");
                linear_backward_input_dgelu_weight_bf16_bits_only =
                    load_symbol<LinearBackwardInputDgeluWeightBf16BitsOnlyFn>(
                        tile_handle,
                        "nfn_native_tile_linear_backward_input_dgelu_weight_bf16_bits_only_float32");
                linear_backward_weight_accumulate = load_symbol<LinearBackwardWeightAccumulateFn>(
                    tile_handle, "nfn_native_tile_linear_backward_weight_accumulate_float32");
                linear_backward_weight_bias_accumulate_bf16 =
                    load_symbol<LinearBackwardWeightBiasAccumulateFn>(
                        tile_handle, "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_float32");
                linear_backward_weight_bias_accumulate_bf16_bits =
                    load_symbol<LinearBackwardWeightBiasAccumulateBf16BitsFn>(
                        tile_handle, "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_float32");
                linear_backward_weight_bias_accumulate_bf16_bits_beta =
                    load_symbol<LinearBackwardWeightBiasAccumulateBf16BitsBetaFn>(
                        tile_handle,
                        "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_float32_beta");
                linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits =
                    load_symbol<LinearBackwardWeightBiasAccumulateBf16BitsBf16BitsFn>(
                        tile_handle,
                        "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32");
                linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_beta =
                    load_symbol<LinearBackwardWeightBiasAccumulateBf16BitsBf16BitsBetaFn>(
                        tile_handle,
                        "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32_beta");
                linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_to_bf16_bits =
                    load_symbol<LinearBackwardWeightBiasAccumulateBf16BitsBf16BitsToBf16BitsFn>(
                        tile_handle,
                        "nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_to_bf16_bits_float32");
                linear_backward_weight_accumulate_float32_bf16_bits =
                    load_symbol<LinearBackwardWeightAccumulateFloat32Bf16BitsFn>(
                        tile_handle, "nfn_native_tile_linear_backward_weight_accumulate_float32_bf16_bits");
                linear_backward_weight_accumulate_bf16_bits_bf16_bits =
                    load_symbol<LinearBackwardWeightAccumulateBf16BitsBf16BitsFn>(
                        tile_handle,
                        "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32");
                linear_backward_weight_accumulate_bf16_bits_bf16_bits_beta =
                    load_symbol<LinearBackwardWeightAccumulateBf16BitsBf16BitsBetaFn>(
                        tile_handle,
                        "nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta");
                linear_backward_weight_bias_accumulate_float32_bf16_bits =
                    load_symbol<LinearBackwardWeightBiasAccumulateFloat32Bf16BitsFn>(
                        tile_handle, "nfn_native_tile_linear_backward_weight_bias_accumulate_float32_bf16_bits");
                linear_backward_weight_bias_accumulate_float32_bf16_bits_beta =
                    load_symbol<LinearBackwardWeightBiasAccumulateFloat32Bf16BitsBetaFn>(
                        tile_handle,
                        "nfn_native_tile_linear_backward_weight_bias_accumulate_float32_bf16_bits_beta");
                gelu_add_bias_bf16_act =
                    load_symbol<GeluAddBiasBf16ActFn>(tile_handle, "nfn_native_tile_gelu_add_bias_bf16_act_float32");
                gelu_backward_inplace = load_symbol<GeluBackwardInplaceFn>(
                    tile_handle, "nfn_native_tile_gelu_backward_inplace_float32");
                gelu_backward_inplace_bf16_bits = load_symbol<GeluBackwardInplaceBf16BitsFn>(
                    tile_handle, "nfn_native_tile_gelu_backward_inplace_bf16_bits_float32");
                attention = load_symbol<AttentionFn>(tile_handle, "nfn_native_tile_scaled_dot_product_attention_float32");
                attention_stats_reset = load_symbol<AttentionStatsResetFn>(
                    tile_handle, "nfn_native_tile_attention_forward_stats_reset");
                attention_row_launch_count = load_symbol<AttentionStatsCountFn>(
                    tile_handle, "nfn_native_tile_attention_forward_row_launch_count");
                attention_forward_tk_launch_count = load_symbol<AttentionStatsCountFn>(
                    tile_handle, "nfn_native_tile_attention_forward_tk_launch_count");
                attention_backward_tk_launch_count = load_symbol<AttentionStatsCountFn>(
                    tile_handle, "nfn_native_tile_attention_backward_tk_launch_count");
                attention_tk_workspace_allocation_count_fn = load_symbol<AttentionStatsCountFn>(
                    tile_handle, "nfn_native_tile_attention_tk_workspace_allocation_count");
                attention_tk_workspace_element_capacity_fn = load_symbol<AttentionStatsCountFn>(
                    tile_handle, "nfn_native_tile_attention_tk_workspace_element_capacity");
                attention_tk_workspace_row_capacity_fn = load_symbol<AttentionStatsCountFn>(
                    tile_handle, "nfn_native_tile_attention_tk_workspace_row_capacity");
                attention_row_fallback_count = load_symbol<AttentionStatsCountFn>(
                    tile_handle, "nfn_native_tile_attention_forward_row_fallback_count");
                attention_scalar_launch_count = load_symbol<AttentionStatsCountFn>(
                    tile_handle, "nfn_native_tile_attention_forward_scalar_launch_count");
                attention_row_last_error = load_symbol<AttentionStatsErrorFn>(
                    tile_handle, "nfn_native_tile_attention_forward_row_last_error");
                attention_row_prelaunch_clear_error = load_symbol<AttentionStatsErrorFn>(
                    tile_handle, "nfn_native_tile_attention_forward_row_prelaunch_clear_error");
                attention_row_prelaunch_peek_error = load_symbol<AttentionStatsErrorFn>(
                    tile_handle, "nfn_native_tile_attention_forward_row_prelaunch_peek_error");
                attention_row_grid_x = load_symbol<AttentionStatsCountFn>(
                    tile_handle, "nfn_native_tile_attention_forward_row_grid_x");
                attention_row_grid_y = load_symbol<AttentionStatsCountFn>(
                    tile_handle, "nfn_native_tile_attention_forward_row_grid_y");
                attention_row_grid_z = load_symbol<AttentionStatsCountFn>(
                    tile_handle, "nfn_native_tile_attention_forward_row_grid_z");
                attention_row_block_x = load_symbol<AttentionStatsCountFn>(
                    tile_handle, "nfn_native_tile_attention_forward_row_block_x");
                attention_row_attr_status = load_symbol<AttentionStatsErrorFn>(
                    tile_handle, "nfn_native_tile_attention_forward_row_attr_status");
                attention_row_attr_max_threads_per_block = load_symbol<AttentionStatsErrorFn>(
                    tile_handle, "nfn_native_tile_attention_forward_row_attr_max_threads_per_block");
                attention_row_attr_num_regs = load_symbol<AttentionStatsErrorFn>(
                    tile_handle, "nfn_native_tile_attention_forward_row_attr_num_regs");
                attention_row_attr_shared_size_bytes = load_symbol<AttentionStatsCountFn>(
                    tile_handle, "nfn_native_tile_attention_forward_row_attr_shared_size_bytes");
                attention_row_attr_const_size_bytes = load_symbol<AttentionStatsCountFn>(
                    tile_handle, "nfn_native_tile_attention_forward_row_attr_const_size_bytes");
                attention_row_attr_local_size_bytes = load_symbol<AttentionStatsCountFn>(
                    tile_handle, "nfn_native_tile_attention_forward_row_attr_local_size_bytes");
                trainer_linear_stats_reset = load_symbol<TrainerLinearStatsResetFn>(
                    tile_handle, "nfn_native_tile_trainer_linear_stats_reset");
                trainer_linear_bf16_cache_reset = load_symbol<TrainerLinearStatsResetFn>(
                    tile_handle, "nfn_native_tile_trainer_linear_bf16_cache_reset");
                trainer_linear_bf16_gemm_count_fn = load_symbol<TrainerLinearStatsCountFn>(
                    tile_handle, "nfn_native_tile_trainer_linear_bf16_gemm_count");
                trainer_linear_tk_gemm_count_fn = load_symbol<TrainerLinearStatsCountFn>(
                    tile_handle, "nfn_native_tile_trainer_linear_tk_gemm_count");
                trainer_linear_tk_float_out_gemm_count_fn = load_symbol<TrainerLinearStatsCountFn>(
                    tile_handle, "nfn_native_tile_trainer_linear_tk_float_out_gemm_count");
                trainer_linear_cublaslt_gemm_count_fn = load_symbol<TrainerLinearStatsCountFn>(
                    tile_handle, "nfn_native_tile_trainer_linear_cublaslt_gemm_count");
                trainer_linear_sgemm_count_fn = load_symbol<TrainerLinearStatsCountFn>(
                    tile_handle, "nfn_native_tile_trainer_linear_sgemm_count");
                trainer_linear_bf16_a_pack_count_fn = load_symbol<TrainerLinearStatsCountFn>(
                    tile_handle, "nfn_native_tile_trainer_linear_bf16_a_pack_count");
                trainer_linear_bf16_a_cache_hit_count_fn = load_symbol<TrainerLinearStatsCountFn>(
                    tile_handle, "nfn_native_tile_trainer_linear_bf16_a_cache_hit_count");
                trainer_linear_bf16_cache_reset_count_fn = load_symbol<TrainerLinearStatsCountFn>(
                    tile_handle, "nfn_native_tile_trainer_linear_bf16_cache_reset_count");
                trainer_linear_bf16_workspace_allocation_count_fn = load_symbol<TrainerLinearStatsCountFn>(
                    tile_handle, "nfn_native_tile_trainer_linear_bf16_workspace_allocation_count");
                trainer_linear_bf16_workspace_a_capacity_fn = load_symbol<TrainerLinearStatsCountFn>(
                    tile_handle, "nfn_native_tile_trainer_linear_bf16_workspace_a_capacity");
                trainer_linear_bf16_workspace_b_capacity_fn = load_symbol<TrainerLinearStatsCountFn>(
                    tile_handle, "nfn_native_tile_trainer_linear_bf16_workspace_b_capacity");
                trainer_linear_bf16_cached_a_capacity_fn = load_symbol<TrainerLinearStatsCountFn>(
                    tile_handle, "nfn_native_tile_trainer_linear_bf16_cached_a_capacity");
                trainer_linear_bf16_cache_entry_count_fn = load_symbol<TrainerLinearStatsCountFn>(
                    tile_handle, "nfn_native_tile_trainer_linear_bf16_cache_entry_count");
                trainer_linear_shape_stats_count_fn = load_symbol<TrainerLinearStatsCountFn>(
                    tile_handle, "nfn_native_tile_trainer_linear_shape_stats_count");
                trainer_linear_shape_stats_entry_fn = load_symbol<TrainerLinearShapeStatsEntryFn>(
                    tile_handle, "nfn_native_tile_trainer_linear_shape_stats_entry");
                attention_stats_reset();
                trainer_linear_stats_reset();
                attention_backward_to_qkv_reuse_forward = load_symbol<AttentionBackwardToQkvReuseForwardFn>(
                    tile_handle, "nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_reuse_forward_from_merged_grad_float32");
                packed_attention_forward = load_symbol<PackedAttentionForwardFn>(
                    tile_handle, "nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32");
                packed_attention_forward_store_lse = load_symbol<PackedAttentionForwardStoreLseFn>(
                    tile_handle, "nfn_native_tile_scaled_dot_product_attention_packed_qkv_store_lse_bf16_float32");
                packed_attention_backward_to_qkv = load_symbol<PackedAttentionBackwardToQkvFn>(
                    tile_handle,
                    "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_merged_grad_float32");
                packed_attention_backward_to_qkv_saved_lse = load_symbol<PackedAttentionBackwardToQkvSavedLseFn>(
                    tile_handle,
                    "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_saved_lse_bf16_from_merged_grad_float32");
                packed_attention_backward_to_qkv_bf16_bits =
                    load_symbol<PackedAttentionBackwardToQkvBf16BitsFn>(
                        tile_handle,
                        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_merged_grad_float32");
                packed_attention_backward_to_qkv_bf16_bits_saved_lse =
                    load_symbol<PackedAttentionBackwardToQkvBf16BitsSavedLseFn>(
                        tile_handle,
                        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_merged_grad_float32");
                packed_attention_backward_to_qkv_bf16_bits_from_bf16_grad =
                    load_symbol<PackedAttentionBackwardToQkvBf16BitsFromBf16GradFn>(
                        tile_handle,
                        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_bf16_merged_grad_float32");
                packed_attention_backward_to_qkv_bf16_bits_saved_lse_from_bf16_grad =
                    load_symbol<PackedAttentionBackwardToQkvBf16BitsSavedLseFromBf16GradFn>(
                        tile_handle,
                        "nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_bf16_merged_grad_float32");
                attention_store_forward_tk = load_symbol<AttentionStoreForwardTkFn>(
                    tile_handle, "nfn_native_tile_scaled_dot_product_attention_store_tk_bf16_float32");
                store_attention_tk_workspace = load_symbol<StoreAttentionTkWorkspaceFn>(
                    tile_handle, "nfn_native_tile_attention_tk_store_forward_workspace_bf16");
                attention_backward_to_qkv_from_saved_tk = load_symbol<AttentionBackwardToQkvFromSavedTkFn>(
                    tile_handle,
                    "nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_saved_tk_bf16_from_merged_grad_float32");
                ce_partials = load_symbol<TokenCrossEntropyPartialsFn>(
                    tile_handle, "nfn_native_tile_token_cross_entropy_partials_float32");
                ce_partials_bf16_bits = load_symbol<TokenCrossEntropyPartialsBf16BitsFn>(
                    tile_handle, "nfn_native_tile_token_cross_entropy_partials_bf16_bits");
                ce_partials_strided = load_symbol<TokenCrossEntropyPartialsStridedFn>(
                    tile_handle, "nfn_native_tile_token_cross_entropy_partials_strided_float32");
                ce_partials_strided_bf16_bits = load_symbol<TokenCrossEntropyPartialsStridedBf16BitsFn>(
                    tile_handle, "nfn_native_tile_token_cross_entropy_partials_strided_bf16_bits");
                ce_partials_strided_bf16_bits_u16_targets =
                    load_symbol<TokenCrossEntropyPartialsStridedBf16BitsU16TargetsFn>(
                        tile_handle,
                        "nfn_native_tile_token_cross_entropy_partials_strided_bf16_bits_u16_targets");
                ce_backward_inplace_workspace = load_symbol<TokenCrossEntropyBackwardInplaceWorkspaceFn>(
                    tile_handle, "nfn_native_tile_token_cross_entropy_backward_inplace_with_workspace_float32");
                ce_backward_inplace_bf16_bits_workspace =
                    load_symbol<TokenCrossEntropyBackwardInplaceBf16BitsWorkspaceFn>(
                        tile_handle, "nfn_native_tile_token_cross_entropy_backward_inplace_bf16_bits_with_workspace");
                ce_backward_inplace_strided_workspace =
                    load_symbol<TokenCrossEntropyBackwardInplaceStridedWorkspaceFn>(
                        tile_handle,
                        "nfn_native_tile_token_cross_entropy_backward_inplace_strided_with_workspace_float32");
                ce_backward_inplace_strided_bf16_bits_workspace =
                    load_symbol<TokenCrossEntropyBackwardInplaceStridedBf16BitsWorkspaceFn>(
                        tile_handle,
                        "nfn_native_tile_token_cross_entropy_backward_inplace_strided_bf16_bits_with_workspace");
                ce_backward_inplace_strided_bf16_bits_u16_targets_workspace =
                    load_symbol<TokenCrossEntropyBackwardInplaceStridedBf16BitsU16TargetsWorkspaceFn>(
                        tile_handle,
                        "nfn_native_tile_token_cross_entropy_backward_inplace_strided_bf16_bits_u16_targets_with_workspace");
                adamw_many_with_device_scale = load_symbol<AdamWManyWithDeviceScaleFn>(
                    tile_handle, "nfn_native_tile_adamw_step_many_with_device_scale_float32");
                adamw_many_with_device_scale_bf16_shadow =
                    load_symbol<AdamWManyWithDeviceScaleBf16ShadowFn>(
                        tile_handle,
                        "nfn_native_tile_adamw_step_many_with_device_scale_bf16_shadow_float32");
                adamw_many_with_device_scale_bf16_param =
                    load_symbol<AdamWManyWithDeviceScaleBf16ParamFn>(
                        tile_handle,
                        "nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_float32");
            }
        }
    }

    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(cfg);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    if (error.empty()) {
        std::string runtime_error;
        for (const std::string& candidate : runtime_candidates) {
            cuda_lib_path = candidate;
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_runtime_loaded = true;
                break;
            }
            runtime_error = dl_last_error("dlopen CUDA runtime failed");
        }
        if (cuda_handle == nullptr) {
            error = runtime_error.empty() ? "could not load CUDA runtime" : runtime_error;
        }
    }

    if (error.empty()) {
        cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
        cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
        cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
        cuda_memcpy_async = load_symbol<CudaMemcpyAsyncFn>(cuda_handle, "cudaMemcpyAsync");
        cuda_memset_async = load_symbol<CudaMemsetAsyncFn>(cuda_handle, "cudaMemsetAsync");
        cuda_malloc_async = load_symbol<CudaMallocAsyncFn>(cuda_handle, "cudaMallocAsync");
        cuda_free_async = load_symbol<CudaFreeAsyncFn>(cuda_handle, "cudaFreeAsync");
        cuda_host_alloc = load_symbol<CudaHostAllocFn>(cuda_handle, "cudaHostAlloc");
        cuda_free_host = load_symbol<CudaFreeHostFn>(cuda_handle, "cudaFreeHost");
        cuda_device_synchronize = load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
        cuda_get_error_string = load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
        cuda_runtime_get_version = load_symbol<CudaVersionFn>(cuda_handle, "cudaRuntimeGetVersion");
        cuda_driver_get_version = load_symbol<CudaVersionFn>(cuda_handle, "cudaDriverGetVersion");
        cuda_event_create_with_flags =
            load_symbol<CudaEventCreateWithFlagsFn>(cuda_handle, "cudaEventCreateWithFlags");
        cuda_event_record = load_symbol<CudaEventRecordFn>(cuda_handle, "cudaEventRecord");
        cuda_event_elapsed_time = load_symbol<CudaEventElapsedTimeFn>(cuda_handle, "cudaEventElapsedTime");
        cuda_event_destroy = load_symbol<CudaEventDestroyFn>(cuda_handle, "cudaEventDestroy");
        if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr ||
            cuda_memcpy_async == nullptr || cuda_host_alloc == nullptr || cuda_free_host == nullptr ||
            cuda_device_synchronize == nullptr) {
            error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaMemcpyAsync/cudaHostAlloc/cudaFreeHost/cudaDeviceSynchronize";
        } else if (cuda_runtime_get_version == nullptr || cuda_driver_get_version == nullptr) {
            error = "CUDA runtime is missing cudaRuntimeGetVersion/cudaDriverGetVersion";
        } else {
            cuda_runtime_preflight_checked = true;
            cuda_runtime_version_status = cuda_runtime_get_version(&cuda_runtime_version);
            cuda_driver_version_status = cuda_driver_get_version(&cuda_driver_version);
            if (cuda_runtime_version_status != 0) {
                error = cuda_error(cuda_runtime_version_status, "cudaRuntimeGetVersion");
            } else if (cuda_driver_version_status != 0) {
                error = cuda_error(cuda_driver_version_status, "cudaDriverGetVersion");
            } else if (cuda_driver_version <= 0) {
                std::ostringstream out;
                out << "CUDA driver is unavailable to the native trainer: loaded " << cuda_lib_path
                    << " reports runtime " << cuda_version_string(cuda_runtime_version)
                    << " but cudaDriverGetVersion returned " << cuda_driver_version
                    << ". Ensure the process has GPU/driver access before benchmarking SM120 throughput.";
                error = out.str();
            } else if (cuda_driver_version > 0 && cuda_runtime_version > 0 && cuda_driver_version < cuda_runtime_version) {
                std::ostringstream out;
                out << "CUDA runtime/driver mismatch: loaded " << cuda_lib_path
                    << " reports runtime " << cuda_version_string(cuda_runtime_version)
                    << " but the driver reports " << cuda_version_string(cuda_driver_version)
                    << ". Rebuild or run with --cuda-runtime-lib/NFN_CUDA_RUNTIME_LIB pointing at a runtime supported by the installed driver.";
                error = out.str();
            }
        }
    }

    struct StageTimingRecord {
        std::string name;
        double total_ms = 0.0;
        std::int64_t count = 0;
    };
    struct StageTimingEvent {
        std::size_t record_index = 0;
        void* start = nullptr;
        void* stop = nullptr;
    };
    const std::string stage_timing_env =
        env_or_empty_any({"NFN_NATIVE_GPT_STAGE_TIMING", "NFN_NATIVE_GPT2_STAGE_TIMING"});
    const bool stage_timing_requested =
        stage_timing_env == "1" || stage_timing_env == "true" || stage_timing_env == "TRUE";
    const std::string store_mlp_activations_env =
        env_or_empty_any({"NFN_NATIVE_GPT_STORE_MLP_ACTIVATIONS", "NFN_NATIVE_GPT2_STORE_MLP_ACTIVATIONS"});
    const bool store_mlp_activations_enabled =
        store_mlp_activations_env.empty() ||
        store_mlp_activations_env == "1" ||
        store_mlp_activations_env == "true" ||
        store_mlp_activations_env == "TRUE" ||
        store_mlp_activations_env == "on" ||
        store_mlp_activations_env == "ON";
    const std::string store_attention_activations_env =
        env_or_empty_any({"NFN_NATIVE_GPT_STORE_ATTENTION_ACTIVATIONS",
                          "NFN_NATIVE_GPT2_STORE_ATTENTION_ACTIVATIONS"});
    const bool store_attention_activations_enabled =
        store_attention_activations_env == "1" ||
        store_attention_activations_env == "true" ||
        store_attention_activations_env == "TRUE" ||
        store_attention_activations_env == "on" ||
        store_attention_activations_env == "ON";
    const bool packed_qkv_attention_enabled = packed_qkv_attention_default_enabled();
    const std::string store_packed_attention_activations_env =
        env_or_empty_any({"NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_ACTIVATIONS",
                          "NFN_NATIVE_GPT2_STORE_PACKED_ATTENTION_ACTIVATIONS"});
    const bool store_packed_attention_activations_enabled =
        packed_qkv_attention_enabled &&
        env_flag_enabled_or_default(store_packed_attention_activations_env, true);
    const std::string store_residual1_activations_env =
        env_or_empty_any({"NFN_NATIVE_GPT_STORE_RESIDUAL1_ACTIVATIONS",
                          "NFN_NATIVE_GPT2_STORE_RESIDUAL1_ACTIVATIONS"});
    const bool store_residual1_activations_enabled =
        env_flag_enabled_or_default(store_residual1_activations_env, true);
    const bool fuse_residual1_store_enabled =
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_FUSE_RESIDUAL1_STORE",
                              "NFN_NATIVE_GPT2_FUSE_RESIDUAL1_STORE"}),
            true);
    const bool fuse_attention_residual_ln2_enabled = fuse_attention_residual_ln2_default_enabled();
    const std::string fuse_mlp_proj_dgelu_env =
        env_or_empty_any({"NFN_NATIVE_GPT_FUSE_MLP_PROJ_DGELU",
                          "NFN_NATIVE_GPT2_FUSE_MLP_PROJ_DGELU"});
    const bool fuse_mlp_proj_dgelu_enabled =
        fuse_mlp_proj_dgelu_env.empty() ||
        fuse_mlp_proj_dgelu_env == "1" ||
        fuse_mlp_proj_dgelu_env == "true" ||
        fuse_mlp_proj_dgelu_env == "TRUE" ||
        fuse_mlp_proj_dgelu_env == "on" ||
        fuse_mlp_proj_dgelu_env == "ON";
    const bool bf16_mlp_grad_handoff_enabled =
        fuse_mlp_proj_dgelu_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_BF16_MLP_GRAD_HANDOFF",
                              "NFN_NATIVE_GPT2_BF16_MLP_GRAD_HANDOFF"}),
            true);
    const bool elide_mlp_dgelu_float_grad_enabled =
        bf16_mlp_grad_handoff_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_ELIDE_MLP_DGELU_FLOAT_GRAD",
                              "NFN_NATIVE_GPT2_ELIDE_MLP_DGELU_FLOAT_GRAD"}),
            true);
    const bool bf16_qkv_grad_handoff_enabled =
        packed_qkv_attention_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_BF16_QKV_GRAD_HANDOFF",
                              "NFN_NATIVE_GPT2_BF16_QKV_GRAD_HANDOFF"}),
            true);
    const bool bf16_attention_grad_out_handoff_enabled =
        packed_qkv_attention_enabled &&
        bf16_qkv_grad_handoff_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_BF16_ATTENTION_GRAD_OUT",
                              "NFN_NATIVE_GPT2_BF16_ATTENTION_GRAD_OUT"}),
            false);
    const bool ln1_bf16_qkv_forward_enabled =
        packed_qkv_attention_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_LN1_BF16_QKV_FORWARD",
                              "NFN_NATIVE_GPT2_LN1_BF16_QKV_FORWARD"}),
            true);
    const bool direct_bf16_qkv_grad_scratch_enabled =
        bf16_qkv_grad_handoff_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_DIRECT_BF16_QKV_GRAD_SCRATCH",
                              "NFN_NATIVE_GPT2_DIRECT_BF16_QKV_GRAD_SCRATCH"}),
            true);
    const bool grad_qkv_float_scratch_elided =
        bf16_qkv_grad_handoff_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_ELIDE_QKV_FLOAT_GRAD_SCRATCH",
                              "NFN_NATIVE_GPT2_ELIDE_QKV_FLOAT_GRAD_SCRATCH"}),
            true);
    const std::int64_t grad_qkv_float_scratch_elements =
        grad_qkv_float_scratch_elided ? 0 : qkv_activation_elements;
    const std::int64_t grad_qkv_float_scratch_bytes_elided =
        grad_qkv_float_scratch_elided
            ? qkv_activation_elements * static_cast<std::int64_t>(sizeof(float))
            : 0;
    const bool bf16_qkv_dweight_enabled =
        direct_bf16_qkv_grad_scratch_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_BF16_QKV_DWEIGHT",
                              "NFN_NATIVE_GPT2_BF16_QKV_DWEIGHT"}),
            true);
    const bool bf16_block_dweight_staging_enabled =
        (bf16_mlp_grad_handoff_enabled || bf16_qkv_dweight_enabled) &&
            env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_BF16_BLOCK_DWEIGHT_STAGING",
                              "NFN_NATIVE_GPT2_BF16_BLOCK_DWEIGHT_STAGING"}),
            false);
    const bool fuse_qkv_bias_tk_gemm_enabled =
        packed_qkv_attention_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_FUSE_QKV_BIAS_TK_GEMM",
                              "NFN_NATIVE_GPT2_FUSE_QKV_BIAS_TK_GEMM"}),
            true);
    const bool reuse_packed_ln2_fc_gelu_enabled =
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_REUSE_PACKED_LN2_FC_GELU",
                              "NFN_NATIVE_GPT2_REUSE_PACKED_LN2_FC_GELU"}),
            true);
    const bool fused_ln2_bf16_out_enabled =
        reuse_packed_ln2_fc_gelu_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_FUSE_LN2_BF16_OUT",
                              "NFN_NATIVE_GPT2_FUSE_LN2_BF16_OUT"}),
            true);
    const bool fused_ln2_bf16_norm_float_store_elision_enabled =
        fused_ln2_bf16_out_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_ELIDE_LN2_BF16_NORM_FLOAT_STORE",
                              "NFN_NATIVE_GPT2_ELIDE_LN2_BF16_NORM_FLOAT_STORE"}),
            true);
    const bool bf16_block_weight_param_update_enabled =
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_BF16_BLOCK_WEIGHT_PARAMS",
                              "NFN_NATIVE_GPT2_BF16_BLOCK_WEIGHT_PARAMS"}),
            true);
    const bool direct_bf16_block_weight_init_enabled =
        bf16_block_weight_param_update_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_DIRECT_BF16_BLOCK_WEIGHT_INIT",
                              "NFN_NATIVE_GPT2_DIRECT_BF16_BLOCK_WEIGHT_INIT"}),
            true);
    const bool token_weight_bf16_shadow_enabled =
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_TOKEN_WEIGHT_BF16_SHADOW",
                              "NFN_NATIVE_GPT2_TOKEN_WEIGHT_BF16_SHADOW"}),
            true);
    const bool fuse_token_weight_bf16_initial_refresh_enabled =
        token_weight_bf16_shadow_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_FUSE_TOKEN_WEIGHT_BF16_INIT",
                              "NFN_NATIVE_GPT2_FUSE_TOKEN_WEIGHT_BF16_INIT"}),
            true);
    const bool legacy_mod17_token_weight_init_enabled =
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_TOKEN_WEIGHT_INIT_LEGACY_MOD17",
                              "NFN_NATIVE_GPT2_TOKEN_WEIGHT_INIT_LEGACY_MOD17"}),
            false);
    const bool dweight_first_microbatch_beta_zero_enabled =
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_DWEIGHT_FIRST_MICROBATCH_BETA_ZERO",
                              "NFN_NATIVE_GPT2_DWEIGHT_FIRST_MICROBATCH_BETA_ZERO"}),
            true);
    const bool fuse_adamw_bf16_shadow_refresh_enabled =
        !bf16_block_weight_param_update_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_FUSE_ADAMW_BF16_SHADOW_REFRESH",
                              "NFN_NATIVE_GPT2_FUSE_ADAMW_BF16_SHADOW_REFRESH"}),
            false);
    const bool packed_qkv_float_attention_tape_elided = packed_qkv_attention_enabled;
    const std::int64_t packed_qkv_float_attention_tape_elements_elided =
        packed_qkv_float_attention_tape_elided
            ? qkv_activation_elements + activation_elements * 5
            : 0;
    const bool layer_norm_stats_enabled = true;
    const bool fuse_ln_backward_residual_enabled =
        layer_norm_stats_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_FUSE_LN_BACKWARD_RESIDUAL",
                              "NFN_NATIVE_GPT2_FUSE_LN_BACKWARD_RESIDUAL"}),
            true);
    const bool fuse_ln_backward_affine_residual_enabled =
        fuse_ln_backward_residual_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_FUSE_LN_BACKWARD_AFFINE_RESIDUAL",
                              "NFN_NATIVE_GPT2_FUSE_LN_BACKWARD_AFFINE_RESIDUAL"}),
            true);
    const bool bf16_residual1_ln_backward_enabled =
        layer_norm_stats_enabled &&
        fuse_ln_backward_residual_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_BF16_RESIDUAL1_LN_BACKWARD",
                              "NFN_NATIVE_GPT2_BF16_RESIDUAL1_LN_BACKWARD"}),
            true);
    const std::string lm_head_bf16_logits_env =
        env_or_empty_any({"NFN_NATIVE_GPT_LM_HEAD_BF16_LOGITS", "NFN_NATIVE_GPT2_LM_HEAD_BF16_LOGITS"});
    const bool lm_head_bf16_logits_enabled =
        lm_head_bf16_logits_env.empty() ||
        lm_head_bf16_logits_env == "1" ||
        lm_head_bf16_logits_env == "true" ||
        lm_head_bf16_logits_env == "TRUE" ||
        lm_head_bf16_logits_env == "on" ||
        lm_head_bf16_logits_env == "ON";
    const bool lm_head_bf16_loss_enabled =
        lm_head_bf16_logits_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_BF16_LM_HEAD_LOSS",
                              "NFN_NATIVE_GPT2_BF16_LM_HEAD_LOSS"}),
            true);
    const bool lm_head_bf16_dweight_enabled =
        lm_head_bf16_logits_enabled &&
            env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_LM_HEAD_BF16_DWEIGHT",
                              "NFN_NATIVE_GPT2_LM_HEAD_BF16_DWEIGHT"}),
            true);
    const bool lm_head_prepack_bf16_hidden_enabled =
        lm_head_bf16_logits_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_LM_HEAD_PREPACK_BF16_HIDDEN",
                              "NFN_NATIVE_GPT2_LM_HEAD_PREPACK_BF16_HIDDEN"}),
            true);
    const bool lm_head_public_vocab_ce_enabled =
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_PUBLIC_VOCAB_CE",
                              "NFN_NATIVE_GPT_STRIDED_PUBLIC_VOCAB_CE",
                              "NFN_NATIVE_GPT2_PUBLIC_VOCAB_CE",
                              "NFN_NATIVE_GPT2_STRIDED_PUBLIC_VOCAB_CE"}),
            true);
    const bool direct_u16_token_ids_enabled =
        lm_head_bf16_loss_enabled &&
        lm_head_bf16_logits_enabled &&
        lm_head_public_vocab_ce_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_DIRECT_U16_TOKENS",
                              "NFN_NATIVE_GPT2_DIRECT_U16_TOKENS"}),
            true);
    const bool startup_zero_adamw_state_only_enabled =
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_ZERO_ADAMW_STATE_ONLY",
                              "NFN_NATIVE_GPT2_ZERO_ADAMW_STATE_ONLY"}),
            true);
    const bool startup_zero_adamw_state_ranges_enabled =
        startup_zero_adamw_state_only_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_ZERO_ADAMW_STATE_RANGES",
                              "NFN_NATIVE_GPT2_ZERO_ADAMW_STATE_RANGES"}),
            true);
    const bool startup_cuda_memset_zero_enabled =
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_CUDA_MEMSET_ZERO",
                              "NFN_NATIVE_GPT2_CUDA_MEMSET_ZERO"}),
            true);
    const bool gradient_cuda_memset_zero_enabled =
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_CUDA_MEMSET_GRAD_ZERO",
                              "NFN_NATIVE_GPT2_CUDA_MEMSET_GRAD_ZERO"}),
            true);
    const bool bf16_projection_residual_enabled =
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_BF16_PROJECTION_RESIDUAL",
                              "NFN_NATIVE_GPT2_BF16_PROJECTION_RESIDUAL"}),
            true);
    const std::string startup_zero_init_strategy =
        startup_zero_adamw_state_only_enabled
            ? (startup_zero_adamw_state_ranges_enabled
                   ? (startup_cuda_memset_zero_enabled
                          ? "adamw-state-contiguous-range-cuda-memset"
                          : "adamw-state-contiguous-range-fill")
                   : "adamw-state-fill-many")
            : (startup_cuda_memset_zero_enabled ? "single-arena-cuda-memset" : "single-arena-fill");
    const std::int64_t logit_workspace_elements = lm_head_bf16_loss_enabled ? 0 : logit_elements;
    bool stage_timing_enabled = false;
    std::int64_t stage_timing_event_count = 0;
    std::int64_t stage_timing_dropped_event_count = 0;
    std::vector<StageTimingRecord> stage_timing_records;
    std::vector<StageTimingEvent> stage_timing_events;
    const std::int64_t stage_timing_max_events = std::max<std::int64_t>(
        1,
        env_nonnegative_i64_or({"NFN_NATIVE_GPT_STAGE_TIMING_MAX_EVENTS",
                                "NFN_NATIVE_GPT2_STAGE_TIMING_MAX_EVENTS"},
                               20000));
    if (error.empty() && stage_timing_requested) {
        if (cuda_event_create_with_flags == nullptr || cuda_event_record == nullptr ||
            cuda_event_elapsed_time == nullptr || cuda_event_destroy == nullptr) {
            error = "NFN_NATIVE_GPT_STAGE_TIMING requested but CUDA event APIs are unavailable";
        } else {
            stage_timing_enabled = true;
        }
    }
    auto stage_record_index = [&](const std::string& name) -> std::size_t {
        for (std::size_t i = 0; i < stage_timing_records.size(); ++i) {
            if (stage_timing_records[i].name == name) {
                return i;
            }
        }
        stage_timing_records.push_back(StageTimingRecord{name, 0.0, 0});
        return stage_timing_records.size() - 1;
    };
    auto stage_begin = [&](const std::string& name) -> std::int64_t {
        if (!stage_timing_enabled || stage_timing_event_count >= stage_timing_max_events) {
            if (stage_timing_enabled) {
                stage_timing_dropped_event_count += 1;
            }
            return -1;
        }
        void* start = nullptr;
        void* stop = nullptr;
        int status = cuda_event_create_with_flags(&start, 0);
        if (status == 0) {
            status = cuda_event_create_with_flags(&stop, 0);
        }
        if (status == 0) {
            status = cuda_event_record(start, nullptr);
        }
        if (status != 0) {
            if (start != nullptr) {
                cuda_event_destroy(start);
            }
            if (stop != nullptr) {
                cuda_event_destroy(stop);
            }
            if (error.empty()) {
                error = cuda_error(status, "cudaEventRecord stage timing " + name);
            }
            return -1;
        }
        const std::size_t record_index = stage_record_index(name);
        stage_timing_events.push_back(StageTimingEvent{record_index, start, stop});
        stage_timing_event_count += 1;
        return static_cast<std::int64_t>(stage_timing_events.size() - 1);
    };
    auto stage_end = [&](std::int64_t event_index, const std::string& name) {
        if (event_index < 0 || !stage_timing_enabled) {
            return;
        }
        const int status = cuda_event_record(stage_timing_events[static_cast<std::size_t>(event_index)].stop, nullptr);
        if (status != 0 && error.empty()) {
            error = cuda_error(status, "cudaEventRecord stage timing stop " + name);
        }
    };
    auto run_timed_stage = [&](const std::string& name, const auto& fn) {
        const std::int64_t event = stage_begin(name);
        fn();
        stage_end(event, name);
    };
    auto finalize_stage_timing = [&]() {
        if (!stage_timing_enabled) {
            return;
        }
        if (cuda_device_synchronize != nullptr) {
            const int sync_status = cuda_device_synchronize();
            if (sync_status != 0 && error.empty()) {
                error = cuda_error(sync_status, "cudaDeviceSynchronize stage timing");
            }
        }
        for (StageTimingEvent& event : stage_timing_events) {
            float elapsed = 0.0f;
            const int status = cuda_event_elapsed_time(&elapsed, event.start, event.stop);
            if (status == 0 && event.record_index < stage_timing_records.size()) {
                stage_timing_records[event.record_index].total_ms += static_cast<double>(elapsed);
                stage_timing_records[event.record_index].count += 1;
            } else if (status != 0 && error.empty()) {
                error = cuda_error(status, "cudaEventElapsedTime stage timing");
            }
            if (event.start != nullptr) {
                cuda_event_destroy(event.start);
                event.start = nullptr;
            }
            if (event.stop != nullptr) {
                cuda_event_destroy(event.stop);
                event.stop = nullptr;
            }
        }
    };

    const float initial_token_weight_sample = kInitialTokenWeightSample;

    struct FloatArenaRequest {
        float** ptr = nullptr;
        std::int64_t elements = 0;
        std::int64_t offset = 0;
        std::string name;
    };
    struct Uint16ArenaRequest {
        std::uint16_t** ptr = nullptr;
        std::int64_t elements = 0;
        std::int64_t offset = 0;
        std::string name;
    };
    constexpr std::int64_t kFloatArenaAlignmentElements = 64;
    constexpr std::int64_t kUint16ArenaAlignmentElements = 128;
    std::vector<FloatArenaRequest> float_arena_requests;
    std::vector<Uint16ArenaRequest> uint16_arena_requests;
    std::vector<float*> float_ptrs;
    std::vector<std::int64_t*> int_ptrs;
    std::vector<std::uint16_t*> uint16_ptrs;
    std::vector<std::uint16_t*> pinned_uint16_ptrs;
    std::vector<void*> token_device_ptrs;
    std::vector<void*> descriptor_ptrs;
    float* float_arena = nullptr;
    std::uint16_t* uint16_arena = nullptr;
    void* descriptor_arena = nullptr;
    void* token_device_arena = nullptr;
    std::int64_t* token_i64_arena = nullptr;
    std::uint16_t* token_u16_device_arena = nullptr;
    std::uint16_t* token_u16_pinned_arena = nullptr;
    float** adamw_param_ptrs = nullptr;
    float** adamw_grad_ptrs = nullptr;
    float** adamw_avg_ptrs = nullptr;
    float** adamw_avg_sq_ptrs = nullptr;
    std::int64_t* adamw_elements = nullptr;
    std::int64_t* adamw_bf16_shadow_offsets = nullptr;
    std::int64_t* gradient_partial_offsets = nullptr;
    float* adamw_weight_decays = nullptr;
    float** adamw_float_update_param_ptrs = nullptr;
    float** adamw_float_update_grad_ptrs = nullptr;
    float** adamw_float_update_avg_ptrs = nullptr;
    float** adamw_float_update_avg_sq_ptrs = nullptr;
    std::int64_t* adamw_float_update_elements = nullptr;
    std::int64_t* adamw_float_update_partial_offsets = nullptr;
    float* adamw_float_update_weight_decays = nullptr;
    std::uint16_t** adamw_bf16_param_ptrs = nullptr;
    float** adamw_bf16_param_grad_ptrs = nullptr;
    float** adamw_bf16_param_avg_ptrs = nullptr;
    float** adamw_bf16_param_avg_sq_ptrs = nullptr;
    std::int64_t* adamw_bf16_param_elements = nullptr;
    std::int64_t* adamw_bf16_param_partial_offsets = nullptr;
    float* adamw_bf16_param_weight_decays = nullptr;
    std::uint16_t** adamw_bf16_param_bf16_grad_param_ptrs = nullptr;
    std::uint16_t** adamw_bf16_param_bf16_grad_grad_ptrs = nullptr;
    float** adamw_bf16_param_bf16_grad_avg_ptrs = nullptr;
    float** adamw_bf16_param_bf16_grad_avg_sq_ptrs = nullptr;
    std::int64_t* adamw_bf16_param_bf16_grad_elements = nullptr;
    std::int64_t* adamw_bf16_param_bf16_grad_partial_offsets = nullptr;
    float* adamw_bf16_param_bf16_grad_weight_decays = nullptr;
    float** parameter_fill_ptrs = nullptr;
    std::int64_t* parameter_fill_elements = nullptr;
    float* parameter_fill_values = nullptr;
    std::uint16_t** bf16_parameter_fill_ptrs = nullptr;
    std::int64_t* bf16_parameter_fill_elements = nullptr;
    float* bf16_parameter_fill_values = nullptr;
    std::int64_t parameter_fill_descriptor_count = 0;
    std::int64_t parameter_fill_max_elements = 0;
    std::int64_t parameter_fill_kernel_launches = 0;
    std::int64_t bf16_parameter_fill_descriptor_count = 0;
    std::int64_t bf16_parameter_fill_max_elements = 0;
    std::int64_t bf16_parameter_fill_kernel_launches = 0;
    std::int64_t adamw_descriptor_count = 0;
    std::int64_t adamw_max_elements = 0;
    std::int64_t adamw_float_update_descriptor_count = 0;
    std::int64_t adamw_float_update_max_elements = 0;
    std::int64_t adamw_bf16_param_descriptor_count = 0;
    std::int64_t adamw_bf16_param_max_elements = 0;
    std::int64_t adamw_bf16_param_bf16_grad_descriptor_count = 0;
    std::int64_t adamw_bf16_param_bf16_grad_max_elements = 0;
    std::int64_t adamw_kernel_launches = 0;
    std::int64_t adamw_float_update_kernel_launches = 0;
    std::int64_t adamw_bf16_param_kernel_launches = 0;
    std::int64_t adamw_bf16_param_bf16_grad_kernel_launches = 0;
    std::int64_t gradient_sumsq_kernel_launches = 0;
    std::int64_t accumulation_zero_kernel_launches = 0;
    std::int64_t float_arena_requested_elements = 0;
    std::int64_t float_arena_allocated_elements = 0;
    std::int64_t float_arena_cuda_malloc_count = 0;
    std::int64_t uint16_arena_requested_elements = 0;
    std::int64_t uint16_arena_allocated_elements = 0;
    std::int64_t uint16_arena_cuda_malloc_count = 0;
    std::int64_t token_i64_arena_elements = 0;
    std::int64_t token_u16_device_arena_elements = 0;
    std::int64_t token_u16_pinned_arena_elements = 0;
    std::int64_t token_i64_arena_cuda_malloc_count = 0;
    std::int64_t token_u16_device_arena_cuda_malloc_count = 0;
    std::int64_t token_u16_pinned_arena_cuda_host_alloc_count = 0;
    std::size_t token_device_arena_requested_bytes = 0;
    std::size_t token_device_arena_bytes = 0;
    std::int64_t token_device_arena_cuda_malloc_count = 0;
    std::int64_t token_device_arena_suballocation_count = 0;
    std::int64_t token_device_cuda_mallocs_elided = 1;
    std::int64_t float_arena_zero_fill_count = 0;
    std::int64_t adamw_state_zero_fill_count = 0;
    std::int64_t startup_cuda_memset_zero_fill_count = 0;
    std::int64_t startup_tile_zero_fill_count = 0;
    std::int64_t adamw_state_zero_range_count = 0;
    std::int64_t adamw_state_zero_range_elements = 0;
    std::int64_t gradient_zero_range_count = 0;
    std::int64_t gradient_zero_range_elements = 0;
    std::int64_t gradient_zero_cuda_memset_count = 0;
    std::int64_t gradient_zero_tile_fill_count = 0;
    std::size_t descriptor_arena_requested_bytes = 0;
    std::size_t descriptor_arena_bytes = 0;
    std::int64_t descriptor_arena_cuda_malloc_count = 0;
    std::int64_t descriptor_arena_suballocation_count = 0;
    std::int64_t descriptor_arena_copy_count = 0;
    constexpr std::int64_t kBaseDescriptorDeviceTableCount = 14;
    constexpr std::int64_t kBf16ParamUpdateDescriptorDeviceTableCount = 24;
    const std::int64_t descriptor_device_table_count =
        kBaseDescriptorDeviceTableCount +
        (bf16_block_weight_param_update_enabled ? kBf16ParamUpdateDescriptorDeviceTableCount : 0);
    const std::int64_t descriptor_cuda_mallocs_elided = descriptor_device_table_count - 1;
    const std::int64_t descriptor_arena_copy_calls_elided = descriptor_device_table_count - 1;
    const bool async_device_allocator_requested =
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_CUDA_MALLOC_ASYNC",
                              "NFN_NATIVE_GPT2_CUDA_MALLOC_ASYNC"}),
            false);
    const bool async_device_allocator_enabled =
        async_device_allocator_requested && cuda_malloc_async != nullptr && cuda_free_async != nullptr;
    std::vector<void*> async_device_ptrs;
    std::int64_t device_cuda_malloc_async_count = 0;
    std::int64_t device_cuda_free_async_count = 0;
    std::int64_t device_cuda_malloc_async_fallback_count = 0;
    auto device_malloc = [&](void** ptr, std::size_t bytes) -> int {
        if (async_device_allocator_enabled) {
            const int status = cuda_malloc_async(ptr, bytes, nullptr);
            if (status == 0) {
                async_device_ptrs.push_back(*ptr);
                device_cuda_malloc_async_count += 1;
                return 0;
            }
            device_cuda_malloc_async_fallback_count += 1;
        }
        return cuda_malloc(ptr, bytes);
    };
    auto device_pointer_was_async_allocated = [&](void* ptr) {
        return std::find(async_device_ptrs.begin(), async_device_ptrs.end(), ptr) != async_device_ptrs.end();
    };
    auto device_free = [&](void* ptr, const std::string& name) {
        if (ptr == nullptr) {
            return;
        }
        int status = 0;
        if (device_pointer_was_async_allocated(ptr) && cuda_free_async != nullptr) {
            status = cuda_free_async(ptr, nullptr);
            if (status == 0) {
                device_cuda_free_async_count += 1;
            }
        } else if (cuda_free != nullptr) {
            status = cuda_free(ptr);
        }
        if (status != 0 && error.empty()) {
            error = cuda_error(status, name);
        }
    };
    auto align_float_arena_offset = [](std::int64_t value) {
        return ((value + kFloatArenaAlignmentElements - 1) / kFloatArenaAlignmentElements) *
               kFloatArenaAlignmentElements;
    };
    auto align_uint16_arena_offset = [](std::int64_t value) {
        return ((value + kUint16ArenaAlignmentElements - 1) / kUint16ArenaAlignmentElements) *
               kUint16ArenaAlignmentElements;
    };
    const bool combined_uint16_arena_enabled =
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_COMBINED_BF16_ARENA",
                              "NFN_NATIVE_GPT2_COMBINED_BF16_ARENA"}),
            true);
    auto allocate = [&](float** ptr, std::int64_t elements, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        if (*ptr != nullptr) {
            return;
        }
        if (elements < 0) {
            error = "negative float allocation requested for " + name;
            return;
        }
        if (elements == 0) {
            return;
        }
        for (const FloatArenaRequest& request : float_arena_requests) {
            if (request.ptr == ptr) {
                if (request.elements != elements && error.empty()) {
                    error = "conflicting duplicate float allocation request for " + name;
                }
                return;
            }
        }
        const std::int64_t offset = align_float_arena_offset(float_arena_allocated_elements);
        if (offset > std::numeric_limits<std::int64_t>::max() - elements) {
            error = "float arena allocation size overflow for " + name;
            return;
        }
        float_arena_requests.push_back(FloatArenaRequest{ptr, elements, offset, name});
        float_arena_requested_elements += elements;
        float_arena_allocated_elements = offset + elements;
    };
    auto materialize_float_arena = [&]() {
        if (!error.empty() || float_arena_requests.empty()) {
            return;
        }
        float_arena_allocated_elements = align_float_arena_offset(float_arena_allocated_elements);
        if (float_arena_allocated_elements >
            static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(float))) {
            error = "float arena allocation byte size overflow";
            return;
        }
        const int status = device_malloc(
            reinterpret_cast<void**>(&float_arena),
            sizeof(float) * static_cast<std::size_t>(float_arena_allocated_elements));
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc transformer_lm_float_arena");
            return;
        }
        float_ptrs.push_back(float_arena);
        float_arena_cuda_malloc_count = 1;
        for (const FloatArenaRequest& request : float_arena_requests) {
            *request.ptr = float_arena + request.offset;
        }
    };
    auto allocate_uint16 = [&](std::uint16_t** ptr, std::int64_t elements, const std::string& name) {
        if (!error.empty() || !combined_uint16_arena_enabled) {
            return;
        }
        if (*ptr != nullptr) {
            return;
        }
        if (elements < 0) {
            error = "negative uint16 allocation requested for " + name;
            return;
        }
        if (elements == 0) {
            return;
        }
        for (const Uint16ArenaRequest& request : uint16_arena_requests) {
            if (request.ptr == ptr) {
                if (request.elements != elements && error.empty()) {
                    error = "conflicting duplicate uint16 allocation request for " + name;
                }
                return;
            }
        }
        const std::int64_t offset = align_uint16_arena_offset(uint16_arena_allocated_elements);
        if (offset > std::numeric_limits<std::int64_t>::max() - elements) {
            error = "uint16 arena allocation size overflow for " + name;
            return;
        }
        uint16_arena_requests.push_back(Uint16ArenaRequest{ptr, elements, offset, name});
        uint16_arena_requested_elements += elements;
        uint16_arena_allocated_elements = offset + elements;
    };
    auto materialize_uint16_arena = [&]() {
        if (!error.empty() || !combined_uint16_arena_enabled || uint16_arena_requests.empty()) {
            return;
        }
        uint16_arena_allocated_elements = align_uint16_arena_offset(uint16_arena_allocated_elements);
        if (uint16_arena_allocated_elements >
            static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t))) {
            error = "uint16 arena allocation byte size overflow";
            return;
        }
        const int status = device_malloc(
            reinterpret_cast<void**>(&uint16_arena),
            sizeof(std::uint16_t) * static_cast<std::size_t>(uint16_arena_allocated_elements));
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc transformer_lm_uint16_arena");
            return;
        }
        uint16_ptrs.push_back(uint16_arena);
        uint16_arena_cuda_malloc_count = 1;
        for (const Uint16ArenaRequest& request : uint16_arena_requests) {
            *request.ptr = uint16_arena + request.offset;
        }
    };

    struct TransformerBlockParams {
        float* ln1_weight = nullptr;
        float* ln1_bias = nullptr;
        float* ln2_weight = nullptr;
        float* ln2_bias = nullptr;
        float* qkv_weight = nullptr;
        std::uint16_t* qkv_weight_bf16 = nullptr;
        float* qkv_bias = nullptr;
        float* attn_proj_weight = nullptr;
        std::uint16_t* attn_proj_weight_bf16 = nullptr;
        float* attn_proj_bias = nullptr;
        float* fc_weight = nullptr;
        std::uint16_t* fc_weight_bf16 = nullptr;
        float* fc_bias = nullptr;
        float* mlp_proj_weight = nullptr;
        std::uint16_t* mlp_proj_weight_bf16 = nullptr;
        float* mlp_proj_bias = nullptr;
        float* accum_grad_ln1_weight = nullptr;
        float* accum_grad_ln1_bias = nullptr;
        float* accum_grad_ln2_weight = nullptr;
        float* accum_grad_ln2_bias = nullptr;
        float* accum_grad_qkv_weight = nullptr;
        std::uint16_t* accum_grad_qkv_weight_bf16 = nullptr;
        float* accum_grad_qkv_bias = nullptr;
        float* accum_grad_attn_proj_weight = nullptr;
        float* accum_grad_attn_proj_bias = nullptr;
        float* accum_grad_fc_weight = nullptr;
        std::uint16_t* accum_grad_fc_weight_bf16 = nullptr;
        float* accum_grad_fc_bias = nullptr;
        float* accum_grad_mlp_proj_weight = nullptr;
        float* accum_grad_mlp_proj_bias = nullptr;
        float* ln1_weight_avg = nullptr;
        float* ln1_weight_avg_sq = nullptr;
        float* ln1_bias_avg = nullptr;
        float* ln1_bias_avg_sq = nullptr;
        float* ln2_weight_avg = nullptr;
        float* ln2_weight_avg_sq = nullptr;
        float* ln2_bias_avg = nullptr;
        float* ln2_bias_avg_sq = nullptr;
        float* qkv_avg = nullptr;
        float* qkv_avg_sq = nullptr;
        float* qkv_bias_avg = nullptr;
        float* qkv_bias_avg_sq = nullptr;
        float* attn_proj_avg = nullptr;
        float* attn_proj_avg_sq = nullptr;
        float* attn_proj_bias_avg = nullptr;
        float* attn_proj_bias_avg_sq = nullptr;
        float* fc_avg = nullptr;
        float* fc_avg_sq = nullptr;
        float* fc_bias_avg = nullptr;
        float* fc_bias_avg_sq = nullptr;
        float* mlp_proj_avg = nullptr;
        float* mlp_proj_avg_sq = nullptr;
        float* mlp_proj_bias_avg = nullptr;
        float* mlp_proj_bias_avg_sq = nullptr;
    };
    struct TransformerBlockActivations {
        float* ln1_out = nullptr;
        std::uint16_t* ln1_out_bf16 = nullptr;
        float* ln1_mean = nullptr;
        float* ln1_rstd = nullptr;
        float* qkv = nullptr;
        std::uint16_t* qkv_bf16 = nullptr;
        float* q = nullptr;
        float* k = nullptr;
        float* v = nullptr;
        float* q_heads = nullptr;
        float* k_heads = nullptr;
        float* v_heads = nullptr;
        std::uint16_t* packed_attn_out_bf16 = nullptr;
        float* attn_heads = nullptr;
        float* attn_out = nullptr;
        float* attn_proj = nullptr;
        std::uint16_t* proj_out_bf16 = nullptr;
        float* residual1 = nullptr;
        float* ln2_out = nullptr;
        float* ln2_mean = nullptr;
        float* ln2_rstd = nullptr;
        float* fc_out = nullptr;
        float* act = nullptr;
        float* mlp_out = nullptr;
        float* residual2 = nullptr;
    };
    struct StoredMlpActivations {
        std::uint16_t* ln2_out = nullptr;
        std::uint16_t* fc_out = nullptr;
        std::uint16_t* act = nullptr;
        float* ln2_mean = nullptr;
        float* ln2_rstd = nullptr;
    };
    struct StoredAttentionActivations {
        std::uint16_t* q = nullptr;
        std::uint16_t* k = nullptr;
        std::uint16_t* v = nullptr;
        std::uint16_t* o = nullptr;
        float* lse = nullptr;
    };
    struct StoredPackedAttentionActivations {
        float* ln1_mean = nullptr;
        float* ln1_rstd = nullptr;
        std::uint16_t* qkv = nullptr;
        std::uint16_t* o = nullptr;
        float* lse = nullptr;
    };
    std::vector<TransformerBlockParams> blocks(static_cast<std::size_t>(trained_layers));
    std::vector<TransformerBlockActivations> block_tapes(static_cast<std::size_t>(kActivationTapeCount));
    std::vector<float*> block_outputs(static_cast<std::size_t>(persistent_block_output_count), nullptr);
    constexpr std::int64_t kBlockWeightBf16ElementsPerBlock =
        kQkvWeightElements + kAttnProjWeightElements + kFcWeightElements + kMlpProjWeightElements;
    constexpr std::int64_t kBlockDweightBf16StagingElementsPerBlock =
        kQkvWeightElements + kFcWeightElements;
    const std::int64_t stored_mlp_activation_block_count =
        store_mlp_activations_enabled && trained_layers > 0
            ? std::min<std::int64_t>(
                  trained_layers,
                  env_nonnegative_i64_or({"NFN_NATIVE_GPT_STORE_MLP_BLOCKS",
                                          "NFN_NATIVE_GPT2_STORE_MLP_BLOCKS"},
                                         kDefaultStoredMlpBlocks))
            : 0;
    const std::int64_t stored_mlp_activation_elements_per_block = activation_elements + hidden_elements * 2;
    const std::int64_t stored_mlp_activation_elements =
        stored_mlp_activation_block_count * stored_mlp_activation_elements_per_block;
    const bool lazy_validation_mlp_float_scratch_enabled =
        stored_mlp_activation_block_count >= trained_layers &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_LAZY_VALIDATION_MLP_FLOAT_SCRATCH",
                              "NFN_NATIVE_GPT2_LAZY_VALIDATION_MLP_FLOAT_SCRATCH"}),
            true);
    std::vector<StoredMlpActivations> stored_mlp_activations(
        static_cast<std::size_t>(stored_mlp_activation_block_count));
    std::uint16_t* stored_mlp_activation_arena = nullptr;
    float* stored_mlp_norm_stats_arena = nullptr;
    std::int64_t stored_mlp_activation_arena_elements = 0;
    std::int64_t stored_mlp_activation_arena_bytes = 0;
    std::int64_t stored_mlp_norm_stats_elements = 0;
    std::int64_t stored_mlp_norm_stats_bytes = 0;
    std::int64_t stored_mlp_activation_store_kernel_launches = 0;
    std::int64_t stored_mlp_ln2_bf16_fused_store_kernel_launches = 0;
    std::int64_t stored_mlp_ln2_bf16_float_store_elided_count = 0;
    std::int64_t stored_mlp_ln2_bf16_float_store_elided_elements = 0;
    std::int64_t stored_mlp_activation_restore_kernel_launches = 0;
    const std::int64_t stored_residual1_block_count =
        store_residual1_activations_enabled && trained_layers > 1 ? trained_layers - 1 : 0;
    const std::int64_t stored_residual1_activation_elements =
        stored_residual1_block_count * activation_elements;
    std::vector<std::uint16_t*> stored_residual1_activations(
        static_cast<std::size_t>(stored_residual1_block_count), nullptr);
    std::uint16_t* stored_residual1_activation_arena = nullptr;
    std::int64_t stored_residual1_activation_arena_elements = 0;
    std::int64_t stored_residual1_activation_arena_bytes = 0;
    std::int64_t stored_residual1_activation_store_kernel_launches = 0;
    std::int64_t stored_residual1_activation_restore_kernel_launches = 0;
    const std::int64_t stored_attention_block_count =
        store_attention_activations_enabled && !packed_qkv_attention_enabled && trained_layers > 0
            ? trained_layers - 1
            : 0;
    const std::int64_t attention_lse_elements = batch_size * kHeads * seq_len;
    const std::int64_t stored_attention_bf16_elements_per_block = activation_elements * 4;
    const std::int64_t stored_attention_lse_elements_per_block = attention_lse_elements;
    const std::int64_t stored_attention_bf16_elements =
        stored_attention_block_count * stored_attention_bf16_elements_per_block;
    const std::int64_t stored_attention_lse_elements =
        stored_attention_block_count * stored_attention_lse_elements_per_block;
    std::vector<StoredAttentionActivations> stored_attention_activations(
        static_cast<std::size_t>(stored_attention_block_count));
    std::uint16_t* stored_attention_bf16_arena = nullptr;
    float* stored_attention_lse_arena = nullptr;
    std::int64_t stored_attention_bf16_arena_elements = 0;
    std::int64_t stored_attention_bf16_arena_bytes = 0;
    std::int64_t stored_attention_lse_arena_elements = 0;
    std::int64_t stored_attention_lse_arena_bytes = 0;
    std::int64_t stored_attention_store_kernel_launches = 0;
    std::int64_t stored_attention_restore_kernel_launches = 0;
    std::int64_t stored_attention_backward_kernel_launches = 0;
    const bool store_packed_attention_lse_enabled =
        store_packed_attention_activations_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LSE",
                              "NFN_NATIVE_GPT2_STORE_PACKED_ATTENTION_LSE"}),
            true);
    const bool store_packed_attention_ln1_stats_enabled =
        store_packed_attention_activations_enabled &&
        ln1_bf16_qkv_forward_enabled &&
        bf16_qkv_dweight_enabled &&
        env_flag_enabled_or_default(
            env_or_empty_any({"NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_LN1_STATS",
                              "NFN_NATIVE_GPT2_STORE_PACKED_ATTENTION_LN1_STATS"}),
            true);
    const std::int64_t stored_packed_attention_block_count =
        store_packed_attention_activations_enabled && trained_layers > 0
            ? std::min<std::int64_t>(
                  trained_layers,
                  env_nonnegative_i64_or({"NFN_NATIVE_GPT_STORE_PACKED_ATTENTION_BLOCKS",
                                          "NFN_NATIVE_GPT2_STORE_PACKED_ATTENTION_BLOCKS"},
                                         kDefaultStoredPackedAttentionBlocks))
            : 0;
    const std::int64_t stored_packed_attention_bf16_elements_per_block =
        qkv_activation_elements + activation_elements;
    const std::int64_t stored_packed_attention_bf16_elements =
        stored_packed_attention_block_count * stored_packed_attention_bf16_elements_per_block;
    const std::int64_t stored_packed_attention_lse_elements_per_block =
        batch_size * kHeads * seq_len;
    const std::int64_t stored_packed_attention_lse_elements =
        store_packed_attention_lse_enabled
            ? stored_packed_attention_block_count * stored_packed_attention_lse_elements_per_block
            : 0;
    std::vector<StoredPackedAttentionActivations> stored_packed_attention_activations(
        static_cast<std::size_t>(stored_packed_attention_block_count));
    std::uint16_t* stored_packed_attention_bf16_arena = nullptr;
    std::int64_t stored_packed_attention_bf16_arena_elements = 0;
    std::int64_t stored_packed_attention_bf16_arena_bytes = 0;
    const std::int64_t stored_packed_attention_ln1_stats_block_count =
        store_packed_attention_ln1_stats_enabled && stored_packed_attention_block_count > 0
            ? std::min<std::int64_t>(stored_packed_attention_block_count, std::max<std::int64_t>(trained_layers - 1, 0))
            : 0;
    const std::int64_t stored_packed_attention_ln1_stats_elements =
        stored_packed_attention_ln1_stats_block_count * rows * 2;
    float* stored_packed_attention_ln1_stats_arena = nullptr;
    std::int64_t stored_packed_attention_ln1_stats_arena_elements = 0;
    std::int64_t stored_packed_attention_ln1_stats_arena_bytes = 0;
    float* stored_packed_attention_lse_arena = nullptr;
    std::int64_t stored_packed_attention_lse_arena_elements = 0;
    std::int64_t stored_packed_attention_lse_arena_bytes = 0;
    std::int64_t stored_packed_attention_store_blocks = 0;
    std::int64_t stored_packed_attention_restore_blocks = 0;
    std::int64_t stored_packed_attention_backward_kernel_launches = 0;
    std::int64_t direct_block_output_write_count = 0;
    std::uint16_t* token_weight_bf16 = nullptr;
    std::uint16_t* lm_head_bf16_logits = nullptr;
    std::int64_t lm_head_bf16_logit_elements = 0;
    std::int64_t lm_head_bf16_logit_bytes = 0;
    std::uint16_t* lm_head_bf16_hidden = nullptr;
    std::int64_t lm_head_bf16_hidden_elements = 0;
    std::int64_t lm_head_bf16_hidden_bytes = 0;
    std::uint16_t* mlp_forward_act_bf16 = nullptr;
    std::int64_t mlp_forward_act_bf16_elements = 0;
    std::int64_t mlp_forward_act_bf16_bytes = 0;
    std::int64_t lazy_validation_mlp_float_scratch_elements = 0;
    std::int64_t lazy_validation_mlp_float_scratch_bytes = 0;
    std::int64_t lazy_validation_mlp_float_scratch_cuda_malloc_count = 0;
    std::uint16_t* projection_bf16_scratch = nullptr;
    std::int64_t projection_bf16_scratch_elements = 0;
    std::int64_t projection_bf16_scratch_bytes = 0;
    std::uint16_t* packed_qkv_attention_bf16_arena = nullptr;
    std::int64_t packed_qkv_attention_bf16_elements = 0;
    std::int64_t packed_qkv_attention_bf16_bytes = 0;
    std::uint16_t* attention_grad_out_bf16 = nullptr;
    std::int64_t attention_grad_out_bf16_elements = 0;
    std::int64_t attention_grad_out_bf16_bytes = 0;

    float *token_weight = nullptr, *position_weight = nullptr, *residual_scale = nullptr;
    float *lnf_weight = nullptr, *lnf_bias = nullptr;
    float *accum_grad_token_weight = nullptr, *accum_grad_position_weight = nullptr;
    float *accum_grad_lnf_weight = nullptr, *accum_grad_lnf_bias = nullptr;
    float *token_avg = nullptr, *token_avg_sq = nullptr, *position_avg = nullptr, *position_avg_sq = nullptr;
    float *lnf_weight_avg = nullptr, *lnf_weight_avg_sq = nullptr, *lnf_bias_avg = nullptr, *lnf_bias_avg_sq = nullptr;
    float *token_out = nullptr, *position_out = nullptr, *x = nullptr;
    float* lnf_input = nullptr;
    float *lnf_mean = nullptr, *lnf_rstd = nullptr;
    float *lnf_out = nullptr, *logits = nullptr, *loss_partials = nullptr;
    float *loss_reduce_a = nullptr, *loss_reduce_b = nullptr, *loss_total = nullptr;
    float *row_max = nullptr, *row_denom = nullptr;
    float *grad_lnf = nullptr, *grad_residual2 = nullptr;
    float *grad_fc_out = nullptr, *grad_ln2 = nullptr, *grad_residual1_from_mlp = nullptr, *grad_residual1 = nullptr;
    float *grad_attn_out = nullptr, *grad_qkv = nullptr;
    float *grad_ln1 = nullptr, *grad_x_from_attn = nullptr, *grad_x = nullptr;
    float *grad_sumsq_partials = nullptr, *grad_clip_scale = nullptr;
    std::int64_t *token_ids = nullptr, *targets = nullptr;
    std::uint16_t *token_ids_u16 = nullptr, *targets_u16 = nullptr;
    std::uint16_t *token_ids_pinned = nullptr, *targets_pinned = nullptr;
    std::uint16_t* checkpoint_bf16_device = nullptr;
    std::uint16_t* block_weight_bf16_arena = nullptr;
    std::int64_t block_weight_bf16_arena_elements = 0;
    std::int64_t block_weight_bf16_arena_bytes = 0;
    std::uint16_t* block_dweight_bf16_staging_arena = nullptr;
    std::int64_t block_dweight_bf16_staging_elements = 0;
    std::int64_t block_dweight_bf16_staging_bytes = 0;
    std::int64_t block_dweight_bf16_staging_convert_kernel_launches = 0;
    std::int64_t block_dweight_bf16_staging_zero_count = 0;
    std::int64_t block_weight_bf16_refresh_count = 0;
    std::int64_t block_weight_bf16_fused_adamw_refresh_count = 0;
    std::int64_t token_weight_bf16_refresh_count = 0;
    const float** block_weight_bf16_sources = nullptr;
    std::int64_t* block_weight_bf16_elements = nullptr;
    std::int64_t* block_weight_bf16_offsets = nullptr;
    std::int64_t block_weight_bf16_descriptor_count = 0;
    std::int64_t block_weight_bf16_max_elements = 0;
    auto allocate_token_arenas = [&]() {
        if (!error.empty()) {
            return;
        }
        if (rows < 0 || rows > std::numeric_limits<std::int64_t>::max() / 2) {
            error = "token arena row count overflow";
            return;
        }
        token_i64_arena_elements = rows * 2;
        token_u16_device_arena_elements = rows * 2;
        token_u16_pinned_arena_elements = rows * 2;
        if (token_i64_arena_elements >
            static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(std::int64_t))) {
            error = "token int64 arena byte size overflow";
            return;
        }
        if (token_u16_device_arena_elements >
            static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t))) {
            error = "token uint16 arena byte size overflow";
            return;
        }

        constexpr std::size_t kTokenDeviceArenaAlignmentBytes = 16;
        auto align_token_device_offset = [](std::size_t value) {
            return ((value + kTokenDeviceArenaAlignmentBytes - 1) / kTokenDeviceArenaAlignmentBytes) *
                   kTokenDeviceArenaAlignmentBytes;
        };
        const std::size_t token_i64_bytes =
            sizeof(std::int64_t) * static_cast<std::size_t>(token_i64_arena_elements);
        const std::size_t token_u16_device_bytes =
            sizeof(std::uint16_t) * static_cast<std::size_t>(token_u16_device_arena_elements);
        const std::size_t token_i64_offset = 0;
        const std::size_t token_u16_device_offset = align_token_device_offset(token_i64_bytes);
        if (token_u16_device_offset > std::numeric_limits<std::size_t>::max() - token_u16_device_bytes) {
            error = "token device arena allocation byte size overflow";
            return;
        }
        token_device_arena_requested_bytes = token_i64_bytes + token_u16_device_bytes;
        token_device_arena_bytes = align_token_device_offset(token_u16_device_offset + token_u16_device_bytes);
        int status = device_malloc(&token_device_arena, token_device_arena_bytes);
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc transformer_lm_token_device_arena");
            return;
        }
        token_device_ptrs.push_back(token_device_arena);
        token_device_arena_cuda_malloc_count = 1;
        token_device_arena_suballocation_count = 2;
        auto* token_device_base = static_cast<unsigned char*>(token_device_arena);
        token_i64_arena = reinterpret_cast<std::int64_t*>(token_device_base + token_i64_offset);
        token_u16_device_arena = reinterpret_cast<std::uint16_t*>(token_device_base + token_u16_device_offset);
        token_ids = token_i64_arena;
        targets = token_i64_arena + rows;
        token_ids_u16 = token_u16_device_arena;
        targets_u16 = token_u16_device_arena + rows;

        status = cuda_host_alloc(
            reinterpret_cast<void**>(&token_u16_pinned_arena),
            sizeof(std::uint16_t) * static_cast<std::size_t>(token_u16_pinned_arena_elements),
            0U);
        if (status != 0) {
            error = cuda_error(status, "cudaHostAlloc transformer_lm_token_u16_pinned_arena");
            return;
        }
        pinned_uint16_ptrs.push_back(token_u16_pinned_arena);
        token_u16_pinned_arena_cuda_host_alloc_count = 1;
        token_ids_pinned = token_u16_pinned_arena;
        targets_pinned = token_u16_pinned_arena + rows;
    };
    auto allocate_stored_mlp_activation_arena = [&]() {
        if (!error.empty() || stored_mlp_activation_elements <= 0) {
            return;
        }
        if (stored_mlp_activation_elements >
            static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t))) {
            error = "stored MLP activation bf16 arena byte size overflow";
            return;
        }
        const std::size_t bytes =
            sizeof(std::uint16_t) * static_cast<std::size_t>(stored_mlp_activation_elements);
        if (stored_mlp_activation_arena == nullptr) {
            void* raw = nullptr;
            const int status = device_malloc(&raw, bytes);
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc stored_mlp_activation_bf16_arena");
                return;
            }
            stored_mlp_activation_arena = static_cast<std::uint16_t*>(raw);
            uint16_ptrs.push_back(stored_mlp_activation_arena);
        }
        stored_mlp_activation_arena_elements = stored_mlp_activation_elements;
        stored_mlp_activation_arena_bytes = static_cast<std::int64_t>(bytes);
        const std::int64_t norm_stats_elements = stored_mlp_activation_block_count * rows * 2;
        if (norm_stats_elements <= 0 ||
            norm_stats_elements >
                static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(float))) {
            error = "stored MLP LayerNorm stats arena byte size overflow";
            return;
        }
        void* raw_stats = nullptr;
        const std::size_t stats_bytes = sizeof(float) * static_cast<std::size_t>(norm_stats_elements);
        const int stats_status = device_malloc(&raw_stats, stats_bytes);
        if (stats_status != 0) {
            error = cuda_error(stats_status, "cudaMalloc stored_mlp_norm_stats_arena");
            return;
        }
        stored_mlp_norm_stats_arena = static_cast<float*>(raw_stats);
        float_ptrs.push_back(stored_mlp_norm_stats_arena);
        stored_mlp_norm_stats_elements = norm_stats_elements;
        stored_mlp_norm_stats_bytes = static_cast<std::int64_t>(stats_bytes);
        for (std::int64_t i = 0; i < stored_mlp_activation_block_count; ++i) {
            const std::int64_t base = i * stored_mlp_activation_elements_per_block;
            const std::int64_t stats_base = i * rows * 2;
            StoredMlpActivations& stored = stored_mlp_activations[static_cast<std::size_t>(i)];
            stored.ln2_out = stored_mlp_activation_arena + base;
            stored.fc_out = stored_mlp_activation_arena + base + activation_elements;
            stored.act = stored_mlp_activation_arena + base + activation_elements + hidden_elements;
            stored.ln2_mean = stored_mlp_norm_stats_arena + stats_base;
            stored.ln2_rstd = stored_mlp_norm_stats_arena + stats_base + rows;
        }
    };
    auto allocate_stored_residual1_activation_arena = [&]() {
        if (!error.empty() || stored_residual1_activation_elements <= 0) {
            return;
        }
        if (stored_residual1_activation_elements >
            static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t))) {
            error = "stored residual1 bf16 arena byte size overflow";
            return;
        }
        const std::size_t bytes =
            sizeof(std::uint16_t) * static_cast<std::size_t>(stored_residual1_activation_elements);
        if (stored_residual1_activation_arena == nullptr) {
            void* raw = nullptr;
            const int status = device_malloc(&raw, bytes);
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc stored_residual1_bf16_arena");
                return;
            }
            stored_residual1_activation_arena = static_cast<std::uint16_t*>(raw);
            uint16_ptrs.push_back(stored_residual1_activation_arena);
        }
        stored_residual1_activation_arena_elements = stored_residual1_activation_elements;
        stored_residual1_activation_arena_bytes = static_cast<std::int64_t>(bytes);
        for (std::int64_t i = 0; i < stored_residual1_block_count; ++i) {
            stored_residual1_activations[static_cast<std::size_t>(i)] =
                stored_residual1_activation_arena + i * activation_elements;
        }
    };
    auto allocate_stored_attention_activation_arenas = [&]() {
        if (!error.empty() || stored_attention_block_count <= 0) {
            return;
        }
        if (stored_attention_bf16_elements <= 0 ||
            stored_attention_bf16_elements >
                static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) ||
            stored_attention_lse_elements <= 0 ||
            stored_attention_lse_elements >
                static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(float))) {
            error = "stored attention activation arena byte size overflow";
            return;
        }
        const std::size_t bf16_bytes =
            sizeof(std::uint16_t) * static_cast<std::size_t>(stored_attention_bf16_elements);
        int status = 0;
        if (stored_attention_bf16_arena == nullptr) {
            void* raw_bf16 = nullptr;
            status = device_malloc(&raw_bf16, bf16_bytes);
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc stored_attention_bf16_arena");
                return;
            }
            stored_attention_bf16_arena = static_cast<std::uint16_t*>(raw_bf16);
            uint16_ptrs.push_back(stored_attention_bf16_arena);
        }
        stored_attention_bf16_arena_elements = stored_attention_bf16_elements;
        stored_attention_bf16_arena_bytes = static_cast<std::int64_t>(bf16_bytes);

        void* raw_lse = nullptr;
        const std::size_t lse_bytes = sizeof(float) * static_cast<std::size_t>(stored_attention_lse_elements);
        status = device_malloc(&raw_lse, lse_bytes);
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc stored_attention_lse_arena");
            return;
        }
        stored_attention_lse_arena = static_cast<float*>(raw_lse);
        float_ptrs.push_back(stored_attention_lse_arena);
        stored_attention_lse_arena_elements = stored_attention_lse_elements;
        stored_attention_lse_arena_bytes = static_cast<std::int64_t>(lse_bytes);

        for (std::int64_t i = 0; i < stored_attention_block_count; ++i) {
            const std::int64_t bf16_base = i * stored_attention_bf16_elements_per_block;
            const std::int64_t lse_base = i * stored_attention_lse_elements_per_block;
            StoredAttentionActivations& stored = stored_attention_activations[static_cast<std::size_t>(i)];
            stored.q = stored_attention_bf16_arena + bf16_base;
            stored.k = stored_attention_bf16_arena + bf16_base + activation_elements;
            stored.v = stored_attention_bf16_arena + bf16_base + activation_elements * 2;
            stored.o = stored_attention_bf16_arena + bf16_base + activation_elements * 3;
            stored.lse = stored_attention_lse_arena + lse_base;
        }
    };
    auto allocate_stored_packed_attention_activation_arena = [&]() {
        if (!error.empty() || stored_packed_attention_block_count <= 0) {
            return;
        }
        if (stored_packed_attention_bf16_elements <= 0 ||
            stored_packed_attention_bf16_elements >
                static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t))) {
            error = "stored packed attention activation arena byte size overflow";
            return;
        }
        const std::size_t bytes =
            sizeof(std::uint16_t) * static_cast<std::size_t>(stored_packed_attention_bf16_elements);
        if (stored_packed_attention_bf16_arena == nullptr) {
            void* raw = nullptr;
            const int status = device_malloc(&raw, bytes);
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc stored_packed_attention_bf16_arena");
                return;
            }
            stored_packed_attention_bf16_arena = static_cast<std::uint16_t*>(raw);
            uint16_ptrs.push_back(stored_packed_attention_bf16_arena);
        }
        stored_packed_attention_bf16_arena_elements = stored_packed_attention_bf16_elements;
        stored_packed_attention_bf16_arena_bytes = static_cast<std::int64_t>(bytes);

        for (std::int64_t i = 0; i < stored_packed_attention_block_count; ++i) {
            const std::int64_t base = i * stored_packed_attention_bf16_elements_per_block;
            StoredPackedAttentionActivations& stored =
                stored_packed_attention_activations[static_cast<std::size_t>(i)];
            stored.qkv = stored_packed_attention_bf16_arena + base;
            stored.o = stored_packed_attention_bf16_arena + base + qkv_activation_elements;
            if (stored_packed_attention_lse_arena != nullptr) {
                stored.lse = stored_packed_attention_lse_arena +
                             i * stored_packed_attention_lse_elements_per_block;
            }
        }
    };
    auto allocate_stored_packed_attention_ln1_stats_arena = [&]() {
        if (!error.empty() || stored_packed_attention_ln1_stats_block_count <= 0) {
            return;
        }
        if (stored_packed_attention_ln1_stats_elements <= 0 ||
            stored_packed_attention_ln1_stats_elements >
                static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(float))) {
            error = "stored packed attention LN1 stats arena byte size overflow";
            return;
        }
        const std::size_t stats_bytes =
            sizeof(float) * static_cast<std::size_t>(stored_packed_attention_ln1_stats_elements);
        if (stored_packed_attention_ln1_stats_arena == nullptr) {
            void* raw = nullptr;
            const int status = device_malloc(&raw, stats_bytes);
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc stored_packed_attention_ln1_stats_arena");
                return;
            }
            stored_packed_attention_ln1_stats_arena = static_cast<float*>(raw);
            float_ptrs.push_back(stored_packed_attention_ln1_stats_arena);
        }
        stored_packed_attention_ln1_stats_arena_elements = stored_packed_attention_ln1_stats_elements;
        stored_packed_attention_ln1_stats_arena_bytes = static_cast<std::int64_t>(stats_bytes);

        for (std::int64_t i = 0; i < stored_packed_attention_ln1_stats_block_count; ++i) {
            StoredPackedAttentionActivations& stored =
                stored_packed_attention_activations[static_cast<std::size_t>(i)];
            const std::int64_t stats_base = i * rows * 2;
            stored.ln1_mean = stored_packed_attention_ln1_stats_arena + stats_base;
            stored.ln1_rstd = stored_packed_attention_ln1_stats_arena + stats_base + rows;
        }
    };
    auto allocate_lm_head_bf16_logits = [&]() {
        if (!error.empty() || !lm_head_bf16_logits_enabled) {
            return;
        }
        if (logit_elements <= 0 ||
            logit_elements > static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t))) {
            error = "LM-head bf16 logit allocation byte size overflow";
            return;
        }
        const std::size_t bytes = sizeof(std::uint16_t) * static_cast<std::size_t>(logit_elements);
        if (lm_head_bf16_logits == nullptr) {
            void* raw = nullptr;
            const int status = device_malloc(&raw, bytes);
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc lm_head_bf16_logits");
                return;
            }
            lm_head_bf16_logits = static_cast<std::uint16_t*>(raw);
            uint16_ptrs.push_back(lm_head_bf16_logits);
        }
        lm_head_bf16_logit_elements = logit_elements;
        lm_head_bf16_logit_bytes = static_cast<std::int64_t>(bytes);
    };
    auto allocate_lm_head_bf16_hidden = [&]() {
        if (!error.empty() || (!lm_head_bf16_dweight_enabled && !lm_head_prepack_bf16_hidden_enabled)) {
            return;
        }
        const std::int64_t elements =
            lm_head_prepack_bf16_hidden_enabled ? activation_elements : (lm_head_chunk_rows * kDim);
        if (elements <= 0 ||
            elements > static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t))) {
            error = "LM-head bf16 hidden allocation byte size overflow";
            return;
        }
        const std::size_t bytes = sizeof(std::uint16_t) * static_cast<std::size_t>(elements);
        if (lm_head_bf16_hidden == nullptr) {
            void* raw = nullptr;
            const int status = device_malloc(&raw, bytes);
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc lm_head_bf16_hidden");
                return;
            }
            lm_head_bf16_hidden = static_cast<std::uint16_t*>(raw);
            uint16_ptrs.push_back(lm_head_bf16_hidden);
        }
        lm_head_bf16_hidden_elements = elements;
        lm_head_bf16_hidden_bytes = static_cast<std::int64_t>(bytes);
    };
    auto allocate_mlp_forward_act_bf16 = [&]() {
        if (!error.empty()) {
            return;
        }
        if (hidden_elements <= 0 ||
            hidden_elements > static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t))) {
            error = "MLP forward bf16 activation scratch allocation byte size overflow";
            return;
        }
        const std::size_t bytes = sizeof(std::uint16_t) * static_cast<std::size_t>(hidden_elements);
        if (mlp_forward_act_bf16 == nullptr) {
            void* raw = nullptr;
            const int status = device_malloc(&raw, bytes);
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc mlp_forward_act_bf16");
                return;
            }
            mlp_forward_act_bf16 = static_cast<std::uint16_t*>(raw);
            uint16_ptrs.push_back(mlp_forward_act_bf16);
        }
        mlp_forward_act_bf16_elements = hidden_elements;
        mlp_forward_act_bf16_bytes = static_cast<std::int64_t>(bytes);
    };
    auto allocate_projection_bf16_scratch = [&]() {
        if (!error.empty() || !bf16_projection_residual_enabled) {
            return;
        }
        if (activation_elements <= 0 ||
            kActivationTapeCount >
                std::numeric_limits<std::int64_t>::max() / activation_elements ||
            activation_elements * kActivationTapeCount >
                static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t))) {
            error = "projection bf16 scratch allocation byte size overflow";
            return;
        }
        const std::int64_t total_elements = activation_elements * kActivationTapeCount;
        const std::size_t bytes = sizeof(std::uint16_t) * static_cast<std::size_t>(total_elements);
        if (projection_bf16_scratch == nullptr) {
            void* raw = nullptr;
            const int status = device_malloc(&raw, bytes);
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc projection_bf16_scratch");
                return;
            }
            projection_bf16_scratch = static_cast<std::uint16_t*>(raw);
            uint16_ptrs.push_back(projection_bf16_scratch);
        }
        projection_bf16_scratch_elements = total_elements;
        projection_bf16_scratch_bytes = static_cast<std::int64_t>(bytes);
        for (std::size_t i = 0; i < block_tapes.size(); ++i) {
            block_tapes[i].proj_out_bf16 =
                projection_bf16_scratch + static_cast<std::int64_t>(i) * activation_elements;
        }
    };
    auto allocate_packed_qkv_attention_scratch = [&]() {
        if (!error.empty() || !packed_qkv_attention_enabled) {
            return;
        }
        const std::int64_t elements_per_tape = qkv_activation_elements + activation_elements * 2;
        if (elements_per_tape <= 0 ||
            elements_per_tape >
                static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) ||
            kActivationTapeCount >
                std::numeric_limits<std::int64_t>::max() / elements_per_tape) {
            error = "packed QKV attention bf16 scratch allocation byte size overflow";
            return;
        }
        const std::int64_t total_elements = elements_per_tape * kActivationTapeCount;
        const std::size_t bytes = sizeof(std::uint16_t) * static_cast<std::size_t>(total_elements);
        if (packed_qkv_attention_bf16_arena == nullptr) {
            void* raw = nullptr;
            const int status = device_malloc(&raw, bytes);
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc packed_qkv_attention_bf16");
                return;
            }
            packed_qkv_attention_bf16_arena = static_cast<std::uint16_t*>(raw);
            uint16_ptrs.push_back(packed_qkv_attention_bf16_arena);
        }
        std::uint16_t* base = packed_qkv_attention_bf16_arena;
        packed_qkv_attention_bf16_elements = total_elements;
        packed_qkv_attention_bf16_bytes = static_cast<std::int64_t>(bytes);
        for (std::size_t i = 0; i < block_tapes.size(); ++i) {
            const std::int64_t offset = static_cast<std::int64_t>(i) * elements_per_tape;
            block_tapes[i].ln1_out_bf16 = base + offset;
            block_tapes[i].qkv_bf16 = base + offset + activation_elements;
            block_tapes[i].packed_attn_out_bf16 = base + offset + activation_elements + qkv_activation_elements;
        }
    };
    auto allocate_attention_grad_out_bf16 = [&]() {
        if (!error.empty() || !bf16_attention_grad_out_handoff_enabled) {
            return;
        }
        if (activation_elements <= 0 ||
            activation_elements >
                static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t))) {
            error = "attention grad-out bf16 scratch allocation byte size overflow";
            return;
        }
        const std::size_t bytes = sizeof(std::uint16_t) * static_cast<std::size_t>(activation_elements);
        if (attention_grad_out_bf16 == nullptr) {
            void* raw = nullptr;
            const int status = device_malloc(&raw, bytes);
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc attention_grad_out_bf16");
                return;
            }
            attention_grad_out_bf16 = static_cast<std::uint16_t*>(raw);
            uint16_ptrs.push_back(attention_grad_out_bf16);
        }
        attention_grad_out_bf16_elements = activation_elements;
        attention_grad_out_bf16_bytes = static_cast<std::int64_t>(bytes);
    };
    auto allocate_block_weight_bf16_arena = [&]() {
        if (!error.empty() || trained_layers <= 0) {
            return;
        }
        if (kBlockWeightBf16ElementsPerBlock <= 0 ||
            trained_layers > std::numeric_limits<std::int64_t>::max() / kBlockWeightBf16ElementsPerBlock ||
            kBlockWeightBf16ElementsPerBlock * trained_layers >
                static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t))) {
            error = "block BF16 weight arena byte size overflow";
            return;
        }
        const std::int64_t total_elements = kBlockWeightBf16ElementsPerBlock * trained_layers;
        const std::size_t bytes = sizeof(std::uint16_t) * static_cast<std::size_t>(total_elements);
        if (block_weight_bf16_arena == nullptr) {
            void* raw = nullptr;
            const int status = device_malloc(&raw, bytes);
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc block_weight_bf16_arena");
                return;
            }
            block_weight_bf16_arena = static_cast<std::uint16_t*>(raw);
            uint16_ptrs.push_back(block_weight_bf16_arena);
        }
        block_weight_bf16_arena_elements = total_elements;
        block_weight_bf16_arena_bytes = static_cast<std::int64_t>(bytes);
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            std::uint16_t* base = block_weight_bf16_arena +
                                  static_cast<std::int64_t>(i) * kBlockWeightBf16ElementsPerBlock;
            TransformerBlockParams& block = blocks[i];
            block.qkv_weight_bf16 = base;
            base += kQkvWeightElements;
            block.attn_proj_weight_bf16 = base;
            base += kAttnProjWeightElements;
            block.fc_weight_bf16 = base;
            base += kFcWeightElements;
            block.mlp_proj_weight_bf16 = base;
        }
    };
    auto allocate_block_dweight_bf16_staging_arena = [&]() {
        if (!error.empty() || trained_layers <= 0 || !bf16_block_dweight_staging_enabled) {
            return;
        }
        if (kBlockDweightBf16StagingElementsPerBlock <= 0 ||
            trained_layers > std::numeric_limits<std::int64_t>::max() / kBlockDweightBf16StagingElementsPerBlock ||
            kBlockDweightBf16StagingElementsPerBlock * trained_layers >
                static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t))) {
            error = "block BF16 dWeight staging arena byte size overflow";
            return;
        }
        const std::int64_t total_elements = kBlockDweightBf16StagingElementsPerBlock * trained_layers;
        const std::size_t bytes = sizeof(std::uint16_t) * static_cast<std::size_t>(total_elements);
        if (block_dweight_bf16_staging_arena == nullptr) {
            void* raw = nullptr;
            const int status = device_malloc(&raw, bytes);
            if (status != 0) {
                error = cuda_error(status, "cudaMalloc block_dweight_bf16_staging_arena");
                return;
            }
            block_dweight_bf16_staging_arena = static_cast<std::uint16_t*>(raw);
            uint16_ptrs.push_back(block_dweight_bf16_staging_arena);
        }
        block_dweight_bf16_staging_elements = total_elements;
        block_dweight_bf16_staging_bytes = static_cast<std::int64_t>(bytes);
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            std::uint16_t* base = block_dweight_bf16_staging_arena +
                                  static_cast<std::int64_t>(i) * kBlockDweightBf16StagingElementsPerBlock;
            TransformerBlockParams& block = blocks[i];
            block.accum_grad_qkv_weight_bf16 = base;
            base += kQkvWeightElements;
            block.accum_grad_fc_weight_bf16 = base;
        }
    };

    auto visit_block_parameter_ptrs = [&](auto&& visit) {
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            TransformerBlockParams& block = blocks[i];
            const std::string prefix = "block" + std::to_string(i);
            visit(&block.ln1_weight, kDim, prefix + ".ln1.weight");
            visit(&block.ln1_bias, kDim, prefix + ".ln1.bias");
            visit(&block.ln2_weight, kDim, prefix + ".ln2.weight");
            visit(&block.ln2_bias, kDim, prefix + ".ln2.bias");
            visit(&block.qkv_weight, kQkvWeightElements, prefix + ".attn.qkv.weight");
            visit(&block.qkv_bias, kQkvDim, prefix + ".attn.qkv.bias");
            visit(&block.attn_proj_weight, kAttnProjWeightElements, prefix + ".attn.proj.weight");
            visit(&block.attn_proj_bias, kDim, prefix + ".attn.proj.bias");
            visit(&block.fc_weight, kFcWeightElements, prefix + ".mlp.fc.weight");
            visit(&block.fc_bias, kHidden, prefix + ".mlp.fc.bias");
            visit(&block.mlp_proj_weight, kMlpProjWeightElements, prefix + ".mlp.proj.weight");
            visit(&block.mlp_proj_bias, kDim, prefix + ".mlp.proj.bias");
        }
    };
    auto visit_block_gradient_ptrs = [&](auto&& visit) {
        (void)visit;
    };
    auto visit_block_accum_gradient_ptrs = [&](auto&& visit) {
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            TransformerBlockParams& block = blocks[i];
            const std::string prefix = "block" + std::to_string(i);
            visit(&block.accum_grad_ln1_weight, kDim, prefix + ".ln1.weight.accum_grad");
            visit(&block.accum_grad_ln1_bias, kDim, prefix + ".ln1.bias.accum_grad");
            visit(&block.accum_grad_ln2_weight, kDim, prefix + ".ln2.weight.accum_grad");
            visit(&block.accum_grad_ln2_bias, kDim, prefix + ".ln2.bias.accum_grad");
            visit(&block.accum_grad_qkv_weight, kQkvWeightElements, prefix + ".attn.qkv.weight.accum_grad");
            visit(&block.accum_grad_qkv_bias, kQkvDim, prefix + ".attn.qkv.bias.accum_grad");
            visit(&block.accum_grad_attn_proj_weight, kAttnProjWeightElements, prefix + ".attn.proj.weight.accum_grad");
            visit(&block.accum_grad_attn_proj_bias, kDim, prefix + ".attn.proj.bias.accum_grad");
            visit(&block.accum_grad_fc_weight, kFcWeightElements, prefix + ".mlp.fc.weight.accum_grad");
            visit(&block.accum_grad_fc_bias, kHidden, prefix + ".mlp.fc.bias.accum_grad");
            visit(&block.accum_grad_mlp_proj_weight, kMlpProjWeightElements, prefix + ".mlp.proj.weight.accum_grad");
            visit(&block.accum_grad_mlp_proj_bias, kDim, prefix + ".mlp.proj.bias.accum_grad");
        }
    };
    auto visit_block_adam_state_ptrs = [&](auto&& visit) {
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            TransformerBlockParams& block = blocks[i];
            const std::string prefix = "block" + std::to_string(i);
            visit(&block.ln1_weight_avg, kDim, prefix + ".ln1.weight.avg");
            visit(&block.ln1_weight_avg_sq, kDim, prefix + ".ln1.weight.avg_sq");
            visit(&block.ln1_bias_avg, kDim, prefix + ".ln1.bias.avg");
            visit(&block.ln1_bias_avg_sq, kDim, prefix + ".ln1.bias.avg_sq");
            visit(&block.ln2_weight_avg, kDim, prefix + ".ln2.weight.avg");
            visit(&block.ln2_weight_avg_sq, kDim, prefix + ".ln2.weight.avg_sq");
            visit(&block.ln2_bias_avg, kDim, prefix + ".ln2.bias.avg");
            visit(&block.ln2_bias_avg_sq, kDim, prefix + ".ln2.bias.avg_sq");
            visit(&block.qkv_avg, kQkvWeightElements, prefix + ".attn.qkv.weight.avg");
            visit(&block.qkv_avg_sq, kQkvWeightElements, prefix + ".attn.qkv.weight.avg_sq");
            visit(&block.qkv_bias_avg, kQkvDim, prefix + ".attn.qkv.bias.avg");
            visit(&block.qkv_bias_avg_sq, kQkvDim, prefix + ".attn.qkv.bias.avg_sq");
            visit(&block.attn_proj_avg, kAttnProjWeightElements, prefix + ".attn.proj.weight.avg");
            visit(&block.attn_proj_avg_sq, kAttnProjWeightElements, prefix + ".attn.proj.weight.avg_sq");
            visit(&block.attn_proj_bias_avg, kDim, prefix + ".attn.proj.bias.avg");
            visit(&block.attn_proj_bias_avg_sq, kDim, prefix + ".attn.proj.bias.avg_sq");
            visit(&block.fc_avg, kFcWeightElements, prefix + ".mlp.fc.weight.avg");
            visit(&block.fc_avg_sq, kFcWeightElements, prefix + ".mlp.fc.weight.avg_sq");
            visit(&block.fc_bias_avg, kHidden, prefix + ".mlp.fc.bias.avg");
            visit(&block.fc_bias_avg_sq, kHidden, prefix + ".mlp.fc.bias.avg_sq");
            visit(&block.mlp_proj_avg, kMlpProjWeightElements, prefix + ".mlp.proj.weight.avg");
            visit(&block.mlp_proj_avg_sq, kMlpProjWeightElements, prefix + ".mlp.proj.weight.avg_sq");
            visit(&block.mlp_proj_bias_avg, kDim, prefix + ".mlp.proj.bias.avg");
            visit(&block.mlp_proj_bias_avg_sq, kDim, prefix + ".mlp.proj.bias.avg_sq");
        }
    };
    auto visit_block_activation_ptrs = [&](auto&& visit) {
        for (std::size_t i = 0; i < block_tapes.size(); ++i) {
            TransformerBlockActivations& tape = block_tapes[i];
            const std::string prefix = "block" + std::to_string(i);
            visit(&tape.ln1_out, activation_elements, prefix + ".ln1.out");
            visit(&tape.ln1_mean, rows, prefix + ".ln1.mean");
            visit(&tape.ln1_rstd, rows, prefix + ".ln1.rstd");
            if (!packed_qkv_attention_enabled) {
                visit(&tape.qkv, qkv_activation_elements, prefix + ".attn.qkv");
                visit(&tape.q_heads, activation_elements, prefix + ".attn.q_heads");
                visit(&tape.k_heads, activation_elements, prefix + ".attn.k_heads");
                visit(&tape.v_heads, activation_elements, prefix + ".attn.v_heads");
                visit(&tape.attn_heads, activation_elements, prefix + ".attn.heads");
                visit(&tape.attn_out, activation_elements, prefix + ".attn.out");
            }
            visit(&tape.attn_proj, activation_elements, prefix + ".attn.proj");
            visit(&tape.residual1, activation_elements, prefix + ".residual1");
            visit(&tape.ln2_out, activation_elements, prefix + ".ln2.out");
            visit(&tape.ln2_mean, rows, prefix + ".ln2.mean");
            visit(&tape.ln2_rstd, rows, prefix + ".ln2.rstd");
            if (!lazy_validation_mlp_float_scratch_enabled) {
                visit(&tape.fc_out, hidden_elements, prefix + ".mlp.fc");
                visit(&tape.act, hidden_elements, prefix + ".mlp.act");
            }
            visit(&tape.mlp_out, activation_elements, prefix + ".mlp.out");
            visit(&tape.residual2, activation_elements, prefix + ".residual2");
        }
    };

    visit_block_parameter_ptrs([&](float** ptr, std::int64_t elements, const std::string& name) {
        allocate(ptr, elements, name);
    });
    visit_block_gradient_ptrs([&](float** ptr, std::int64_t elements, const std::string& name) {
        allocate(ptr, elements, name);
    });
    visit_block_accum_gradient_ptrs([&](float** ptr, std::int64_t elements, const std::string& name) {
        allocate(ptr, elements, name);
    });
    visit_block_adam_state_ptrs([&](float** ptr, std::int64_t elements, const std::string& name) {
        allocate(ptr, elements, name);
    });
    visit_block_activation_ptrs([&](float** ptr, std::int64_t elements, const std::string& name) {
        allocate(ptr, elements, name);
    });
    for (std::size_t i = 0; i < block_outputs.size(); ++i) {
        allocate(&block_outputs[i], activation_elements, "block" + std::to_string(i) + ".persistent_output");
    }

    for (auto item : {
             std::pair<float**, std::int64_t>{&token_weight, kTokenWeightElements},
             {&position_weight, position_weight_elements},
             {&residual_scale, 1},
             {&lnf_weight, kDim}, {&lnf_bias, kDim},
             {&accum_grad_token_weight, kTokenWeightElements},
             {&accum_grad_position_weight, position_weight_elements},
             {&accum_grad_lnf_weight, kDim}, {&accum_grad_lnf_bias, kDim},
             {&token_avg, kTokenWeightElements}, {&token_avg_sq, kTokenWeightElements},
	             {&position_avg, position_weight_elements}, {&position_avg_sq, position_weight_elements},
	             {&lnf_weight_avg, kDim}, {&lnf_weight_avg_sq, kDim}, {&lnf_bias_avg, kDim}, {&lnf_bias_avg_sq, kDim},
             {&token_out, activation_elements}, {&position_out, activation_elements}, {&x, activation_elements},
             {&lnf_mean, rows}, {&lnf_rstd, rows},
             {&lnf_out, activation_elements}, {&logits, logit_workspace_elements}, {&loss_partials, loss_partial_count},
             {&loss_reduce_a, loss_partial_count}, {&loss_reduce_b, loss_partial_count}, {&loss_total, 1},
             {&row_max, lm_head_chunk_rows}, {&row_denom, lm_head_chunk_rows},
             {&grad_lnf, activation_elements},
             {&grad_residual2, activation_elements}, {&grad_fc_out, hidden_elements},
             {&grad_ln2, activation_elements},
             {&grad_residual1_from_mlp, fuse_ln_backward_residual_enabled ? 0 : activation_elements},
             {&grad_residual1, activation_elements},
             {&grad_attn_out, bf16_attention_grad_out_handoff_enabled ? 0 : activation_elements},
             {&grad_qkv, grad_qkv_float_scratch_elements}, {&grad_ln1, activation_elements},
             {&grad_x_from_attn, fuse_ln_backward_residual_enabled ? 0 : activation_elements},
             {&grad_x, activation_elements},
             {&stored_packed_attention_lse_arena, stored_packed_attention_lse_elements},
             {&grad_sumsq_partials, gradient_partial_count}, {&grad_clip_scale, 1},
         }) {
        allocate(item.first, item.second, "transformer_lm_buffer");
    }
    auto request_uint16_arenas = [&]() {
        if (!combined_uint16_arena_enabled || !error.empty()) {
            return;
        }
        allocate_uint16(
            &stored_mlp_activation_arena,
            stored_mlp_activation_elements,
            "stored_mlp_activation_bf16_arena");
        allocate_uint16(
            &stored_residual1_activation_arena,
            stored_residual1_activation_elements,
            "stored_residual1_bf16_arena");
        allocate_uint16(
            &token_weight_bf16,
            token_weight_bf16_shadow_enabled ? kTokenWeightElements : 0,
            "token_weight_bf16_shadow");
        allocate_uint16(
            &stored_attention_bf16_arena,
            stored_attention_bf16_elements,
            "stored_attention_bf16_arena");
        allocate_uint16(
            &stored_packed_attention_bf16_arena,
            stored_packed_attention_bf16_elements,
            "stored_packed_attention_bf16_arena");
        allocate_uint16(
            &lm_head_bf16_logits,
            lm_head_bf16_logits_enabled ? logit_elements : 0,
            "lm_head_bf16_logits");
        allocate_uint16(
            &lm_head_bf16_hidden,
            (lm_head_bf16_dweight_enabled || lm_head_prepack_bf16_hidden_enabled)
                ? (lm_head_prepack_bf16_hidden_enabled ? activation_elements : lm_head_chunk_rows * kDim)
                : 0,
            "lm_head_bf16_hidden");
        allocate_uint16(&mlp_forward_act_bf16, hidden_elements, "mlp_forward_act_bf16");
        allocate_uint16(
            &projection_bf16_scratch,
            bf16_projection_residual_enabled ? activation_elements * kActivationTapeCount : 0,
            "projection_bf16_scratch");
        if (packed_qkv_attention_enabled) {
            const std::int64_t elements_per_tape = qkv_activation_elements + activation_elements * 2;
            if (elements_per_tape > 0 &&
                kActivationTapeCount <=
                    std::numeric_limits<std::int64_t>::max() / elements_per_tape) {
                allocate_uint16(
                    &packed_qkv_attention_bf16_arena,
                    elements_per_tape * kActivationTapeCount,
                    "packed_qkv_attention_bf16");
            }
        }
        allocate_uint16(
            &attention_grad_out_bf16,
            bf16_attention_grad_out_handoff_enabled ? activation_elements : 0,
            "attention_grad_out_bf16");
        if (trained_layers > 0 &&
            kBlockWeightBf16ElementsPerBlock > 0 &&
            trained_layers <=
                std::numeric_limits<std::int64_t>::max() / kBlockWeightBf16ElementsPerBlock) {
            allocate_uint16(
                &block_weight_bf16_arena,
                kBlockWeightBf16ElementsPerBlock * trained_layers,
                "block_weight_bf16_arena");
        }
        if (bf16_block_dweight_staging_enabled &&
            trained_layers > 0 &&
            kBlockDweightBf16StagingElementsPerBlock > 0 &&
            trained_layers <=
                std::numeric_limits<std::int64_t>::max() / kBlockDweightBf16StagingElementsPerBlock) {
            allocate_uint16(
                &block_dweight_bf16_staging_arena,
                kBlockDweightBf16StagingElementsPerBlock * trained_layers,
                "block_dweight_bf16_staging_arena");
        }
    };
    run_setup_timed("setup.float_arena_materialize", [&]() {
        materialize_float_arena();
    });
    run_setup_timed("setup.uint16_arena_materialize", [&]() {
        request_uint16_arenas();
        materialize_uint16_arena();
    });
    if (stored_packed_attention_lse_arena != nullptr && stored_packed_attention_lse_elements > 0) {
        stored_packed_attention_lse_arena_elements = stored_packed_attention_lse_elements;
        stored_packed_attention_lse_arena_bytes =
            stored_packed_attention_lse_elements * static_cast<std::int64_t>(sizeof(float));
    }
    run_setup_timed("setup.token_arenas", [&]() {
        allocate_token_arenas();
    });
    run_setup_timed("setup.stored_mlp_activation_arena", [&]() {
        allocate_stored_mlp_activation_arena();
    });
    run_setup_timed("setup.stored_residual1_activation_arena", [&]() {
        allocate_stored_residual1_activation_arena();
    });
    run_setup_timed("setup.stored_attention_activation_arenas", [&]() {
        allocate_stored_attention_activation_arenas();
    });
    run_setup_timed("setup.stored_packed_attention_activation_arena", [&]() {
        allocate_stored_packed_attention_activation_arena();
    });
    run_setup_timed("setup.stored_packed_attention_ln1_stats_arena", [&]() {
        allocate_stored_packed_attention_ln1_stats_arena();
    });
    run_setup_timed("setup.lm_head_bf16_logits", [&]() {
        allocate_lm_head_bf16_logits();
    });
    run_setup_timed("setup.lm_head_bf16_hidden", [&]() {
        allocate_lm_head_bf16_hidden();
    });
    run_setup_timed("setup.mlp_forward_act_bf16", [&]() {
        allocate_mlp_forward_act_bf16();
    });
    run_setup_timed("setup.projection_bf16_scratch", [&]() {
        allocate_projection_bf16_scratch();
    });
    run_setup_timed("setup.packed_qkv_attention_scratch", [&]() {
        allocate_packed_qkv_attention_scratch();
    });
    run_setup_timed("setup.attention_grad_out_bf16", [&]() {
        allocate_attention_grad_out_bf16();
    });
    run_setup_timed("setup.block_weight_bf16_arena", [&]() {
        allocate_block_weight_bf16_arena();
    });
    run_setup_timed("setup.block_dweight_bf16_staging_arena", [&]() {
        allocate_block_dweight_bf16_staging_arena();
    });
    const std::int64_t startup_per_buffer_zero_fill_launches_elided =
        1 + trained_layers * 6 + 8 + trained_layers * kPerBlockAdamWStateBuffers;
    const std::int64_t nonzero_parameter_fill_buffer_count =
        direct_bf16_block_weight_init_enabled ? 3 + trained_layers * 2 : 3 + trained_layers * 6;
    const std::int64_t nonzero_bf16_parameter_fill_buffer_count =
        direct_bf16_block_weight_init_enabled ? trained_layers * 4 : 0;

    auto run = [&](int status, const std::string& name) {
        if (status != 0 && error.empty()) {
            error = cuda_error(status, name);
        }
    };
    auto allocate_validation_mlp_float_scratch = [&]() {
        if (!lazy_validation_mlp_float_scratch_enabled || !error.empty()) {
            return;
        }
        if (hidden_elements <= 0 ||
            hidden_elements > static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(float))) {
            error = "validation MLP float scratch allocation byte size overflow";
            return;
        }
        const std::size_t bytes = sizeof(float) * static_cast<std::size_t>(hidden_elements);
        for (TransformerBlockActivations& tape : block_tapes) {
            if (tape.fc_out == nullptr) {
                void* raw = nullptr;
                const int status = device_malloc(&raw, bytes);
                if (status != 0) {
                    error = cuda_error(status, "cudaMalloc validation_mlp_fc_out");
                    return;
                }
                tape.fc_out = static_cast<float*>(raw);
                float_ptrs.push_back(tape.fc_out);
                lazy_validation_mlp_float_scratch_cuda_malloc_count += 1;
                lazy_validation_mlp_float_scratch_elements += hidden_elements;
                lazy_validation_mlp_float_scratch_bytes += static_cast<std::int64_t>(bytes);
            }
            if (tape.act == nullptr) {
                void* raw = nullptr;
                const int status = device_malloc(&raw, bytes);
                if (status != 0) {
                    error = cuda_error(status, "cudaMalloc validation_mlp_act");
                    return;
                }
                tape.act = static_cast<float*>(raw);
                float_ptrs.push_back(tape.act);
                lazy_validation_mlp_float_scratch_cuda_malloc_count += 1;
                lazy_validation_mlp_float_scratch_elements += hidden_elements;
                lazy_validation_mlp_float_scratch_bytes += static_cast<std::int64_t>(bytes);
            }
        }
    };
    auto fill_buffer = [&](float* ptr, std::int64_t elements, float value, const std::string& name) {
        if (error.empty()) {
            run(fill(ptr, elements, value, nullptr), name);
        }
    };
    auto zero_float_buffer = [&](float* ptr, std::int64_t elements, const std::string& name) {
        if (!error.empty() || ptr == nullptr || elements <= 0) {
            return;
        }
        if (startup_cuda_memset_zero_enabled && cuda_memset_async != nullptr) {
            const std::size_t bytes = sizeof(float) * static_cast<std::size_t>(elements);
            const int status = cuda_memset_async(ptr, 0, bytes, nullptr);
            if (status == 0) {
                startup_cuda_memset_zero_fill_count += 1;
                return;
            }
        }
        run(fill(ptr, elements, 0.0f, nullptr), name);
        if (error.empty()) {
            startup_tile_zero_fill_count += 1;
        }
    };
    auto fill_adamw_many = [&](float* const* ptrs, float value, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        run(fill_many(
                ptrs,
                adamw_elements,
                adamw_descriptor_count,
                adamw_max_elements,
                value,
                nullptr),
            name);
    };
    struct AdamWDescriptorHost {
        std::vector<float*> params;
        std::vector<float*> grads;
        std::vector<float*> avgs;
        std::vector<float*> avg_sqs;
        std::vector<std::int64_t> elements;
        std::vector<std::int64_t> partial_offsets;
        std::vector<std::int64_t> bf16_shadow_offsets;
        std::vector<float> decays;
    };
    struct AdamWBf16ParamDescriptorHost {
        std::vector<std::uint16_t*> params;
        std::vector<float*> grads;
        std::vector<float*> avgs;
        std::vector<float*> avg_sqs;
        std::vector<std::int64_t> elements;
        std::vector<std::int64_t> partial_offsets;
        std::vector<float> decays;
    };
    struct AdamWBf16ParamBf16GradDescriptorHost {
        std::vector<std::uint16_t*> params;
        std::vector<std::uint16_t*> grads;
        std::vector<float*> avgs;
        std::vector<float*> avg_sqs;
        std::vector<std::int64_t> elements;
        std::vector<std::int64_t> partial_offsets;
        std::vector<float> decays;
    };
    struct ParameterFillDescriptorHost {
        std::vector<float*> ptrs;
        std::vector<std::int64_t> elements;
        std::vector<float> values;
    };
    struct Bf16ParameterFillDescriptorHost {
        std::vector<std::uint16_t*> ptrs;
        std::vector<std::int64_t> elements;
        std::vector<float> values;
    };
    struct Bf16ShadowDescriptorHost {
        std::vector<const float*> sources;
        std::vector<std::int64_t> elements;
        std::vector<std::int64_t> offsets;
    };
    AdamWDescriptorHost adamw_host;
    AdamWDescriptorHost adamw_float_update_host;
    AdamWBf16ParamDescriptorHost adamw_bf16_param_host;
    AdamWBf16ParamBf16GradDescriptorHost adamw_bf16_param_bf16_grad_host;
    ParameterFillDescriptorHost parameter_fill_host;
    Bf16ParameterFillDescriptorHost bf16_parameter_fill_host;
    Bf16ShadowDescriptorHost block_weight_bf16_host;
    auto build_adamw_descriptors = [&]() {
        if (!error.empty()) {
            return;
        }
        std::int64_t partial_offset = 0;
        auto add = [&](float* param,
                       float* grad,
                       float* avg,
                       float* avg_sq,
                       std::int64_t elements,
                       float decay,
                       std::int64_t bf16_shadow_offset = -1,
                       std::uint16_t* bf16_param = nullptr,
                       std::uint16_t* bf16_grad = nullptr) {
            if (!error.empty()) {
                return;
            }
            if (param == nullptr || grad == nullptr || avg == nullptr || avg_sq == nullptr) {
                error = "null pointer while building fused AdamW descriptors";
                return;
            }
            if (elements <= 0) {
                error = "non-positive element count while building fused AdamW descriptors";
                return;
            }
            adamw_host.params.push_back(param);
            adamw_host.grads.push_back(grad);
            adamw_host.avgs.push_back(avg);
            adamw_host.avg_sqs.push_back(avg_sq);
            adamw_host.elements.push_back(elements);
            adamw_host.partial_offsets.push_back(partial_offset);
            adamw_host.bf16_shadow_offsets.push_back(bf16_shadow_offset);
            adamw_host.decays.push_back(decay);
            adamw_max_elements = std::max<std::int64_t>(adamw_max_elements, elements);
            if (bf16_block_weight_param_update_enabled && bf16_param != nullptr) {
                if (bf16_block_dweight_staging_enabled && bf16_grad != nullptr) {
                    adamw_bf16_param_bf16_grad_host.params.push_back(bf16_param);
                    adamw_bf16_param_bf16_grad_host.grads.push_back(bf16_grad);
                    adamw_bf16_param_bf16_grad_host.avgs.push_back(avg);
                    adamw_bf16_param_bf16_grad_host.avg_sqs.push_back(avg_sq);
                    adamw_bf16_param_bf16_grad_host.elements.push_back(elements);
                    adamw_bf16_param_bf16_grad_host.partial_offsets.push_back(partial_offset);
                    adamw_bf16_param_bf16_grad_host.decays.push_back(decay);
                    adamw_bf16_param_bf16_grad_max_elements =
                        std::max<std::int64_t>(adamw_bf16_param_bf16_grad_max_elements, elements);
                } else {
                    adamw_bf16_param_host.params.push_back(bf16_param);
                    adamw_bf16_param_host.grads.push_back(grad);
                    adamw_bf16_param_host.avgs.push_back(avg);
                    adamw_bf16_param_host.avg_sqs.push_back(avg_sq);
                    adamw_bf16_param_host.elements.push_back(elements);
                    adamw_bf16_param_host.partial_offsets.push_back(partial_offset);
                    adamw_bf16_param_host.decays.push_back(decay);
                    adamw_bf16_param_max_elements =
                        std::max<std::int64_t>(adamw_bf16_param_max_elements, elements);
                }
            } else {
                adamw_float_update_host.params.push_back(param);
                adamw_float_update_host.grads.push_back(grad);
                adamw_float_update_host.avgs.push_back(avg);
                adamw_float_update_host.avg_sqs.push_back(avg_sq);
                adamw_float_update_host.elements.push_back(elements);
                adamw_float_update_host.partial_offsets.push_back(partial_offset);
                adamw_float_update_host.decays.push_back(decay);
                adamw_float_update_max_elements =
                    std::max<std::int64_t>(adamw_float_update_max_elements, elements);
            }
            partial_offset += partial_count_for(elements);
        };

        add(token_weight, accum_grad_token_weight, token_avg, token_avg_sq, kTokenWeightElements, kWeightDecay);
        add(position_weight, accum_grad_position_weight, position_avg, position_avg_sq, position_weight_elements, kWeightDecay);
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            TransformerBlockParams& block = blocks[i];
            const std::int64_t shadow_base =
                static_cast<std::int64_t>(i) * kBlockWeightBf16ElementsPerBlock;
            add(block.ln1_weight, block.accum_grad_ln1_weight, block.ln1_weight_avg, block.ln1_weight_avg_sq, kDim, 0.0f);
            add(block.ln1_bias, block.accum_grad_ln1_bias, block.ln1_bias_avg, block.ln1_bias_avg_sq, kDim, 0.0f);
            add(block.qkv_weight, block.accum_grad_qkv_weight, block.qkv_avg, block.qkv_avg_sq, kQkvWeightElements, kWeightDecay, shadow_base, block.qkv_weight_bf16, block.accum_grad_qkv_weight_bf16);
            add(block.qkv_bias, block.accum_grad_qkv_bias, block.qkv_bias_avg, block.qkv_bias_avg_sq, kQkvDim, 0.0f);
            add(block.attn_proj_weight, block.accum_grad_attn_proj_weight, block.attn_proj_avg, block.attn_proj_avg_sq, kAttnProjWeightElements, kWeightDecay, shadow_base + kQkvWeightElements, block.attn_proj_weight_bf16);
            add(block.attn_proj_bias, block.accum_grad_attn_proj_bias, block.attn_proj_bias_avg, block.attn_proj_bias_avg_sq, kDim, 0.0f);
            add(block.ln2_weight, block.accum_grad_ln2_weight, block.ln2_weight_avg, block.ln2_weight_avg_sq, kDim, 0.0f);
            add(block.ln2_bias, block.accum_grad_ln2_bias, block.ln2_bias_avg, block.ln2_bias_avg_sq, kDim, 0.0f);
            add(block.fc_weight, block.accum_grad_fc_weight, block.fc_avg, block.fc_avg_sq, kFcWeightElements, kWeightDecay, shadow_base + kQkvWeightElements + kAttnProjWeightElements, block.fc_weight_bf16, block.accum_grad_fc_weight_bf16);
            add(block.fc_bias, block.accum_grad_fc_bias, block.fc_bias_avg, block.fc_bias_avg_sq, kHidden, 0.0f);
            add(block.mlp_proj_weight, block.accum_grad_mlp_proj_weight, block.mlp_proj_avg, block.mlp_proj_avg_sq, kMlpProjWeightElements, kWeightDecay, shadow_base + kQkvWeightElements + kAttnProjWeightElements + kFcWeightElements, block.mlp_proj_weight_bf16);
            add(block.mlp_proj_bias, block.accum_grad_mlp_proj_bias, block.mlp_proj_bias_avg, block.mlp_proj_bias_avg_sq, kDim, 0.0f);
        }
        add(lnf_weight, accum_grad_lnf_weight, lnf_weight_avg, lnf_weight_avg_sq, kDim, 0.0f);
        add(lnf_bias, accum_grad_lnf_bias, lnf_bias_avg, lnf_bias_avg_sq, kDim, 0.0f);
        if (!error.empty()) {
            return;
        }
        if (partial_offset != gradient_partial_count) {
            std::ostringstream out;
            out << "GPT-2 transformer/LM gradient partial descriptor mismatch: expected "
                << gradient_partial_count << " described " << partial_offset;
            error = out.str();
            return;
        }
        adamw_descriptor_count = static_cast<std::int64_t>(adamw_host.params.size());
        adamw_float_update_descriptor_count =
            static_cast<std::int64_t>(adamw_float_update_host.params.size());
        adamw_bf16_param_descriptor_count =
            static_cast<std::int64_t>(adamw_bf16_param_host.params.size());
        adamw_bf16_param_bf16_grad_descriptor_count =
            static_cast<std::int64_t>(adamw_bf16_param_bf16_grad_host.params.size());
    };
    run_setup_timed("setup.build_adamw_descriptors", [&]() {
        build_adamw_descriptors();
    });
    struct FloatZeroRange {
        float* ptr = nullptr;
        std::int64_t elements = 0;
    };
    std::vector<FloatZeroRange> adamw_zero_ranges;
    std::vector<FloatZeroRange> gradient_zero_ranges;
    auto build_zero_ranges_from_spans =
        [&](const std::vector<std::pair<float*, std::int64_t>>& buffers,
            std::vector<FloatZeroRange>& ranges,
            std::int64_t& range_elements,
            const std::string& label) {
        ranges.clear();
        range_elements = 0;
        if (!error.empty() || buffers.empty()) {
            return;
        }
        struct Span {
            std::uintptr_t begin = 0;
            std::uintptr_t end = 0;
        };
        std::vector<Span> spans;
        spans.reserve(buffers.size());
        auto add_span = [&](float* ptr, std::int64_t elements) {
            if (error.empty() && ptr == nullptr) {
                error = "null pointer while building " + label + " zero ranges";
                return;
            }
            if (error.empty() && elements <= 0) {
                error = "non-positive element count while building " + label + " zero ranges";
                return;
            }
            if (!error.empty()) {
                return;
            }
            const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(ptr);
            const std::uintptr_t bytes = static_cast<std::uintptr_t>(
                sizeof(float) * static_cast<std::size_t>(elements));
            spans.push_back(Span{begin, begin + bytes});
        };
        for (const auto& buffer : buffers) {
            add_span(buffer.first, buffer.second);
        }
        if (!error.empty() || spans.empty()) {
            return;
        }
        std::sort(spans.begin(), spans.end(), [](const Span& a, const Span& b) {
            return a.begin < b.begin;
        });
        constexpr std::uintptr_t kMaxCoalescedZeroPaddingBytes =
            static_cast<std::uintptr_t>(kFloatArenaAlignmentElements * sizeof(float));
        Span current = spans.front();
        auto flush = [&]() {
            if (current.end <= current.begin ||
                (current.end - current.begin) % sizeof(float) != 0) {
                error = "invalid " + label + " zero range alignment";
                return;
            }
            const std::int64_t elements =
                static_cast<std::int64_t>((current.end - current.begin) / sizeof(float));
            ranges.push_back(FloatZeroRange{reinterpret_cast<float*>(current.begin), elements});
            range_elements += elements;
        };
        for (std::size_t i = 1; i < spans.size(); ++i) {
            const Span& span = spans[i];
            if (span.begin <= current.end + kMaxCoalescedZeroPaddingBytes) {
                current.end = std::max(current.end, span.end);
            } else {
                flush();
                if (!error.empty()) {
                    return;
                }
                current = span;
            }
        }
        flush();
    };
    auto build_adamw_zero_ranges = [&]() {
        adamw_zero_ranges.clear();
        adamw_state_zero_range_elements = 0;
        if (!error.empty() || !startup_zero_adamw_state_ranges_enabled) {
            return;
        }
        if (adamw_host.avgs.size() != adamw_host.elements.size() ||
            adamw_host.avg_sqs.size() != adamw_host.elements.size()) {
            error = "AdamW zero range descriptor size mismatch";
            return;
        }
        std::vector<std::pair<float*, std::int64_t>> buffers;
        buffers.reserve(adamw_host.elements.size() * 2);
        for (std::size_t i = 0; i < adamw_host.elements.size(); ++i) {
            buffers.push_back({adamw_host.avgs[i], adamw_host.elements[i]});
            buffers.push_back({adamw_host.avg_sqs[i], adamw_host.elements[i]});
        }
        build_zero_ranges_from_spans(
            buffers, adamw_zero_ranges, adamw_state_zero_range_elements, "AdamW state");
    };
    build_adamw_zero_ranges();
    auto build_gradient_zero_ranges = [&]() {
        gradient_zero_ranges.clear();
        gradient_zero_range_elements = 0;
        if (!error.empty() || !gradient_cuda_memset_zero_enabled) {
            return;
        }
        if (adamw_host.grads.size() != adamw_host.elements.size()) {
            error = "gradient zero range descriptor size mismatch";
            return;
        }
        std::vector<std::pair<float*, std::int64_t>> buffers;
        if (bf16_block_weight_param_update_enabled) {
            if (adamw_float_update_host.grads.size() != adamw_float_update_host.elements.size() ||
                adamw_bf16_param_host.grads.size() != adamw_bf16_param_host.elements.size()) {
                error = "split gradient zero range descriptor size mismatch";
                return;
            }
            buffers.reserve(adamw_float_update_host.elements.size() + adamw_bf16_param_host.elements.size());
            for (std::size_t i = 0; i < adamw_float_update_host.elements.size(); ++i) {
                buffers.push_back({adamw_float_update_host.grads[i], adamw_float_update_host.elements[i]});
            }
            for (std::size_t i = 0; i < adamw_bf16_param_host.elements.size(); ++i) {
                buffers.push_back({adamw_bf16_param_host.grads[i], adamw_bf16_param_host.elements[i]});
            }
        } else {
            buffers.reserve(adamw_host.elements.size());
            for (std::size_t i = 0; i < adamw_host.elements.size(); ++i) {
                buffers.push_back({adamw_host.grads[i], adamw_host.elements[i]});
            }
        }
        build_zero_ranges_from_spans(
            buffers, gradient_zero_ranges, gradient_zero_range_elements, "gradient");
        gradient_zero_range_count = static_cast<std::int64_t>(gradient_zero_ranges.size());
    };
    build_gradient_zero_ranges();
    auto build_parameter_fill_descriptors = [&]() {
        if (!error.empty()) {
            return;
        }
        auto add = [&](float* ptr, std::int64_t elements, float value, const std::string& name) {
            if (!error.empty()) {
                return;
            }
            if (ptr == nullptr) {
                error = "null pointer while building fused parameter-fill descriptors: " + name;
                return;
            }
            if (elements <= 0) {
                error = "non-positive element count while building fused parameter-fill descriptors: " + name;
                return;
            }
            parameter_fill_host.ptrs.push_back(ptr);
            parameter_fill_host.elements.push_back(elements);
            parameter_fill_host.values.push_back(value);
            parameter_fill_max_elements = std::max<std::int64_t>(parameter_fill_max_elements, elements);
        };
        auto add_bf16 = [&](std::uint16_t* ptr, std::int64_t elements, float value, const std::string& name) {
            if (!error.empty()) {
                return;
            }
            if (ptr == nullptr) {
                error = "null pointer while building fused BF16 parameter-fill descriptors: " + name;
                return;
            }
            if (elements <= 0) {
                error = "non-positive element count while building fused BF16 parameter-fill descriptors: " + name;
                return;
            }
            bf16_parameter_fill_host.ptrs.push_back(ptr);
            bf16_parameter_fill_host.elements.push_back(elements);
            bf16_parameter_fill_host.values.push_back(value);
            bf16_parameter_fill_max_elements =
                std::max<std::int64_t>(bf16_parameter_fill_max_elements, elements);
        };

        add(position_weight, position_weight_elements, kInitialPositionWeight, "position_weight");
        add(residual_scale, 1, kResidualScale, "residual_scale");
        add(lnf_weight, kDim, kLnWeight, "lnf_weight");
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            TransformerBlockParams& block = blocks[i];
            const std::string prefix = "block" + std::to_string(i);
            add(block.ln1_weight, kDim, kLnWeight, prefix + ".ln1.weight");
            add(block.ln2_weight, kDim, kLnWeight, prefix + ".ln2.weight");
            if (direct_bf16_block_weight_init_enabled) {
                add_bf16(block.qkv_weight_bf16, kQkvWeightElements, kQkvWeight, prefix + ".attn.qkv.weight.bf16");
                add_bf16(
                    block.attn_proj_weight_bf16,
                    kAttnProjWeightElements,
                    kAttnProjWeight,
                    prefix + ".attn.proj.weight.bf16");
                add_bf16(block.fc_weight_bf16, kFcWeightElements, kFcWeight, prefix + ".mlp.fc.weight.bf16");
                add_bf16(
                    block.mlp_proj_weight_bf16,
                    kMlpProjWeightElements,
                    kMlpProjWeight,
                    prefix + ".mlp.proj.weight.bf16");
            } else {
                add(block.qkv_weight, kQkvWeightElements, kQkvWeight, prefix + ".attn.qkv.weight");
                add(block.attn_proj_weight, kAttnProjWeightElements, kAttnProjWeight, prefix + ".attn.proj.weight");
                add(block.fc_weight, kFcWeightElements, kFcWeight, prefix + ".mlp.fc.weight");
                add(block.mlp_proj_weight, kMlpProjWeightElements, kMlpProjWeight, prefix + ".mlp.proj.weight");
            }
        }
        if (!error.empty()) {
            return;
        }
        if (static_cast<std::int64_t>(parameter_fill_host.ptrs.size()) != nonzero_parameter_fill_buffer_count) {
            std::ostringstream out;
            out << "GPT-2 transformer/LM parameter fill descriptor mismatch: expected "
                << nonzero_parameter_fill_buffer_count << " described " << parameter_fill_host.ptrs.size();
            error = out.str();
            return;
        }
        if (static_cast<std::int64_t>(bf16_parameter_fill_host.ptrs.size()) !=
            nonzero_bf16_parameter_fill_buffer_count) {
            std::ostringstream out;
            out << "GPT-2 transformer/LM BF16 parameter fill descriptor mismatch: expected "
                << nonzero_bf16_parameter_fill_buffer_count << " described "
                << bf16_parameter_fill_host.ptrs.size();
            error = out.str();
            return;
        }
        parameter_fill_descriptor_count = static_cast<std::int64_t>(parameter_fill_host.ptrs.size());
        bf16_parameter_fill_descriptor_count =
            static_cast<std::int64_t>(bf16_parameter_fill_host.ptrs.size());
    };
    run_setup_timed("setup.build_parameter_fill_descriptors", [&]() {
        build_parameter_fill_descriptors();
    });
    auto build_block_weight_bf16_descriptors = [&]() {
        if (!error.empty()) {
            return;
        }
        if (block_weight_bf16_arena == nullptr) {
            error = "block BF16 weight arena was not allocated";
            return;
        }
        std::int64_t offset = 0;
        auto add = [&](const float* source, std::int64_t elements, const std::string& name) {
            if (!error.empty()) {
                return;
            }
            if (source == nullptr) {
                error = "null pointer while building BF16 block weight descriptors: " + name;
                return;
            }
            if (elements <= 0) {
                error = "non-positive element count while building BF16 block weight descriptors: " + name;
                return;
            }
            block_weight_bf16_host.sources.push_back(source);
            block_weight_bf16_host.elements.push_back(elements);
            block_weight_bf16_host.offsets.push_back(offset);
            block_weight_bf16_max_elements = std::max<std::int64_t>(block_weight_bf16_max_elements, elements);
            offset += elements;
        };
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            TransformerBlockParams& block = blocks[i];
            const std::string prefix = "block" + std::to_string(i);
            add(block.qkv_weight, kQkvWeightElements, prefix + ".attn.qkv.weight.bf16_shadow");
            add(block.attn_proj_weight, kAttnProjWeightElements, prefix + ".attn.proj.weight.bf16_shadow");
            add(block.fc_weight, kFcWeightElements, prefix + ".mlp.fc.weight.bf16_shadow");
            add(block.mlp_proj_weight, kMlpProjWeightElements, prefix + ".mlp.proj.weight.bf16_shadow");
        }
        if (error.empty() && offset != block_weight_bf16_arena_elements) {
            std::ostringstream out;
            out << "BF16 block weight descriptor mismatch: arena has "
                << block_weight_bf16_arena_elements << " elements but descriptors cover " << offset;
            error = out.str();
        }
        block_weight_bf16_descriptor_count = static_cast<std::int64_t>(block_weight_bf16_host.sources.size());
    };
    run_setup_timed("setup.build_block_weight_bf16_descriptors", [&]() {
        build_block_weight_bf16_descriptors();
    });
    auto materialize_descriptor_arena = [&]() {
        if (!error.empty()) {
            return;
        }
        struct DescriptorArenaRequest {
            void** ptr = nullptr;
            const void* host = nullptr;
            std::size_t bytes = 0;
            std::size_t offset = 0;
            std::string name;
        };
        constexpr std::size_t kDescriptorArenaAlignmentBytes = 16;
        std::vector<DescriptorArenaRequest> requests;
        auto align_descriptor_offset = [](std::size_t value) {
            return ((value + kDescriptorArenaAlignmentBytes - 1) / kDescriptorArenaAlignmentBytes) *
                   kDescriptorArenaAlignmentBytes;
        };
        auto add_region = [&](void** ptr, const void* host, std::size_t bytes, const std::string& name) {
            if (!error.empty() || bytes == 0) {
                return;
            }
            const std::size_t offset = align_descriptor_offset(descriptor_arena_bytes);
            if (offset > std::numeric_limits<std::size_t>::max() - bytes) {
                error = "descriptor arena allocation size overflow for " + name;
                return;
            }
            descriptor_arena_requested_bytes += bytes;
            requests.push_back(DescriptorArenaRequest{ptr, host, bytes, offset, name});
            descriptor_arena_bytes = offset + bytes;
        };

        const std::size_t adamw_pointer_bytes = sizeof(float*) * adamw_host.params.size();
        const std::size_t adamw_element_bytes = sizeof(std::int64_t) * adamw_host.elements.size();
        const std::size_t adamw_partial_offset_bytes = sizeof(std::int64_t) * adamw_host.partial_offsets.size();
        const std::size_t adamw_bf16_shadow_offset_bytes =
            sizeof(std::int64_t) * adamw_host.bf16_shadow_offsets.size();
        const std::size_t adamw_decay_bytes = sizeof(float) * adamw_host.decays.size();
        const std::size_t adamw_float_update_pointer_bytes =
            sizeof(float*) * adamw_float_update_host.params.size();
        const std::size_t adamw_float_update_element_bytes =
            sizeof(std::int64_t) * adamw_float_update_host.elements.size();
        const std::size_t adamw_float_update_partial_offset_bytes =
            sizeof(std::int64_t) * adamw_float_update_host.partial_offsets.size();
        const std::size_t adamw_float_update_decay_bytes =
            sizeof(float) * adamw_float_update_host.decays.size();
        const std::size_t adamw_bf16_param_pointer_bytes =
            sizeof(std::uint16_t*) * adamw_bf16_param_host.params.size();
        const std::size_t adamw_bf16_param_float_pointer_bytes =
            sizeof(float*) * adamw_bf16_param_host.grads.size();
        const std::size_t adamw_bf16_param_element_bytes =
            sizeof(std::int64_t) * adamw_bf16_param_host.elements.size();
        const std::size_t adamw_bf16_param_partial_offset_bytes =
            sizeof(std::int64_t) * adamw_bf16_param_host.partial_offsets.size();
        const std::size_t adamw_bf16_param_decay_bytes =
            sizeof(float) * adamw_bf16_param_host.decays.size();
        const std::size_t adamw_bf16_param_bf16_grad_param_pointer_bytes =
            sizeof(std::uint16_t*) * adamw_bf16_param_bf16_grad_host.params.size();
        const std::size_t adamw_bf16_param_bf16_grad_pointer_bytes =
            sizeof(std::uint16_t*) * adamw_bf16_param_bf16_grad_host.grads.size();
        const std::size_t adamw_bf16_param_bf16_grad_float_pointer_bytes =
            sizeof(float*) * adamw_bf16_param_bf16_grad_host.avgs.size();
        const std::size_t adamw_bf16_param_bf16_grad_element_bytes =
            sizeof(std::int64_t) * adamw_bf16_param_bf16_grad_host.elements.size();
        const std::size_t adamw_bf16_param_bf16_grad_partial_offset_bytes =
            sizeof(std::int64_t) * adamw_bf16_param_bf16_grad_host.partial_offsets.size();
        const std::size_t adamw_bf16_param_bf16_grad_decay_bytes =
            sizeof(float) * adamw_bf16_param_bf16_grad_host.decays.size();
        const std::size_t fill_pointer_bytes = sizeof(float*) * parameter_fill_host.ptrs.size();
        const std::size_t fill_element_bytes = sizeof(std::int64_t) * parameter_fill_host.elements.size();
        const std::size_t fill_value_bytes = sizeof(float) * parameter_fill_host.values.size();
        const std::size_t bf16_fill_pointer_bytes =
            sizeof(std::uint16_t*) * bf16_parameter_fill_host.ptrs.size();
        const std::size_t bf16_fill_element_bytes =
            sizeof(std::int64_t) * bf16_parameter_fill_host.elements.size();
        const std::size_t bf16_fill_value_bytes =
            sizeof(float) * bf16_parameter_fill_host.values.size();
        const std::size_t bf16_source_bytes = sizeof(const float*) * block_weight_bf16_host.sources.size();
        const std::size_t bf16_element_bytes = sizeof(std::int64_t) * block_weight_bf16_host.elements.size();
        const std::size_t bf16_offset_bytes = sizeof(std::int64_t) * block_weight_bf16_host.offsets.size();
        add_region(reinterpret_cast<void**>(&adamw_param_ptrs), adamw_host.params.data(), adamw_pointer_bytes, "adamw_param_ptrs");
        add_region(reinterpret_cast<void**>(&adamw_grad_ptrs), adamw_host.grads.data(), adamw_pointer_bytes, "adamw_grad_ptrs");
        add_region(reinterpret_cast<void**>(&adamw_avg_ptrs), adamw_host.avgs.data(), adamw_pointer_bytes, "adamw_avg_ptrs");
        add_region(reinterpret_cast<void**>(&adamw_avg_sq_ptrs), adamw_host.avg_sqs.data(), adamw_pointer_bytes, "adamw_avg_sq_ptrs");
        add_region(reinterpret_cast<void**>(&adamw_elements), adamw_host.elements.data(), adamw_element_bytes, "adamw_elements");
        add_region(reinterpret_cast<void**>(&gradient_partial_offsets), adamw_host.partial_offsets.data(), adamw_partial_offset_bytes, "gradient_partial_offsets");
        add_region(reinterpret_cast<void**>(&adamw_bf16_shadow_offsets), adamw_host.bf16_shadow_offsets.data(), adamw_bf16_shadow_offset_bytes, "adamw_bf16_shadow_offsets");
        add_region(reinterpret_cast<void**>(&adamw_weight_decays), adamw_host.decays.data(), adamw_decay_bytes, "adamw_weight_decays");
        if (bf16_block_weight_param_update_enabled) {
            add_region(reinterpret_cast<void**>(&adamw_float_update_param_ptrs), adamw_float_update_host.params.data(), adamw_float_update_pointer_bytes, "adamw_float_update_param_ptrs");
            add_region(reinterpret_cast<void**>(&adamw_float_update_grad_ptrs), adamw_float_update_host.grads.data(), adamw_float_update_pointer_bytes, "adamw_float_update_grad_ptrs");
            add_region(reinterpret_cast<void**>(&adamw_float_update_avg_ptrs), adamw_float_update_host.avgs.data(), adamw_float_update_pointer_bytes, "adamw_float_update_avg_ptrs");
            add_region(reinterpret_cast<void**>(&adamw_float_update_avg_sq_ptrs), adamw_float_update_host.avg_sqs.data(), adamw_float_update_pointer_bytes, "adamw_float_update_avg_sq_ptrs");
            add_region(reinterpret_cast<void**>(&adamw_float_update_elements), adamw_float_update_host.elements.data(), adamw_float_update_element_bytes, "adamw_float_update_elements");
            add_region(reinterpret_cast<void**>(&adamw_float_update_partial_offsets), adamw_float_update_host.partial_offsets.data(), adamw_float_update_partial_offset_bytes, "adamw_float_update_partial_offsets");
            add_region(reinterpret_cast<void**>(&adamw_float_update_weight_decays), adamw_float_update_host.decays.data(), adamw_float_update_decay_bytes, "adamw_float_update_weight_decays");
            add_region(reinterpret_cast<void**>(&adamw_bf16_param_ptrs), adamw_bf16_param_host.params.data(), adamw_bf16_param_pointer_bytes, "adamw_bf16_param_ptrs");
            add_region(reinterpret_cast<void**>(&adamw_bf16_param_grad_ptrs), adamw_bf16_param_host.grads.data(), adamw_bf16_param_float_pointer_bytes, "adamw_bf16_param_grad_ptrs");
            add_region(reinterpret_cast<void**>(&adamw_bf16_param_avg_ptrs), adamw_bf16_param_host.avgs.data(), adamw_bf16_param_float_pointer_bytes, "adamw_bf16_param_avg_ptrs");
            add_region(reinterpret_cast<void**>(&adamw_bf16_param_avg_sq_ptrs), adamw_bf16_param_host.avg_sqs.data(), adamw_bf16_param_float_pointer_bytes, "adamw_bf16_param_avg_sq_ptrs");
            add_region(reinterpret_cast<void**>(&adamw_bf16_param_elements), adamw_bf16_param_host.elements.data(), adamw_bf16_param_element_bytes, "adamw_bf16_param_elements");
            add_region(reinterpret_cast<void**>(&adamw_bf16_param_partial_offsets), adamw_bf16_param_host.partial_offsets.data(), adamw_bf16_param_partial_offset_bytes, "adamw_bf16_param_partial_offsets");
            add_region(reinterpret_cast<void**>(&adamw_bf16_param_weight_decays), adamw_bf16_param_host.decays.data(), adamw_bf16_param_decay_bytes, "adamw_bf16_param_weight_decays");
            add_region(reinterpret_cast<void**>(&adamw_bf16_param_bf16_grad_param_ptrs), adamw_bf16_param_bf16_grad_host.params.data(), adamw_bf16_param_bf16_grad_param_pointer_bytes, "adamw_bf16_param_bf16_grad_param_ptrs");
            add_region(reinterpret_cast<void**>(&adamw_bf16_param_bf16_grad_grad_ptrs), adamw_bf16_param_bf16_grad_host.grads.data(), adamw_bf16_param_bf16_grad_pointer_bytes, "adamw_bf16_param_bf16_grad_grad_ptrs");
            add_region(reinterpret_cast<void**>(&adamw_bf16_param_bf16_grad_avg_ptrs), adamw_bf16_param_bf16_grad_host.avgs.data(), adamw_bf16_param_bf16_grad_float_pointer_bytes, "adamw_bf16_param_bf16_grad_avg_ptrs");
            add_region(reinterpret_cast<void**>(&adamw_bf16_param_bf16_grad_avg_sq_ptrs), adamw_bf16_param_bf16_grad_host.avg_sqs.data(), adamw_bf16_param_bf16_grad_float_pointer_bytes, "adamw_bf16_param_bf16_grad_avg_sq_ptrs");
            add_region(reinterpret_cast<void**>(&adamw_bf16_param_bf16_grad_elements), adamw_bf16_param_bf16_grad_host.elements.data(), adamw_bf16_param_bf16_grad_element_bytes, "adamw_bf16_param_bf16_grad_elements");
            add_region(reinterpret_cast<void**>(&adamw_bf16_param_bf16_grad_partial_offsets), adamw_bf16_param_bf16_grad_host.partial_offsets.data(), adamw_bf16_param_bf16_grad_partial_offset_bytes, "adamw_bf16_param_bf16_grad_partial_offsets");
            add_region(reinterpret_cast<void**>(&adamw_bf16_param_bf16_grad_weight_decays), adamw_bf16_param_bf16_grad_host.decays.data(), adamw_bf16_param_bf16_grad_decay_bytes, "adamw_bf16_param_bf16_grad_weight_decays");
        }
        add_region(reinterpret_cast<void**>(&parameter_fill_ptrs), parameter_fill_host.ptrs.data(), fill_pointer_bytes, "parameter_fill_ptrs");
        add_region(reinterpret_cast<void**>(&parameter_fill_elements), parameter_fill_host.elements.data(), fill_element_bytes, "parameter_fill_elements");
        add_region(reinterpret_cast<void**>(&parameter_fill_values), parameter_fill_host.values.data(), fill_value_bytes, "parameter_fill_values");
        add_region(reinterpret_cast<void**>(&bf16_parameter_fill_ptrs), bf16_parameter_fill_host.ptrs.data(), bf16_fill_pointer_bytes, "bf16_parameter_fill_ptrs");
        add_region(reinterpret_cast<void**>(&bf16_parameter_fill_elements), bf16_parameter_fill_host.elements.data(), bf16_fill_element_bytes, "bf16_parameter_fill_elements");
        add_region(reinterpret_cast<void**>(&bf16_parameter_fill_values), bf16_parameter_fill_host.values.data(), bf16_fill_value_bytes, "bf16_parameter_fill_values");
        add_region(reinterpret_cast<void**>(&block_weight_bf16_sources), block_weight_bf16_host.sources.data(), bf16_source_bytes, "block_weight_bf16_sources");
        add_region(reinterpret_cast<void**>(&block_weight_bf16_elements), block_weight_bf16_host.elements.data(), bf16_element_bytes, "block_weight_bf16_elements");
        add_region(reinterpret_cast<void**>(&block_weight_bf16_offsets), block_weight_bf16_host.offsets.data(), bf16_offset_bytes, "block_weight_bf16_offsets");
        if (!error.empty() || requests.empty()) {
            return;
        }
        descriptor_arena_bytes = align_descriptor_offset(descriptor_arena_bytes);
        const int status = device_malloc(&descriptor_arena, descriptor_arena_bytes);
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc transformer_lm_descriptor_arena");
            return;
        }
        descriptor_ptrs.push_back(descriptor_arena);
        descriptor_arena_cuda_malloc_count = 1;
        auto* base = static_cast<unsigned char*>(descriptor_arena);
        std::vector<unsigned char> host_descriptor_arena(descriptor_arena_bytes, 0);
        for (const DescriptorArenaRequest& request : requests) {
            *request.ptr = base + request.offset;
            std::memcpy(host_descriptor_arena.data() + request.offset, request.host, request.bytes);
            descriptor_arena_suballocation_count += 1;
        }
        const int copy_status = cuda_memcpy(
            descriptor_arena,
            host_descriptor_arena.data(),
            descriptor_arena_bytes,
            kCudaMemcpyHostToDevice);
        if (copy_status != 0) {
            error = cuda_error(copy_status, "cudaMemcpy transformer_lm_descriptor_arena");
            return;
        }
        descriptor_arena_copy_count = 1;
    };
    run_setup_timed("setup.descriptor_arena_materialize", [&]() {
        materialize_descriptor_arena();
    });
    run_setup_timed("setup.zero_init", [&]() {
        if (error.empty() && startup_zero_adamw_state_ranges_enabled && !adamw_zero_ranges.empty()) {
            for (const FloatZeroRange& range : adamw_zero_ranges) {
                zero_float_buffer(range.ptr, range.elements, "adamw_state.zero_range");
                if (!error.empty()) {
                    break;
                }
                adamw_state_zero_fill_count += 1;
            }
            if (error.empty()) {
                adamw_state_zero_range_count = static_cast<std::int64_t>(adamw_zero_ranges.size());
            }
        } else if (error.empty() && startup_zero_adamw_state_only_enabled && adamw_descriptor_count > 0) {
            fill_adamw_many(adamw_avg_ptrs, 0.0f, "adamw_avg.zero_many");
            if (error.empty()) {
                adamw_state_zero_fill_count += 1;
            }
            fill_adamw_many(adamw_avg_sq_ptrs, 0.0f, "adamw_avg_sq.zero_many");
            if (error.empty()) {
                adamw_state_zero_fill_count += 1;
            }
        } else if (error.empty() && float_arena != nullptr && float_arena_allocated_elements > 0) {
            zero_float_buffer(float_arena, float_arena_allocated_elements, "transformer_lm_float_arena.zero");
            if (error.empty()) {
                float_arena_zero_fill_count = 1;
            }
        }
    });
    run_setup_timed("setup.token_weight_init", [&]() {
        if (error.empty()) {
            if (fuse_token_weight_bf16_initial_refresh_enabled &&
                token_weight_bf16 != nullptr) {
                auto* init_with_shadow =
                    legacy_mod17_token_weight_init_enabled
                        ? init_gpt2_token_weight_with_bf16_shadow
                        : init_gpt2_token_weight_fast_with_bf16_shadow;
                run(init_with_shadow(
                        token_weight,
                        token_weight_bf16,
                        kTokenWeightElements,
                        nullptr),
                    "token_weight.init_device_with_bf16_shadow");
                if (error.empty()) {
                    token_weight_bf16_refresh_count += 1;
                    token_weight_bf16_initial_refresh_elided = true;
                }
            } else {
                auto* init_without_shadow =
                    legacy_mod17_token_weight_init_enabled
                        ? init_gpt2_token_weight
                        : init_gpt2_token_weight_fast;
                run(init_without_shadow(token_weight, kTokenWeightElements, nullptr),
                    "token_weight.init_device");
            }
        }
    });
    auto refresh_token_weight_bf16 = [&](const std::string& name) {
        if (!token_weight_bf16_shadow_enabled || token_weight_bf16 == nullptr || !error.empty()) {
            return;
        }
        run(float32_to_bf16_bits(token_weight, token_weight_bf16, kTokenWeightElements, nullptr), name);
        if (error.empty()) {
            token_weight_bf16_refresh_count += 1;
        }
    };
    run_setup_timed("setup.token_weight_bf16_initial_refresh", [&]() {
        if (token_weight_bf16_initial_refresh_elided) {
            return;
        }
        refresh_token_weight_bf16("token_weight_bf16.initial_refresh");
    });
    run_setup_timed("setup.nonzero_parameter_fill", [&]() {
        if (error.empty() && parameter_fill_descriptor_count > 0) {
            run(fill_many_values(
                    parameter_fill_ptrs,
                    parameter_fill_elements,
                    parameter_fill_values,
                    parameter_fill_descriptor_count,
                    parameter_fill_max_elements,
                    nullptr),
                "nonzero_parameters.fill_many_values");
            if (error.empty()) {
                parameter_fill_kernel_launches += 1;
            }
        }
        if (error.empty() && bf16_parameter_fill_descriptor_count > 0) {
            run(fill_many_values_bf16_bits(
                    bf16_parameter_fill_ptrs,
                    bf16_parameter_fill_elements,
                    bf16_parameter_fill_values,
                    bf16_parameter_fill_descriptor_count,
                    bf16_parameter_fill_max_elements,
                    nullptr),
                "nonzero_bf16_parameters.fill_many_values");
            if (error.empty()) {
                bf16_parameter_fill_kernel_launches += 1;
            }
        }
    });
    auto refresh_block_weight_bf16 = [&](const std::string& name) {
        if (!error.empty()) {
            return;
        }
        run(float32_to_bf16_bits_many(
                block_weight_bf16_sources,
                block_weight_bf16_elements,
                block_weight_bf16_offsets,
                block_weight_bf16_arena,
                block_weight_bf16_descriptor_count,
                block_weight_bf16_max_elements,
                nullptr),
            name);
        if (error.empty()) {
            block_weight_bf16_refresh_count += 1;
        }
    };
    run_setup_timed("setup.block_weight_bf16_initial_refresh", [&]() {
        if (direct_bf16_block_weight_init_enabled) {
            return;
        }
        refresh_block_weight_bf16("block_weight_bf16.initial_refresh");
    });

    const std::int64_t eval_batch_size =
        cfg.eval_batch_size > 0 ? static_cast<std::int64_t>(cfg.eval_batch_size) : batch_size;
    if (error.empty() && (eval_batch_size <= 0 || eval_batch_size > batch_size)) {
        std::ostringstream out;
        out << "GPT transformer/LM eval batch size must be in [1, " << batch_size
            << "] for the current fixed-size activation arena; got " << eval_batch_size;
        error = out.str();
    }
    std::int64_t active_batch_size = batch_size;
    std::int64_t active_rows = rows;
    std::int64_t active_activation_elements = activation_elements;
    std::int64_t active_hidden_elements = hidden_elements;
    std::int64_t active_qkv_activation_elements = qkv_activation_elements;
    std::int64_t* active_targets = targets;
    std::uint16_t* active_targets_u16 = targets_u16;
    std::uint16_t* active_targets_pinned = targets_pinned;
    auto set_active_batch_size = [&](std::int64_t value) {
        if (!error.empty()) {
            return;
        }
        if (value <= 0 || value > batch_size) {
            std::ostringstream out;
            out << "active GPT transformer/LM batch size must be in [1, " << batch_size << "]; got " << value;
            error = out.str();
            return;
        }
        active_batch_size = value;
        active_rows = active_batch_size * seq_len;
        active_activation_elements = active_rows * kDim;
        active_hidden_elements = active_rows * kHidden;
        active_qkv_activation_elements = active_rows * kQkvDim;
        active_targets = token_i64_arena + active_rows;
        active_targets_u16 = token_u16_device_arena + active_rows;
        active_targets_pinned = token_u16_pinned_arena + active_rows;
    };
    set_active_batch_size(batch_size);

    neuralfn::native_train::SequentialTokenBatchSampler sampler(dataset.train_shards, seq_len, batch_size);
    std::vector<float> host_loss(1, 0.0f);
    std::vector<ValidationLossRecord> validation_losses;
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

    auto run_layer_norm = [&](
                              const float* input,
                              const float* weight,
                              const float* bias,
                              float* output,
                              float* mean,
                              float* rstd,
                              const std::string& name) {
        if (!error.empty()) {
            return;
        }
        if (layer_norm_stats_enabled) {
            run(layer_norm_with_stats(input, weight, bias, output, mean, rstd, active_rows, kDim, kNormEps, nullptr), name);
        } else {
            run(layer_norm(input, weight, bias, output, active_rows, kDim, kNormEps, nullptr), name);
        }
    };
    auto run_layer_norm_bf16_out = [&](
                                       const float* input,
                                       const float* weight,
                                       const float* bias,
                                       float* output,
                                       float* mean,
                                       float* rstd,
                                       std::uint16_t* output_bf16,
                                       const std::string& name) {
        if (!error.empty()) {
            return;
        }
        if (output_bf16 != nullptr && layer_norm_stats_enabled && layer_norm_with_stats_bf16_out != nullptr) {
            run(layer_norm_with_stats_bf16_out(
                    input, weight, bias, output, mean, rstd, output_bf16, active_rows, kDim, kNormEps, nullptr),
                name);
        } else {
            run_layer_norm(input, weight, bias, output, mean, rstd, name);
            if (output_bf16 != nullptr && error.empty()) {
                run(float32_to_bf16_bits(output, output_bf16, active_activation_elements, nullptr),
                    name + ".store_bf16");
            }
        }
    };
    auto run_layer_norm_apply_stats_bf16_out = [&](
                                                   const float* input,
                                                   const float* weight,
                                                   const float* bias,
                                                   const float* mean,
                                                   const float* rstd,
                                                   std::uint16_t* output_bf16,
                                                   const std::string& name) {
        if (!error.empty()) {
            return;
        }
        if (layer_norm_apply_stats_bf16_out == nullptr) {
            error = "required Tile CUDA symbol nfn_native_tile_layer_norm_apply_stats_bf16_out_float32 is unavailable";
            return;
        }
        run(layer_norm_apply_stats_bf16_out(
                input, weight, bias, mean, rstd, output_bf16, active_rows, kDim, nullptr),
            name);
    };
    auto run_layer_norm_backward_affine_accumulate = [&](
                                                        const float* input,
                                                        const float* grad_out,
                                                        const float* mean,
                                                        const float* rstd,
                                                        float* grad_weight,
                                                        float* grad_bias,
                                                        const std::string& name) {
        if (!error.empty()) {
            return;
        }
        if (layer_norm_stats_enabled) {
            run(layer_norm_backward_affine_accumulate_with_stats(
                    input, grad_out, mean, rstd, grad_weight, grad_bias, active_rows, kDim, nullptr),
                name);
        } else {
            run(layer_norm_backward_affine_accumulate(
                    input, grad_out, grad_weight, grad_bias, active_rows, kDim, kNormEps, nullptr),
                name);
        }
    };
    auto run_layer_norm_backward_affine_accumulate_bf16_bits = [&](
                                                                  const std::uint16_t* input_bf16_bits,
                                                                  const float* grad_out,
                                                                  const float* mean,
                                                                  const float* rstd,
                                                                  float* grad_weight,
                                                                  float* grad_bias,
                                                                  const std::string& name) {
        if (!error.empty()) {
            return;
        }
        run(layer_norm_backward_affine_accumulate_with_stats_bf16_bits(
                input_bf16_bits, grad_out, mean, rstd, grad_weight, grad_bias, active_rows, kDim, nullptr),
            name);
    };
    std::int64_t layer_norm_backward_affine_residual_fused_kernel_launches = 0;
    auto run_layer_norm_backward_affine_residual_add_accumulate = [&](
                                                                      const float* input,
                                                                      const float* grad_out,
                                                                      const float* weight,
                                                                      const float* mean,
                                                                      const float* rstd,
                                                                      const float* residual_grad,
                                                                      const float* residual_scale_ptr,
                                                                      float* out,
                                                                      float* grad_weight,
                                                                      float* grad_bias,
                                                                      const std::string& name) {
        if (!error.empty()) {
            return;
        }
        run(layer_norm_backward_affine_residual_add_accumulate_with_stats(
                input,
                grad_out,
                weight,
                mean,
                rstd,
                residual_grad,
                residual_scale_ptr,
                out,
                grad_weight,
                grad_bias,
                active_rows,
                kDim,
                nullptr),
            name);
        if (error.empty()) {
            layer_norm_backward_affine_residual_fused_kernel_launches += 1;
        }
    };
    auto run_layer_norm_backward_affine_residual_add_accumulate_bf16_bits = [&](
                                                                                const std::uint16_t* input_bf16_bits,
                                                                                const float* grad_out,
                                                                                const float* weight,
                                                                                const float* mean,
                                                                                const float* rstd,
                                                                                const float* residual_grad,
                                                                                const float* residual_scale_ptr,
                                                                                float* out,
                                                                                float* grad_weight,
                                                                                float* grad_bias,
                                                                                const std::string& name) {
        if (!error.empty()) {
            return;
        }
        run(layer_norm_backward_affine_residual_add_accumulate_with_stats_bf16_bits(
                input_bf16_bits,
                grad_out,
                weight,
                mean,
                rstd,
                residual_grad,
                residual_scale_ptr,
                out,
                grad_weight,
                grad_bias,
                active_rows,
                kDim,
                nullptr),
            name);
        if (error.empty()) {
            layer_norm_backward_affine_residual_fused_kernel_launches += 1;
        }
    };
    auto run_layer_norm_backward_input = [&](
                                             const float* input,
                                             const float* grad_out,
                                             const float* weight,
                                             const float* mean,
                                             const float* rstd,
                                             float* grad_input,
                                             const std::string& name) {
        if (!error.empty()) {
            return;
        }
        if (layer_norm_stats_enabled) {
            run(layer_norm_backward_input_with_stats(
                    input, grad_out, weight, mean, rstd, grad_input, active_rows, kDim, nullptr),
                name);
        } else {
            run(layer_norm_backward_input(input, grad_out, weight, grad_input, active_rows, kDim, kNormEps, nullptr), name);
        }
    };

    auto zero_gradients = [&]() {};

    auto zero_accumulated_gradients = [&]() {
        const std::int64_t stage_event = stage_begin("gradient_zero");
        if (error.empty()) {
            bool zeroed_with_cuda_memset = false;
            if (gradient_cuda_memset_zero_enabled &&
                cuda_memset_async != nullptr &&
                !gradient_zero_ranges.empty()) {
                bool all_memsets_queued = true;
                for (const FloatZeroRange& range : gradient_zero_ranges) {
                    const std::size_t bytes =
                        sizeof(float) * static_cast<std::size_t>(range.elements);
                    const int status = cuda_memset_async(range.ptr, 0, bytes, nullptr);
                    if (status != 0) {
                        all_memsets_queued = false;
                        break;
                    }
                    gradient_zero_cuda_memset_count += 1;
                }
                zeroed_with_cuda_memset = all_memsets_queued;
                if (zeroed_with_cuda_memset) {
                    accumulation_zero_kernel_launches +=
                        static_cast<std::int64_t>(gradient_zero_ranges.size());
                }
            }
            if (!zeroed_with_cuda_memset) {
                fill_adamw_many(adamw_grad_ptrs, 0.0f, "accumulated_gradients.zero_many");
                if (error.empty()) {
                    accumulation_zero_kernel_launches += 1;
                    gradient_zero_tile_fill_count += 1;
                }
            }
            if (error.empty() &&
                bf16_block_dweight_staging_enabled &&
                block_dweight_bf16_staging_arena != nullptr &&
                block_dweight_bf16_staging_elements > 0) {
                const std::size_t bytes =
                    sizeof(std::uint16_t) * static_cast<std::size_t>(block_dweight_bf16_staging_elements);
                if (cuda_memset_async == nullptr) {
                    error = "CUDA memset is required for BF16 block dWeight staging zero";
                } else {
                    const int status = cuda_memset_async(block_dweight_bf16_staging_arena, 0, bytes, nullptr);
                    if (status != 0) {
                        error = cuda_error(status, "cudaMemsetAsync block_dweight_bf16_staging_arena");
                    } else {
                        accumulation_zero_kernel_launches += 1;
                        block_dweight_bf16_staging_zero_count += 1;
                    }
                }
            }
        }
        stage_end(stage_event, "gradient_zero");
    };

    auto accumulate_gradients = [&](float scale) {
        (void)scale;
    };

    auto flush_bf16_block_dweight_staging = [&]() {
        if (!bf16_block_dweight_staging_enabled ||
            block_dweight_bf16_staging_arena == nullptr ||
            block_dweight_bf16_staging_elements <= 0) {
            return;
        }
        if (bf16_block_weight_param_update_enabled &&
            adamw_bf16_param_bf16_grad_descriptor_count > 0) {
            return;
        }
        const std::int64_t stage_event = stage_begin("block_dweight_bf16_staging.flush_to_float32");
        for (TransformerBlockParams& block : blocks) {
            if (!error.empty()) {
                break;
            }
            if (block.accum_grad_qkv_weight_bf16 != nullptr) {
                run(bf16_bits_to_float32(
                        block.accum_grad_qkv_weight_bf16,
                        block.accum_grad_qkv_weight,
                        kQkvWeightElements,
                        nullptr),
                    "block_dweight_bf16_staging.qkv.to_float32");
                if (error.empty()) {
                    block_dweight_bf16_staging_convert_kernel_launches += 1;
                }
            }
            if (!error.empty()) {
                break;
            }
            if (block.accum_grad_fc_weight_bf16 != nullptr) {
                run(bf16_bits_to_float32(
                        block.accum_grad_fc_weight_bf16,
                        block.accum_grad_fc_weight,
                        kFcWeightElements,
                        nullptr),
                    "block_dweight_bf16_staging.fc.to_float32");
                if (error.empty()) {
                    block_dweight_bf16_staging_convert_kernel_launches += 1;
                }
            }
        }
        stage_end(stage_event, "block_dweight_bf16_staging.flush_to_float32");
    };

    auto clip_gradients = [&]() {
        const std::int64_t stage_event = stage_begin("gradient_clip");
        if (error.empty() && bf16_block_weight_param_update_enabled) {
            if (adamw_float_update_descriptor_count > 0) {
                run(sumsq_partials_many(
                        reinterpret_cast<const float* const*>(adamw_float_update_grad_ptrs),
                        adamw_float_update_elements,
                        adamw_float_update_partial_offsets,
                        grad_sumsq_partials,
                        adamw_float_update_descriptor_count,
                        adamw_float_update_max_elements,
                        nullptr),
                    "accumulated_gradients.float_params.sumsq_partials_many");
                if (error.empty()) {
                    gradient_sumsq_kernel_launches += 1;
                }
            }
            if (error.empty() && adamw_bf16_param_descriptor_count > 0) {
                run(sumsq_partials_many(
                        reinterpret_cast<const float* const*>(adamw_bf16_param_grad_ptrs),
                        adamw_bf16_param_elements,
                        adamw_bf16_param_partial_offsets,
                        grad_sumsq_partials,
                        adamw_bf16_param_descriptor_count,
                        adamw_bf16_param_max_elements,
                        nullptr),
                    "accumulated_gradients.bf16_params_float_grads.sumsq_partials_many");
                if (error.empty()) {
                    gradient_sumsq_kernel_launches += 1;
                }
            }
            if (error.empty() && adamw_bf16_param_bf16_grad_descriptor_count > 0) {
                run(sumsq_partials_many_bf16_bits(
                        reinterpret_cast<const std::uint16_t* const*>(adamw_bf16_param_bf16_grad_grad_ptrs),
                        adamw_bf16_param_bf16_grad_elements,
                        adamw_bf16_param_bf16_grad_partial_offsets,
                        grad_sumsq_partials,
                        adamw_bf16_param_bf16_grad_descriptor_count,
                        adamw_bf16_param_bf16_grad_max_elements,
                        nullptr),
                    "accumulated_gradients.bf16_params_bf16_grads.sumsq_partials_many");
                if (error.empty()) {
                    gradient_sumsq_kernel_launches += 1;
                }
            }
        } else if (error.empty()) {
            run(sumsq_partials_many(
                    reinterpret_cast<const float* const*>(adamw_grad_ptrs),
                    adamw_elements,
                    gradient_partial_offsets,
                    grad_sumsq_partials,
                    adamw_descriptor_count,
                    adamw_max_elements,
                    nullptr),
                "accumulated_gradients.sumsq_partials_many");
            if (error.empty()) {
                gradient_sumsq_kernel_launches += 1;
            }
        }
        if (error.empty()) {
            run(clip_scale(grad_sumsq_partials, grad_clip_scale, gradient_partial_count, kGradClipNorm, kClipEps, nullptr),
                "gradient_clip_scale");
        }
        stage_end(stage_event, "gradient_clip");
    };

    auto upload_pinned_batch = [&]() {
        if (!error.empty()) {
            return;
        }
        const std::int64_t stage_event = stage_begin("token_upload");
        const std::size_t bytes = sizeof(std::uint16_t) * static_cast<std::size_t>(active_rows);
        const std::size_t arena_bytes = bytes * 2;
        run(cuda_memcpy_async(
                token_u16_device_arena,
                token_u16_pinned_arena,
                arena_bytes,
                kCudaMemcpyHostToDevice,
                nullptr),
            "token_u16_arena.copy_async");
        if (error.empty() && !direct_u16_token_ids_enabled) {
            run(uint16_to_int64(token_u16_device_arena, token_i64_arena, active_rows * 2, nullptr),
                "token_i64_arena.device_widen");
        }
        stage_end(stage_event, "token_upload");
    };

    auto lm_head_forward_loss = [&](const std::string& label) -> double {
        const std::int64_t stage_event = stage_begin(label + ".lm_head_loss");
        fill_buffer(loss_total, 1, 0.0f, label + ".loss_total.zero");
        for (std::int64_t row_start = 0; row_start < active_rows && error.empty(); row_start += lm_head_chunk_rows) {
            const std::int64_t row_count =
                (row_start + lm_head_chunk_rows < active_rows) ? lm_head_chunk_rows : (active_rows - row_start);
            const float* hidden_chunk = lnf_out + row_start * kDim;
            const std::int64_t* target_chunk = active_targets + row_start;
            const std::uint16_t* target_chunk_u16 = active_targets_u16 + row_start;
            if (lm_head_bf16_loss_enabled) {
                if (token_weight_bf16_shadow_enabled && token_weight_bf16 != nullptr) {
                    run(linear_weight_bf16_output(
                            hidden_chunk,
                            token_weight_bf16,
                            nullptr,
                            lm_head_bf16_logits,
                            row_count,
                            kDim,
                            kPaddedVocab,
                            false,
                            nullptr),
                        label + ".lm_head.forward.bf16_shadow_logits");
                } else {
                    run(linear_bf16_output(
                            hidden_chunk,
                            token_weight,
                            nullptr,
                            lm_head_bf16_logits,
                            row_count,
                            kDim,
                            kPaddedVocab,
                            false,
                            nullptr),
                        label + ".lm_head.forward.bf16_logits");
                }
            } else {
                run(linear(hidden_chunk, token_weight, nullptr, logits, row_count, kDim, kPaddedVocab, false, nullptr),
                    label + ".lm_head.forward");
            }
            if (error.empty()) {
                if (lm_head_bf16_loss_enabled) {
                    if (lm_head_public_vocab_ce_enabled) {
                        if (direct_u16_token_ids_enabled) {
                            run(ce_partials_strided_bf16_bits_u16_targets(
                                    lm_head_bf16_logits,
                                    target_chunk_u16,
                                    loss_partials,
                                    row_count,
                                    kVocab,
                                    kPaddedVocab,
                                    nullptr),
                                label + ".ce.forward.public_vocab_strided_bf16_bits_u16_targets");
                        } else {
                            run(ce_partials_strided_bf16_bits(
                                    lm_head_bf16_logits,
                                    target_chunk,
                                    loss_partials,
                                    row_count,
                                    kVocab,
                                    kPaddedVocab,
                                    nullptr),
                                label + ".ce.forward.public_vocab_strided_bf16_bits");
                        }
                    } else {
                        run(ce_partials_bf16_bits(
                                lm_head_bf16_logits,
                                target_chunk,
                                loss_partials,
                                row_count,
                                kPaddedVocab,
                                nullptr),
                            label + ".ce.forward.padded_vocab_bf16_bits");
                    }
                } else {
                    if (lm_head_public_vocab_ce_enabled) {
                        run(ce_partials_strided(
                                logits,
                                target_chunk,
                                loss_partials,
                                row_count,
                                kVocab,
                                kPaddedVocab,
                                nullptr),
                            label + ".ce.forward.public_vocab_strided");
                    } else {
                        run(ce_partials(logits, target_chunk, loss_partials, row_count, kPaddedVocab, nullptr),
                            label + ".ce.forward.padded_vocab");
                    }
                }
            }
            if (error.empty()) {
                const std::int64_t chunk_loss_partials = partial_count_for(row_count);
                const float* current = loss_partials;
                std::int64_t current_count = chunk_loss_partials;
                float* next = loss_reduce_a;
                while (current_count > 1 && error.empty()) {
                    run(sum_partials(current, next, current_count, nullptr), label + ".loss.sum_partials");
                    current = next;
                    current_count = partial_count_for(current_count);
                    next = (next == loss_reduce_a) ? loss_reduce_b : loss_reduce_a;
                }
                if (error.empty()) {
                    run(gradient_accumulate(loss_total, current, 1, 1.0f, nullptr), label + ".loss.accumulate");
                }
            }
        }
        if (error.empty()) {
            run(cuda_device_synchronize(), label + ".cudaDeviceSynchronize");
        }
        if (error.empty()) {
            run(cuda_memcpy(host_loss.data(), loss_total, sizeof(float), kCudaMemcpyDeviceToHost),
                label + ".loss.copy");
        }
        stage_end(stage_event, label + ".lm_head_loss");
        return error.empty() ? static_cast<double>(host_loss[0]) : 0.0;
    };

    auto lm_head_backward = [&](float accumulation_scale, bool dweight_accumulate) {
        const std::int64_t stage_event = stage_begin("lm_head_backward");
        const float dweight_beta =
            dweight_first_microbatch_beta_zero_enabled && !dweight_accumulate ? 0.0f : 1.0f;
        if (lm_head_prepack_bf16_hidden_enabled) {
            run_timed_stage("lm_head_backward.hidden_prepack", [&]() {
                run(float32_to_bf16_bits(
                        lnf_out,
                        lm_head_bf16_hidden,
                        active_rows * kDim,
                        nullptr),
                    "lm_head.hidden_full.to_bf16_bits");
            });
        }
        for (std::int64_t row_start = 0; row_start < active_rows && error.empty(); row_start += lm_head_chunk_rows) {
            const std::int64_t row_count =
                (row_start + lm_head_chunk_rows < active_rows) ? lm_head_chunk_rows : (active_rows - row_start);
            const float* hidden_chunk = lnf_out + row_start * kDim;
            const std::uint16_t* hidden_bf16_chunk =
                lm_head_prepack_bf16_hidden_enabled ? (lm_head_bf16_hidden + row_start * kDim) : lm_head_bf16_hidden;
            const std::int64_t* target_chunk = active_targets + row_start;
            const std::uint16_t* target_chunk_u16 = active_targets_u16 + row_start;
            float* grad_hidden_chunk = grad_lnf + row_start * kDim;
            run_timed_stage("lm_head_backward.logits", [&]() {
                if (lm_head_bf16_logits_enabled) {
                    if (lm_head_prepack_bf16_hidden_enabled) {
                        if (token_weight_bf16_shadow_enabled && token_weight_bf16 != nullptr) {
                            run(linear_bf16_input_weight_bf16_output(
                                    hidden_bf16_chunk,
                                    token_weight_bf16,
                                    nullptr,
                                    lm_head_bf16_logits,
                                    row_count,
                                    kDim,
                                    kPaddedVocab,
                                    false,
                                    nullptr),
                                "lm_head.forward.recompute.prepacked_hidden_bf16_shadow_logits");
                        } else {
                            run(linear_bf16_input_float_weight_bf16_output(
                                    hidden_bf16_chunk,
                                    token_weight,
                                    nullptr,
                                    lm_head_bf16_logits,
                                    row_count,
                                    kDim,
                                    kPaddedVocab,
                                    false,
                                    nullptr),
                                "lm_head.forward.recompute.prepacked_hidden_bf16_logits");
                        }
                    } else {
                        if (token_weight_bf16_shadow_enabled && token_weight_bf16 != nullptr) {
                            run(linear_weight_bf16_output(
                                    hidden_chunk,
                                    token_weight_bf16,
                                    nullptr,
                                    lm_head_bf16_logits,
                                    row_count,
                                    kDim,
                                    kPaddedVocab,
                                    false,
                                    nullptr),
                                "lm_head.forward.recompute.bf16_shadow_logits");
                        } else {
                            run(linear_bf16_output(
                                    hidden_chunk,
                                    token_weight,
                                    nullptr,
                                    lm_head_bf16_logits,
                                    row_count,
                                    kDim,
                                    kPaddedVocab,
                                    false,
                                    nullptr),
                                "lm_head.forward.recompute.bf16_logits");
                        }
                    }
                } else {
                    run(linear(hidden_chunk, token_weight, nullptr, logits, row_count, kDim, kPaddedVocab, false, nullptr),
                        "lm_head.forward.recompute");
                }
            });
            run_timed_stage("lm_head_backward.ce", [&]() {
                if (error.empty()) {
                    if (lm_head_bf16_logits_enabled) {
                        if (lm_head_public_vocab_ce_enabled) {
                            if (direct_u16_token_ids_enabled) {
                                run(ce_backward_inplace_strided_bf16_bits_u16_targets_workspace(
                                        lm_head_bf16_logits,
                                        target_chunk_u16,
                                        row_max,
                                        row_denom,
                                        row_count,
                                        kVocab,
                                        kPaddedVocab,
                                        accumulation_scale / static_cast<float>(active_rows),
                                        nullptr),
                                    "ce.backward.inplace.public_vocab_strided_bf16_bits_u16_targets");
                            } else {
                                run(ce_backward_inplace_strided_bf16_bits_workspace(
                                        lm_head_bf16_logits,
                                        target_chunk,
                                        row_max,
                                        row_denom,
                                        row_count,
                                        kVocab,
                                        kPaddedVocab,
                                        accumulation_scale / static_cast<float>(active_rows),
                                        nullptr),
                                    "ce.backward.inplace.public_vocab_strided_bf16_bits");
                            }
                        } else {
                            run(ce_backward_inplace_bf16_bits_workspace(
                                    lm_head_bf16_logits,
                                    target_chunk,
                                    row_max,
                                    row_denom,
                                    row_count,
                                    kPaddedVocab,
                                    accumulation_scale / static_cast<float>(active_rows),
                                    nullptr),
                                "ce.backward.inplace.padded_vocab_bf16_bits");
                        }
                    } else {
                        if (lm_head_public_vocab_ce_enabled) {
                            run(ce_backward_inplace_strided_workspace(
                                    logits,
                                    target_chunk,
                                    row_max,
                                    row_denom,
                                    row_count,
                                    kVocab,
                                    kPaddedVocab,
                                    accumulation_scale / static_cast<float>(active_rows),
                                    nullptr),
                                "ce.backward.inplace.public_vocab_strided");
                        } else {
                            run(ce_backward_inplace_workspace(
                                    logits,
                                    target_chunk,
                                    row_max,
                                    row_denom,
                                    row_count,
                                    kPaddedVocab,
                                    accumulation_scale / static_cast<float>(active_rows),
                                    nullptr),
                                "ce.backward.inplace.padded_vocab");
                        }
                    }
                }
            });
            run_timed_stage("lm_head_backward.dhidden", [&]() {
                if (error.empty()) {
                    if (lm_head_bf16_logits_enabled) {
                        if (token_weight_bf16_shadow_enabled && token_weight_bf16 != nullptr) {
                            run(linear_backward_input_bf16_bits_weight_bf16(
                                    lm_head_bf16_logits,
                                    token_weight_bf16,
                                    grad_hidden_chunk,
                                    row_count,
                                    kDim,
                                    kPaddedVocab,
                                    nullptr),
                                "lm_head.backward_input.bf16_bits_weight_bf16_shadow");
                        } else {
                            run(linear_backward_input_bf16_bits(
                                    lm_head_bf16_logits,
                                    token_weight,
                                    grad_hidden_chunk,
                                    row_count,
                                    kDim,
                                    kPaddedVocab,
                                    nullptr),
                                "lm_head.backward_input.bf16_bits");
                        }
                    } else {
                        run(linear_backward_input(
                                logits, token_weight, grad_hidden_chunk, row_count, kDim, kPaddedVocab, nullptr),
                            "lm_head.backward_input");
                    }
                }
            });
            run_timed_stage("lm_head_backward.dweight", [&]() {
                if (error.empty()) {
                    if (lm_head_bf16_logits_enabled) {
                        if (lm_head_prepack_bf16_hidden_enabled) {
                            auto* dweight_fn = dweight_first_microbatch_beta_zero_enabled
                                ? linear_backward_weight_accumulate_bf16_bits_bf16_bits_beta
                                : nullptr;
                            if (dweight_fn != nullptr) {
                                run(dweight_fn(
                                        hidden_bf16_chunk,
                                        lm_head_bf16_logits,
                                        accum_grad_token_weight,
                                        row_count,
                                        kDim,
                                        kPaddedVocab,
                                        dweight_beta,
                                        nullptr),
                                    "lm_head.backward_weight.beta.prepacked_bf16_bits_bf16_bits");
                            } else {
                                run(linear_backward_weight_accumulate_bf16_bits_bf16_bits(
                                    hidden_bf16_chunk,
                                    lm_head_bf16_logits,
                                    accum_grad_token_weight,
                                    row_count,
                                    kDim,
                                    kPaddedVocab,
                                    nullptr),
                                    "lm_head.backward_weight.accumulate.prepacked_bf16_bits_bf16_bits");
                            }
                        } else if (lm_head_bf16_dweight_enabled) {
                            run(float32_to_bf16_bits(
                                    hidden_chunk,
                                    lm_head_bf16_hidden,
                                    row_count * kDim,
                                    nullptr),
                                "lm_head.hidden.to_bf16_bits");
                            if (error.empty()) {
                                auto* dweight_fn = dweight_first_microbatch_beta_zero_enabled
                                    ? linear_backward_weight_accumulate_bf16_bits_bf16_bits_beta
                                    : nullptr;
                                if (dweight_fn != nullptr) {
                                    run(dweight_fn(
                                            lm_head_bf16_hidden,
                                            lm_head_bf16_logits,
                                            accum_grad_token_weight,
                                            row_count,
                                            kDim,
                                            kPaddedVocab,
                                            dweight_beta,
                                            nullptr),
                                        "lm_head.backward_weight.beta.bf16_bits_bf16_bits");
                                } else {
                                    run(linear_backward_weight_accumulate_bf16_bits_bf16_bits(
                                            lm_head_bf16_hidden,
                                            lm_head_bf16_logits,
                                            accum_grad_token_weight,
                                            row_count,
                                            kDim,
                                            kPaddedVocab,
                                            nullptr),
                                        "lm_head.backward_weight.accumulate.bf16_bits_bf16_bits");
                                }
                            }
                        } else {
                            run(linear_backward_weight_accumulate_float32_bf16_bits(
                                    hidden_chunk,
                                    lm_head_bf16_logits,
                                    accum_grad_token_weight,
                                    row_count,
                                    kDim,
                                    kPaddedVocab,
                                    nullptr),
                                "lm_head.backward_weight.accumulate.bf16_bits");
                        }
                    } else {
                        run(linear_backward_weight_accumulate(
                                hidden_chunk,
                                logits,
                                accum_grad_token_weight,
                                row_count,
                                kDim,
                                kPaddedVocab,
                                nullptr),
                            "lm_head.backward_weight.accumulate");
                    }
                }
            });
        }
        stage_end(stage_event, "lm_head_backward");
    };

    auto block_input_for = [&](std::size_t block_index) -> const float* {
        return block_index == 0 ? x : block_outputs[block_index - 1];
    };
    auto store_mlp_activations = [&](std::size_t block_index, TransformerBlockActivations& tape) {
        if (block_index >= stored_mlp_activations.size()) {
            return;
        }
        StoredMlpActivations& stored = stored_mlp_activations[block_index];
        if (error.empty()) {
            run(store_mlp_activations_bf16(
                    tape.ln2_out,
                    tape.fc_out,
                    tape.act,
                    stored.ln2_out,
                    active_activation_elements,
                    active_hidden_elements,
                    nullptr),
                "block" + std::to_string(block_index) + ".mlp_activation_store.bf16");
            if (error.empty()) stored_mlp_activation_store_kernel_launches += 1;
        }
    };
    auto store_attention_activations = [&](std::size_t block_index) {
        if (block_index >= stored_attention_activations.size()) {
            return;
        }
        StoredAttentionActivations& stored = stored_attention_activations[block_index];
        if (error.empty()) {
            run(store_attention_tk_workspace(
                    stored.q,
                    stored.k,
                    stored.v,
                    stored.o,
                    stored.lse,
                    active_batch_size,
                    kHeads,
                    seq_len,
                    kHeadDim,
                    nullptr),
                "block" + std::to_string(block_index) + ".attention_activation_store.tk_bf16");
            if (error.empty()) stored_attention_store_kernel_launches += 1;
        }
    };
    auto forward_block = [&](TransformerBlockParams& block,
                             TransformerBlockActivations& tape,
                             const float* block_input,
                             const std::string& label,
                             bool compute_final_output,
                             bool compute_mlp_activations,
                             StoredAttentionActivations* fused_attention_store,
                             StoredPackedAttentionActivations* fused_packed_attention_store,
                             StoredMlpActivations* fused_mlp_store,
                             std::uint16_t* fused_residual1_store,
                             float* direct_residual2_output) {
        const std::string stage_name = label.find("recompute") == std::string::npos ? "block_forward" : "block_recompute";
        const std::int64_t stage_event = stage_begin(stage_name);
        bool ln2_precomputed = false;
        bool ln2_bf16_prepacked = false;
        std::uint16_t* active_qkv_bf16 =
            fused_packed_attention_store != nullptr ? fused_packed_attention_store->qkv : tape.qkv_bf16;
        std::uint16_t* active_packed_attn_out_bf16 =
            fused_packed_attention_store != nullptr ? fused_packed_attention_store->o : tape.packed_attn_out_bf16;
        std::uint16_t* active_ln1_out_bf16 = tape.ln1_out_bf16;
        float* active_ln1_mean =
            fused_packed_attention_store != nullptr && fused_packed_attention_store->ln1_mean != nullptr
                ? fused_packed_attention_store->ln1_mean
                : tape.ln1_mean;
        float* active_ln1_rstd =
            fused_packed_attention_store != nullptr && fused_packed_attention_store->ln1_rstd != nullptr
                ? fused_packed_attention_store->ln1_rstd
                : tape.ln1_rstd;
        run_timed_stage(stage_name + ".attention", [&]() {
            run_timed_stage(stage_name + ".attention.ln1", [&]() {
                run_layer_norm_bf16_out(
                    block_input,
                    block.ln1_weight,
                    block.ln1_bias,
                    tape.ln1_out,
                    active_ln1_mean,
                    active_ln1_rstd,
                    ln1_bf16_qkv_forward_enabled ? active_ln1_out_bf16 : nullptr,
                    label + ".ln1.forward");
            });
            run_timed_stage(stage_name + ".attention.qkv", [&]() {
                if (packed_qkv_attention_enabled) {
                    if (error.empty()) {
                        if (ln1_bf16_qkv_forward_enabled &&
                            active_ln1_out_bf16 != nullptr &&
                            linear_bf16_input_weight_bf16_output != nullptr) {
                            run(linear_bf16_input_weight_bf16_output(
                                    active_ln1_out_bf16,
                                    block.qkv_weight_bf16,
                                    fuse_qkv_bias_tk_gemm_enabled ? block.qkv_bias : nullptr,
                                    active_qkv_bf16,
                                    active_rows,
                                    kDim,
                                    kQkvDim,
                                    fuse_qkv_bias_tk_gemm_enabled,
                                    nullptr),
                                fuse_qkv_bias_tk_gemm_enabled
                                    ? label + ".attn.qkv.forward.fused_bias.bf16_ln1_bf16_bits"
                                    : label + ".attn.qkv.forward.no_bias.bf16_ln1_bf16_bits");
                        } else {
                            run(linear_weight_bf16_output(
                                    tape.ln1_out,
                                    block.qkv_weight_bf16,
                                    fuse_qkv_bias_tk_gemm_enabled ? block.qkv_bias : nullptr,
                                    active_qkv_bf16,
                                    active_rows,
                                    kDim,
                                    kQkvDim,
                                    fuse_qkv_bias_tk_gemm_enabled,
                                    nullptr),
                                fuse_qkv_bias_tk_gemm_enabled
                                    ? label + ".attn.qkv.forward.fused_bias.bf16_bits"
                                    : label + ".attn.qkv.forward.no_bias.bf16_bits");
                        }
                    }
                } else {
                    if (error.empty()) run(linear_weight_bf16(tape.ln1_out, block.qkv_weight_bf16, nullptr, tape.qkv, active_rows, kDim, kQkvDim, false, nullptr), label + ".attn.qkv.forward.no_bias.weight_bf16");
                }
            });
            run_timed_stage(stage_name + ".attention.qkv_layout", [&]() {
                if (packed_qkv_attention_enabled) {
                    if (!fuse_qkv_bias_tk_gemm_enabled && error.empty()) {
                        run(bf16_bits_add_bias_inplace(
                                active_qkv_bf16,
                                block.qkv_bias,
                                active_rows,
                                kQkvDim,
                                nullptr),
                            label + ".attn.qkv.bias_add_bf16_bits");
                    }
                } else {
                    if (error.empty()) run(split_qkv_to_heads_add_bias(tape.qkv, block.qkv_bias, tape.q_heads, tape.k_heads, tape.v_heads, active_batch_size, seq_len, kHeads, kHeadDim, nullptr), label + ".attn.qkv.bias_split_to_heads");
                }
            });
            run_timed_stage(stage_name + ".attention.sdpa", [&]() {
                if (packed_qkv_attention_enabled) {
                    if (error.empty()) {
                        if (fused_packed_attention_store != nullptr &&
                            fused_packed_attention_store->lse != nullptr) {
                            run(packed_attention_forward_store_lse(
                                    active_qkv_bf16,
                                    active_packed_attn_out_bf16,
                                    fused_packed_attention_store->lse,
                                    active_batch_size,
                                    kHeads,
                                    kHeads,
                                    seq_len,
                                    seq_len,
                                    kHeadDim,
                                    kHeadDim,
                                    attention_scale,
                                    true,
                                    false,
                                    false,
                                    0,
                                    0,
                                    0,
                                    0,
                                    nullptr),
                                label + ".attn.sdpa.forward_packed_qkv_bf16_store_lse");
                        } else {
                            run(packed_attention_forward(
                                    active_qkv_bf16,
                                    active_packed_attn_out_bf16,
                                    active_batch_size,
                                    kHeads,
                                    kHeads,
                                    seq_len,
                                    seq_len,
                                    kHeadDim,
                                    kHeadDim,
                                    attention_scale,
                                    true,
                                    false,
                                    false,
                                    0,
                                    0,
                                    0,
                                    0,
                                    nullptr),
                                label + ".attn.sdpa.forward_packed_qkv_bf16");
                        }
                        if (error.empty() && fused_packed_attention_store != nullptr) {
                            stored_packed_attention_store_blocks += 1;
                        }
                    }
                } else if (fused_attention_store != nullptr) {
                    if (error.empty()) {
                        run(attention_store_forward_tk(
                                tape.q_heads,
                                tape.k_heads,
                                tape.v_heads,
                                tape.attn_heads,
                                fused_attention_store->q,
                                fused_attention_store->k,
                                fused_attention_store->v,
                                fused_attention_store->o,
                                fused_attention_store->lse,
                                active_batch_size,
                                kHeads,
                                kHeads,
                                seq_len,
                                seq_len,
                                kHeadDim,
                                kHeadDim,
                                attention_scale,
                                true,
                                false,
                                false,
                                0,
                                0,
                                0,
                                0,
                                nullptr),
                            label + ".attn.sdpa.forward_store_tk_bf16");
                        if (error.empty()) stored_attention_store_kernel_launches += 1;
                    }
                } else {
                    if (error.empty()) run(attention(tape.q_heads, tape.k_heads, tape.v_heads, tape.attn_heads, active_activation_elements, kHeads, kHeads, seq_len, seq_len, kHeadDim, kHeadDim, attention_scale, true, false, false, 0, 0, 0, 0, nullptr), label + ".attn.sdpa.forward");
                }
            });
            run_timed_stage(stage_name + ".attention.merge_heads", [&]() {
                if (packed_qkv_attention_enabled) {
                    return;
                } else {
                    if (error.empty()) run(merge_heads(tape.attn_heads, tape.attn_out, active_batch_size, kHeads, seq_len, kHeadDim, nullptr), label + ".attn.merge_heads");
                }
            });
            run_timed_stage(stage_name + ".attention.proj", [&]() {
                if (packed_qkv_attention_enabled) {
                    if (error.empty()) {
                        if (bf16_projection_residual_enabled &&
                            tape.proj_out_bf16 != nullptr &&
                            linear_bf16_input_weight_bf16_output != nullptr) {
                            run(linear_bf16_input_weight_bf16_output(
                                    active_packed_attn_out_bf16,
                                    block.attn_proj_weight_bf16,
                                    nullptr,
                                    tape.proj_out_bf16,
                                    active_rows,
                                    kDim,
                                    kDim,
                                    false,
                                    nullptr),
                                label + ".attn.out.forward.no_bias.packed_o_bf16_bits_to_bf16");
                        } else {
                            run(linear_bf16_input_weight_bf16(
                                    active_packed_attn_out_bf16,
                                    block.attn_proj_weight_bf16,
                                    nullptr,
                                    tape.attn_proj,
                                    active_rows,
                                    kDim,
                                    kDim,
                                    false,
                                    nullptr),
                                label + ".attn.out.forward.no_bias.packed_o_bf16_bits");
                        }
                    }
                } else {
                    if (error.empty()) {
                        if (bf16_projection_residual_enabled &&
                            tape.proj_out_bf16 != nullptr &&
                            linear_weight_bf16_output != nullptr) {
                            run(linear_weight_bf16_output(tape.attn_out, block.attn_proj_weight_bf16, nullptr, tape.proj_out_bf16, active_rows, kDim, kDim, false, nullptr), label + ".attn.out.forward.no_bias.weight_bf16_to_bf16");
                        } else {
                            run(linear_weight_bf16(tape.attn_out, block.attn_proj_weight_bf16, nullptr, tape.attn_proj, active_rows, kDim, kDim, false, nullptr), label + ".attn.out.forward.no_bias.weight_bf16");
                        }
                    }
                }
            });
            run_timed_stage(stage_name + ".attention.residual", [&]() {
                const bool use_bf16_projection_residual =
                    bf16_projection_residual_enabled && tape.proj_out_bf16 != nullptr;
                if (compute_mlp_activations && fuse_attention_residual_ln2_enabled) {
                    if (error.empty()) {
                        float* ln2_mean = fused_mlp_store != nullptr ? fused_mlp_store->ln2_mean : tape.ln2_mean;
                        float* ln2_rstd = fused_mlp_store != nullptr ? fused_mlp_store->ln2_rstd : tape.ln2_rstd;
                        const bool can_fuse_ln2_bf16_out =
                            fused_ln2_bf16_out_enabled &&
                            fused_mlp_store != nullptr &&
                            ((use_bf16_projection_residual &&
                              linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_bf16_norm != nullptr) ||
                             (!use_bf16_projection_residual &&
                              linear_bias_residual_layer_norm_with_stats_bf16_residual_bf16_norm != nullptr));
                        const bool elide_ln2_norm_float_store =
                            can_fuse_ln2_bf16_out &&
                            fused_ln2_bf16_norm_float_store_elision_enabled;
                        float* ln2_norm_out =
                            elide_ln2_norm_float_store ? nullptr : tape.ln2_out;
                        if (fused_residual1_store != nullptr) {
                            if (use_bf16_projection_residual &&
                                can_fuse_ln2_bf16_out) {
                                run(linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_bf16_norm(
                                        block_input,
                                        tape.proj_out_bf16,
                                        block.attn_proj_bias,
                                        residual_scale,
                                        block.ln2_weight,
                                        block.ln2_bias,
                                        tape.residual1,
                                        ln2_norm_out,
                                        ln2_mean,
                                        ln2_rstd,
                                        fused_residual1_store,
                                        fused_mlp_store->ln2_out,
                                        active_rows,
                                        kDim,
                                        kNormEps,
                                        nullptr),
                                    label + ".attn.bias_residual_ln2_bf16_linear_bf16_residual_bf16_norm");
                            } else if (can_fuse_ln2_bf16_out) {
                                run(linear_bias_residual_layer_norm_with_stats_bf16_residual_bf16_norm(
                                        block_input,
                                        tape.attn_proj,
                                        block.attn_proj_bias,
                                        residual_scale,
                                        block.ln2_weight,
                                        block.ln2_bias,
                                        tape.residual1,
                                        ln2_norm_out,
                                        ln2_mean,
                                        ln2_rstd,
                                        fused_residual1_store,
                                        fused_mlp_store->ln2_out,
                                        active_rows,
                                        kDim,
                                        kNormEps,
                                        nullptr),
                                    label + ".attn.bias_residual_ln2_bf16_residual_bf16_norm");
                            } else if (use_bf16_projection_residual &&
                                       linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual != nullptr) {
                                run(linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual(
                                        block_input,
                                        tape.proj_out_bf16,
                                        block.attn_proj_bias,
                                        residual_scale,
                                        block.ln2_weight,
                                        block.ln2_bias,
                                        tape.residual1,
                                        tape.ln2_out,
                                        ln2_mean,
                                        ln2_rstd,
                                        fused_residual1_store,
                                        active_rows,
                                        kDim,
                                        kNormEps,
                                        nullptr),
                                    label + ".attn.bias_residual_ln2_bf16_linear_bf16_residual");
                            } else {
                                run(linear_bias_residual_layer_norm_with_stats_bf16_residual(
                                        block_input,
                                        tape.attn_proj,
                                        block.attn_proj_bias,
                                        residual_scale,
                                        block.ln2_weight,
                                        block.ln2_bias,
                                        tape.residual1,
                                        tape.ln2_out,
                                        ln2_mean,
                                        ln2_rstd,
                                        fused_residual1_store,
                                        active_rows,
                                        kDim,
                                        kNormEps,
                                        nullptr),
                                    label + ".attn.bias_residual_ln2_bf16_residual");
                            }
                            if (error.empty()) {
                                stored_residual1_activation_store_kernel_launches += 1;
                                if (can_fuse_ln2_bf16_out) {
                                    ln2_bf16_prepacked = true;
                                    stored_mlp_ln2_bf16_fused_store_kernel_launches += 1;
                                    if (elide_ln2_norm_float_store) {
                                        stored_mlp_ln2_bf16_float_store_elided_count += 1;
                                        stored_mlp_ln2_bf16_float_store_elided_elements += active_activation_elements;
                                    }
                                }
                            }
                        } else {
                            if (use_bf16_projection_residual &&
                                can_fuse_ln2_bf16_out) {
                                run(linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_bf16_norm(
                                        block_input,
                                        tape.proj_out_bf16,
                                        block.attn_proj_bias,
                                        residual_scale,
                                        block.ln2_weight,
                                        block.ln2_bias,
                                        tape.residual1,
                                        ln2_norm_out,
                                        ln2_mean,
                                        ln2_rstd,
                                        nullptr,
                                        fused_mlp_store->ln2_out,
                                        active_rows,
                                        kDim,
                                        kNormEps,
                                        nullptr),
                                    label + ".attn.bias_residual_ln2_bf16_linear_bf16_norm");
                                if (error.empty()) {
                                    ln2_bf16_prepacked = true;
                                    stored_mlp_ln2_bf16_fused_store_kernel_launches += 1;
                                    if (elide_ln2_norm_float_store) {
                                        stored_mlp_ln2_bf16_float_store_elided_count += 1;
                                        stored_mlp_ln2_bf16_float_store_elided_elements += active_activation_elements;
                                    }
                                }
                            } else if (use_bf16_projection_residual &&
                                       linear_bias_residual_layer_norm_with_stats_bf16_linear != nullptr) {
                                run(linear_bias_residual_layer_norm_with_stats_bf16_linear(
                                        block_input,
                                        tape.proj_out_bf16,
                                        block.attn_proj_bias,
                                        residual_scale,
                                        block.ln2_weight,
                                        block.ln2_bias,
                                        tape.residual1,
                                        tape.ln2_out,
                                        ln2_mean,
                                        ln2_rstd,
                                        active_rows,
                                        kDim,
                                        kNormEps,
                                        nullptr),
                                    label + ".attn.bias_residual_ln2_bf16_linear");
                            } else {
                                run(linear_bias_residual_layer_norm_with_stats(
                                    block_input,
                                    tape.attn_proj,
                                    block.attn_proj_bias,
                                    residual_scale,
                                    block.ln2_weight,
                                    block.ln2_bias,
                                    tape.residual1,
                                    tape.ln2_out,
                                    ln2_mean,
                                    ln2_rstd,
                                    active_rows,
                                    kDim,
                                    kNormEps,
                                    nullptr),
                                    label + ".attn.bias_residual_ln2");
                            }
                        }
                        ln2_precomputed = error.empty();
                    }
                } else {
                    if (error.empty()) {
                        if (use_bf16_projection_residual && linear_bias_residual_add_bf16_linear != nullptr) {
                            run(linear_bias_residual_add_bf16_linear(block_input, tape.proj_out_bf16, block.attn_proj_bias, residual_scale, tape.residual1, active_rows, kDim, nullptr), label + ".attn.bias_residual_bf16_linear");
                        } else {
                            run(linear_bias_residual_add(block_input, tape.attn_proj, block.attn_proj_bias, residual_scale, tape.residual1, active_rows, kDim, nullptr), label + ".attn.bias_residual");
                        }
                    }
                }
            });
        });
        if (compute_mlp_activations) {
            run_timed_stage(stage_name + ".mlp_fc_gelu", [&]() {
                run_timed_stage(stage_name + ".mlp_fc_gelu.ln2", [&]() {
                    if (!ln2_precomputed) {
                        float* ln2_mean = fused_mlp_store != nullptr ? fused_mlp_store->ln2_mean : tape.ln2_mean;
                        float* ln2_rstd = fused_mlp_store != nullptr ? fused_mlp_store->ln2_rstd : tape.ln2_rstd;
                        run_layer_norm(tape.residual1, block.ln2_weight, block.ln2_bias, tape.ln2_out, ln2_mean, ln2_rstd, label + ".ln2.forward");
                    }
                });
                if (fused_mlp_store != nullptr) {
                    run_timed_stage(stage_name + ".mlp_fc_gelu.pack_ln2", [&]() {
                        if (ln2_bf16_prepacked) {
                            return;
                        }
                        if (error.empty()) run(float32_to_bf16_bits(tape.ln2_out, fused_mlp_store->ln2_out, active_activation_elements, nullptr), label + ".mlp.ln2.store_bf16");
                    });
                    run_timed_stage(stage_name + ".mlp_fc_gelu.fc_gelu", [&]() {
                        if (error.empty()) {
                            if (reuse_packed_ln2_fc_gelu_enabled &&
                                linear_bf16_input_weight_bf16_gelu_bf16 != nullptr) {
                                run(linear_bf16_input_weight_bf16_gelu_bf16(
                                        fused_mlp_store->ln2_out,
                                        block.fc_weight_bf16,
                                        block.fc_bias,
                                        fused_mlp_store->fc_out,
                                        fused_mlp_store->act,
                                        active_rows,
                                        kDim,
                                        kHidden,
                                        nullptr),
                                    label + ".mlp.fc_bias_gelu.forward.prepacked_bf16_fused");
                            } else {
                                run(linear_weight_bf16_gelu_bf16(
                                        tape.ln2_out,
                                        block.fc_weight_bf16,
                                        block.fc_bias,
                                        fused_mlp_store->fc_out,
                                        fused_mlp_store->act,
                                        active_rows,
                                        kDim,
                                        kHidden,
                                        nullptr),
                                    label + ".mlp.fc_bias_gelu.forward.bf16_fused");
                            }
                            if (error.empty()) stored_mlp_activation_store_kernel_launches += 1;
                        }
                    });
                } else {
                    run_timed_stage(stage_name + ".mlp_fc_gelu.fc", [&]() {
                        if (error.empty()) run(linear_weight_bf16(tape.ln2_out, block.fc_weight_bf16, nullptr, tape.fc_out, active_rows, kDim, kHidden, false, nullptr), label + ".mlp.fc.forward.no_bias.weight_bf16");
                    });
                    run_timed_stage(stage_name + ".mlp_fc_gelu.gelu", [&]() {
                        if (error.empty()) run(gelu_add_bias_bf16_act(tape.fc_out, block.fc_bias, tape.fc_out, tape.act, mlp_forward_act_bf16, active_rows, kHidden, nullptr), label + ".mlp.bias_gelu.forward.bf16_act");
                    });
                }
            });
        }
        if (compute_final_output) {
            run_timed_stage(stage_name + ".mlp_proj", [&]() {
                run_timed_stage(stage_name + ".mlp_proj.proj", [&]() {
                    const std::uint16_t* act_bits =
                        fused_mlp_store != nullptr ? fused_mlp_store->act : mlp_forward_act_bf16;
                    if (error.empty()) {
                        if (bf16_projection_residual_enabled &&
                            tape.proj_out_bf16 != nullptr &&
                            linear_bf16_input_weight_bf16_output != nullptr) {
                            run(linear_bf16_input_weight_bf16_output(act_bits, block.mlp_proj_weight_bf16, nullptr, tape.proj_out_bf16, active_rows, kHidden, kDim, false, nullptr), label + ".mlp.proj.forward.no_bias.bf16_act_weight_bf16_to_bf16");
                        } else {
                            run(linear_bf16_input_weight_bf16(act_bits, block.mlp_proj_weight_bf16, nullptr, tape.mlp_out, active_rows, kHidden, kDim, false, nullptr), label + ".mlp.proj.forward.no_bias.bf16_act_weight_bf16");
                        }
                    }
                });
                run_timed_stage(stage_name + ".mlp_proj.residual", [&]() {
                    float* residual2_output =
                        direct_residual2_output != nullptr ? direct_residual2_output : tape.residual2;
                    if (error.empty()) {
                        if (bf16_projection_residual_enabled &&
                            tape.proj_out_bf16 != nullptr &&
                            linear_bias_residual_add_bf16_linear != nullptr) {
                            run(linear_bias_residual_add_bf16_linear(tape.residual1, tape.proj_out_bf16, block.mlp_proj_bias, residual_scale, residual2_output, active_rows, kDim, nullptr), label + ".mlp.bias_residual_bf16_linear");
                        } else {
                            run(linear_bias_residual_add(tape.residual1, tape.mlp_out, block.mlp_proj_bias, residual_scale, residual2_output, active_rows, kDim, nullptr), label + ".mlp.bias_residual");
                        }
                    }
                });
            });
        }
        stage_end(stage_event, stage_name);
    };
    auto recompute_block_from_saved_attention = [&](TransformerBlockParams& block,
                                                    TransformerBlockActivations& tape,
                                                    const StoredAttentionActivations& stored_attention,
                                                    const float* block_input,
                                                    bool compute_mlp_activations,
                                                    const std::string& label) {
        const std::string stage_name = "block_recompute_saved_attention";
        const std::int64_t stage_event = stage_begin(stage_name);
        run_timed_stage(stage_name + ".ln1", [&]() {
            run_layer_norm(block_input, block.ln1_weight, block.ln1_bias, tape.ln1_out, tape.ln1_mean, tape.ln1_rstd, label + ".ln1.forward");
        });
        run_timed_stage(stage_name + ".attention.restore_o", [&]() {
            if (error.empty()) run(bf16_bits_to_float32(stored_attention.o, tape.attn_heads, active_activation_elements, nullptr), label + ".attn.restore_o_bf16");
        });
        run_timed_stage(stage_name + ".attention.merge_heads", [&]() {
            if (error.empty()) run(merge_heads(tape.attn_heads, tape.attn_out, active_batch_size, kHeads, seq_len, kHeadDim, nullptr), label + ".attn.merge_heads");
        });
        run_timed_stage(stage_name + ".attention.proj", [&]() {
            if (error.empty()) run(linear_weight_bf16(tape.attn_out, block.attn_proj_weight_bf16, nullptr, tape.attn_proj, active_rows, kDim, kDim, false, nullptr), label + ".attn.out.forward.no_bias.weight_bf16");
        });
        run_timed_stage(stage_name + ".attention.residual", [&]() {
            if (error.empty()) run(linear_bias_residual_add(block_input, tape.attn_proj, block.attn_proj_bias, residual_scale, tape.residual1, active_rows, kDim, nullptr), label + ".attn.bias_residual");
        });
        if (compute_mlp_activations) {
            run_timed_stage(stage_name + ".mlp_fc_gelu", [&]() {
                run_timed_stage(stage_name + ".mlp_fc_gelu.ln2", [&]() {
                    run_layer_norm(tape.residual1, block.ln2_weight, block.ln2_bias, tape.ln2_out, tape.ln2_mean, tape.ln2_rstd, label + ".ln2.forward");
                });
                run_timed_stage(stage_name + ".mlp_fc_gelu.fc", [&]() {
                    if (error.empty()) run(linear_weight_bf16(tape.ln2_out, block.fc_weight_bf16, nullptr, tape.fc_out, active_rows, kDim, kHidden, false, nullptr), label + ".mlp.fc.forward.no_bias.weight_bf16");
                });
                run_timed_stage(stage_name + ".mlp_fc_gelu.gelu", [&]() {
                    if (error.empty()) run(gelu_add_bias_bf16_act(tape.fc_out, block.fc_bias, tape.fc_out, tape.act, mlp_forward_act_bf16, active_rows, kHidden, nullptr), label + ".mlp.bias_gelu.forward.bf16_act");
                });
            });
        }
        stage_end(stage_event, stage_name);
    };
    auto recompute_block_from_saved_packed_attention = [&](TransformerBlockParams& block,
                                                           TransformerBlockActivations& tape,
                                                           const StoredPackedAttentionActivations& stored_attention,
                                                           const std::uint16_t* stored_residual1_bf16,
                                                           const float* block_input,
                                                           bool compute_mlp_activations,
                                                           const std::string& label) {
        const std::string stage_name = "block_recompute_saved_packed_attention";
        const std::int64_t stage_event = stage_begin(stage_name);
        bool ln2_precomputed = false;
        run_timed_stage(stage_name + ".ln1", [&]() {
            if (stored_attention.ln1_mean != nullptr &&
                stored_attention.ln1_rstd != nullptr) {
                run_layer_norm_apply_stats_bf16_out(
                    block_input,
                    block.ln1_weight,
                    block.ln1_bias,
                    stored_attention.ln1_mean,
                    stored_attention.ln1_rstd,
                    tape.ln1_out_bf16,
                    label + ".ln1.apply_stats_bf16");
                return;
            }
            run_layer_norm(
                block_input,
                block.ln1_weight,
                block.ln1_bias,
                tape.ln1_out,
                tape.ln1_mean,
                tape.ln1_rstd,
                label + ".ln1.forward");
        });
        run_timed_stage(stage_name + ".attention.proj", [&]() {
            if (stored_residual1_bf16 != nullptr) {
                return;
            }
            if (error.empty()) {
                run(linear_bf16_input_weight_bf16(
                        stored_attention.o,
                        block.attn_proj_weight_bf16,
                        nullptr,
                        tape.attn_proj,
                        active_rows,
                        kDim,
                        kDim,
                        false,
                        nullptr),
                    label + ".attn.out.forward.no_bias.packed_o_bf16_bits");
            }
        });
        run_timed_stage(stage_name + ".attention.residual", [&]() {
            if (stored_residual1_bf16 != nullptr) {
                const bool can_defer_residual1_restore =
                    bf16_residual1_ln_backward_enabled && !compute_mlp_activations;
                if (can_defer_residual1_restore) {
                    return;
                }
                if (error.empty()) {
                    run(bf16_bits_to_float32(
                            stored_residual1_bf16,
                            tape.residual1,
                            active_activation_elements,
                            nullptr),
                        label + ".residual1.restore_bf16");
                    if (error.empty()) {
                        stored_residual1_activation_restore_kernel_launches += 1;
                    }
                }
                return;
            }
            if (compute_mlp_activations && fuse_attention_residual_ln2_enabled) {
                if (error.empty()) {
                    run(linear_bias_residual_layer_norm_with_stats(
                            block_input,
                            tape.attn_proj,
                            block.attn_proj_bias,
                            residual_scale,
                            block.ln2_weight,
                            block.ln2_bias,
                            tape.residual1,
                            tape.ln2_out,
                            tape.ln2_mean,
                            tape.ln2_rstd,
                            active_rows,
                            kDim,
                            kNormEps,
                            nullptr),
                        label + ".attn.bias_residual_ln2");
                    ln2_precomputed = error.empty();
                }
            } else {
                if (error.empty()) {
                    run(linear_bias_residual_add(
                            block_input,
                            tape.attn_proj,
                            block.attn_proj_bias,
                            residual_scale,
                            tape.residual1,
                            active_rows,
                            kDim,
                            nullptr),
                        label + ".attn.bias_residual");
                }
            }
        });
        if (compute_mlp_activations) {
            run_timed_stage(stage_name + ".mlp_fc_gelu", [&]() {
                run_timed_stage(stage_name + ".mlp_fc_gelu.ln2", [&]() {
                    if (!ln2_precomputed) {
                        run_layer_norm(
                            tape.residual1,
                            block.ln2_weight,
                            block.ln2_bias,
                            tape.ln2_out,
                            tape.ln2_mean,
                            tape.ln2_rstd,
                            label + ".ln2.forward");
                    }
                });
                run_timed_stage(stage_name + ".mlp_fc_gelu.fc", [&]() {
                    if (error.empty()) {
                        run(linear_weight_bf16(
                                tape.ln2_out,
                                block.fc_weight_bf16,
                                nullptr,
                                tape.fc_out,
                                active_rows,
                                kDim,
                                kHidden,
                                false,
                                nullptr),
                            label + ".mlp.fc.forward.no_bias.bf16");
                    }
                });
                run_timed_stage(stage_name + ".mlp_fc_gelu.gelu", [&]() {
                    if (error.empty()) {
                        run(gelu_add_bias_bf16_act(
                                tape.fc_out,
                                block.fc_bias,
                                tape.fc_out,
                                tape.act,
                                mlp_forward_act_bf16,
                                active_rows,
                                kHidden,
                                nullptr),
                            label + ".mlp.bias_gelu.forward.bf16_act");
                    }
                });
            });
        }
        stage_end(stage_event, stage_name);
    };
    auto backward_block = [&](TransformerBlockParams& block,
                              TransformerBlockActivations& tape,
                              const StoredMlpActivations* stored_mlp,
                              const StoredAttentionActivations* stored_attention,
                              const StoredPackedAttentionActivations* stored_packed_attention,
                              const std::uint16_t* stored_residual1_bf16,
                              const float* block_input,
                              float* incoming_grad,
                              float* output_grad,
                              bool dweight_accumulate,
                              const std::string& label) {
        const std::int64_t stage_event = stage_begin("block_backward");
        const float dweight_beta =
            dweight_first_microbatch_beta_zero_enabled && !dweight_accumulate ? 0.0f : 1.0f;
        std::uint16_t* active_qkv_bf16 =
            stored_packed_attention != nullptr ? stored_packed_attention->qkv : tape.qkv_bf16;
        std::uint16_t* active_qkv_grad_bf16 =
            direct_bf16_qkv_grad_scratch_enabled &&
                    mlp_forward_act_bf16 != nullptr &&
                    mlp_forward_act_bf16_elements >= qkv_activation_elements
                ? mlp_forward_act_bf16
                : active_qkv_bf16;
        std::uint16_t* active_attention_grad_out_bf16 =
            bf16_attention_grad_out_handoff_enabled ? attention_grad_out_bf16 : nullptr;
        const std::uint16_t* active_packed_attn_out_bf16 =
            stored_packed_attention != nullptr ? stored_packed_attention->o : tape.packed_attn_out_bf16;
        const float* active_ln1_mean =
            stored_packed_attention != nullptr && stored_packed_attention->ln1_mean != nullptr
                ? stored_packed_attention->ln1_mean
                : tape.ln1_mean;
        const float* active_ln1_rstd =
            stored_packed_attention != nullptr && stored_packed_attention->ln1_rstd != nullptr
                ? stored_packed_attention->ln1_rstd
                : tape.ln1_rstd;
        run_timed_stage("block_backward.mlp_proj", [&]() {
            run_timed_stage("block_backward.mlp_proj.dweight_bias", [&]() {
                if (stored_mlp != nullptr) {
                    if (error.empty()) {
                        auto* dweight_fn = dweight_first_microbatch_beta_zero_enabled
                            ? linear_backward_weight_bias_accumulate_bf16_bits_beta
                            : nullptr;
                        if (dweight_fn != nullptr) {
                            run(dweight_fn(
                                    stored_mlp->act,
                                    incoming_grad,
                                    block.accum_grad_mlp_proj_weight,
                                    block.accum_grad_mlp_proj_bias,
                                    active_rows,
                                    kHidden,
                                    kDim,
                                    dweight_beta,
                                    nullptr),
                                label + ".mlp.proj.backward_weight_bias.beta.bf16_bits");
                        } else {
                            run(linear_backward_weight_bias_accumulate_bf16_bits(
                                    stored_mlp->act,
                                    incoming_grad,
                                    block.accum_grad_mlp_proj_weight,
                                    block.accum_grad_mlp_proj_bias,
                                    active_rows,
                                    kHidden,
                                    kDim,
                                    nullptr),
                                label + ".mlp.proj.backward_weight_bias.accumulate.bf16_bits");
                        }
                    }
                } else {
                    if (error.empty()) run(linear_backward_weight_bias_accumulate_bf16(tape.act, incoming_grad, block.accum_grad_mlp_proj_weight, block.accum_grad_mlp_proj_bias, active_rows, kHidden, kDim, nullptr), label + ".mlp.proj.backward_weight_bias.accumulate.bf16");
                }
            });
            run_timed_stage("block_backward.mlp_proj.dinput", [&]() {
                if (stored_mlp != nullptr && fuse_mlp_proj_dgelu_enabled) {
                    if (error.empty()) {
                        if (elide_mlp_dgelu_float_grad_enabled) {
                            run(linear_backward_input_dgelu_weight_bf16_bits_only(
                                    incoming_grad,
                                    block.mlp_proj_weight_bf16,
                                    stored_mlp->fc_out,
                                    mlp_forward_act_bf16,
                                    grad_fc_out,
                                    active_rows,
                                    kHidden,
                                    kDim,
                                    nullptr),
                                label + ".mlp.proj.backward_input_dgelu.bf16_bits_only");
                        } else {
                            run(linear_backward_input_dgelu_weight_bf16_bits(
                                    incoming_grad,
                                    block.mlp_proj_weight_bf16,
                                    stored_mlp->fc_out,
                                    mlp_forward_act_bf16,
                                    grad_fc_out,
                                    active_rows,
                                    kHidden,
                                    kDim,
                                    nullptr),
                                label + ".mlp.proj.backward_input_dgelu.bf16_bits");
                        }
                    }
                } else {
                    if (error.empty()) {
                        run(linear_backward_input_weight_bf16(
                                incoming_grad,
                                block.mlp_proj_weight_bf16,
                                grad_fc_out,
                                active_rows,
                                kHidden,
                                kDim,
                                nullptr),
                            label + ".mlp.proj.backward_input.bf16");
                    }
                }
            });
            run_timed_stage("block_backward.mlp_proj.gelu", [&]() {
                if (stored_mlp != nullptr && fuse_mlp_proj_dgelu_enabled) {
                    return;
                } else if (stored_mlp != nullptr) {
                    if (error.empty()) {
                        run(gelu_backward_inplace_bf16_bits(
                                stored_mlp->fc_out,
                                grad_fc_out,
                                active_hidden_elements,
                                nullptr),
                            label + ".mlp.gelu.backward_inplace.bf16_bits");
                    }
                } else {
                    if (error.empty()) run(gelu_backward_inplace(tape.fc_out, grad_fc_out, active_hidden_elements, nullptr), label + ".mlp.gelu.backward_inplace");
                }
            });
        });
        run_timed_stage("block_backward.mlp_fc", [&]() {
            run_timed_stage("block_backward.mlp_fc.dweight_bias", [&]() {
                if (stored_mlp != nullptr && bf16_mlp_grad_handoff_enabled) {
                    if (error.empty()) {
                        if (bf16_block_dweight_staging_enabled &&
                            block.accum_grad_fc_weight_bf16 != nullptr) {
                            run(linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_to_bf16_bits(
                                    stored_mlp->ln2_out,
                                    mlp_forward_act_bf16,
                                    block.accum_grad_fc_weight_bf16,
                                    block.accum_grad_fc_bias,
                                    active_rows,
                                    kDim,
                                    kHidden,
                                    nullptr),
                                label + ".mlp.fc.backward_weight_bias.accumulate.bf16_bits_bf16_grad_to_bf16");
                        } else {
                            auto* dweight_fn = dweight_first_microbatch_beta_zero_enabled
                                ? linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_beta
                                : nullptr;
                            if (dweight_fn != nullptr) {
                                run(dweight_fn(
                                        stored_mlp->ln2_out,
                                        mlp_forward_act_bf16,
                                        block.accum_grad_fc_weight,
                                        block.accum_grad_fc_bias,
                                        active_rows,
                                        kDim,
                                        kHidden,
                                        dweight_beta,
                                        nullptr),
                                    label + ".mlp.fc.backward_weight_bias.beta.bf16_bits_bf16_grad");
                            } else {
                                run(linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits(
                                        stored_mlp->ln2_out,
                                        mlp_forward_act_bf16,
                                        block.accum_grad_fc_weight,
                                        block.accum_grad_fc_bias,
                                        active_rows,
                                        kDim,
                                        kHidden,
                                        nullptr),
                                    label + ".mlp.fc.backward_weight_bias.accumulate.bf16_bits_bf16_grad");
                            }
                        }
                    }
                } else if (stored_mlp != nullptr) {
                    if (error.empty()) run(linear_backward_weight_bias_accumulate_bf16_bits(stored_mlp->ln2_out, grad_fc_out, block.accum_grad_fc_weight, block.accum_grad_fc_bias, active_rows, kDim, kHidden, nullptr), label + ".mlp.fc.backward_weight_bias.accumulate.bf16_bits");
                } else {
                    if (error.empty()) run(linear_backward_weight_bias_accumulate_bf16(tape.ln2_out, grad_fc_out, block.accum_grad_fc_weight, block.accum_grad_fc_bias, active_rows, kDim, kHidden, nullptr), label + ".mlp.fc.backward_weight_bias.accumulate.bf16");
                }
            });
            run_timed_stage("block_backward.mlp_fc.dinput", [&]() {
                if (stored_mlp != nullptr && bf16_mlp_grad_handoff_enabled) {
                    if (error.empty()) {
                        run(linear_backward_input_bf16_bits_weight_bf16(
                                mlp_forward_act_bf16,
                                block.fc_weight_bf16,
                                grad_ln2,
                                active_rows,
                                kDim,
                                kHidden,
                                nullptr),
                            label + ".mlp.fc.backward_input.bf16_bits_weight_bf16");
                    }
                } else {
                    if (error.empty()) run(linear_backward_input_weight_bf16(grad_fc_out, block.fc_weight_bf16, grad_ln2, active_rows, kDim, kHidden, nullptr), label + ".mlp.fc.backward_input.weight_bf16");
                }
            });
        });
        run_timed_stage("block_backward.ln2_residual", [&]() {
            const float* ln2_mean = stored_mlp != nullptr ? stored_mlp->ln2_mean : tape.ln2_mean;
            const float* ln2_rstd = stored_mlp != nullptr ? stored_mlp->ln2_rstd : tape.ln2_rstd;
            const bool use_bf16_residual1_ln_backward =
                bf16_residual1_ln_backward_enabled && stored_residual1_bf16 != nullptr;
            const bool use_fused_ln2_affine_residual =
                fuse_ln_backward_affine_residual_enabled && active_rows > 0 && kDim <= 1024;
            if (use_fused_ln2_affine_residual) {
                run_timed_stage("block_backward.ln2_residual.fused_affine_dinput_add", [&]() {
                    if (use_bf16_residual1_ln_backward) {
                        run_layer_norm_backward_affine_residual_add_accumulate_bf16_bits(
                            stored_residual1_bf16,
                            grad_ln2,
                            block.ln2_weight,
                            ln2_mean,
                            ln2_rstd,
                            incoming_grad,
                            residual_scale,
                            grad_residual1,
                            block.accum_grad_ln2_weight,
                            block.accum_grad_ln2_bias,
                            label + ".ln2.backward_affine_input_residual_add.accumulate.bf16_bits");
                    } else {
                        run_layer_norm_backward_affine_residual_add_accumulate(
                            tape.residual1,
                            grad_ln2,
                            block.ln2_weight,
                            ln2_mean,
                            ln2_rstd,
                            incoming_grad,
                            residual_scale,
                            grad_residual1,
                            block.accum_grad_ln2_weight,
                            block.accum_grad_ln2_bias,
                            label + ".ln2.backward_affine_input_residual_add.accumulate");
                    }
                });
            } else if (fuse_ln_backward_residual_enabled) {
                run_timed_stage("block_backward.ln2_residual.affine", [&]() {
                    if (use_bf16_residual1_ln_backward) {
                        run_layer_norm_backward_affine_accumulate_bf16_bits(
                            stored_residual1_bf16,
                            grad_ln2,
                            ln2_mean,
                            ln2_rstd,
                            block.accum_grad_ln2_weight,
                            block.accum_grad_ln2_bias,
                            label + ".ln2.backward_affine.accumulate.bf16_bits");
                    } else {
                        run_layer_norm_backward_affine_accumulate(
                            tape.residual1,
                            grad_ln2,
                            ln2_mean,
                            ln2_rstd,
                            block.accum_grad_ln2_weight,
                            block.accum_grad_ln2_bias,
                            label + ".ln2.backward_affine.accumulate");
                    }
                });
                run_timed_stage("block_backward.ln2_residual.dinput_add", [&]() {
                    if (error.empty()) {
                        if (use_bf16_residual1_ln_backward) {
                            run(layer_norm_backward_input_residual_add_with_stats_bf16_bits(
                                    stored_residual1_bf16,
                                    grad_ln2,
                                    block.ln2_weight,
                                    ln2_mean,
                                    ln2_rstd,
                                    incoming_grad,
                                    residual_scale,
                                    grad_residual1,
                                    active_rows,
                                    kDim,
                                    nullptr),
                                label + ".ln2.backward_input_residual_add.with_stats.bf16_bits");
                        } else {
                            run(layer_norm_backward_input_residual_add_with_stats(
                                    tape.residual1,
                                    grad_ln2,
                                    block.ln2_weight,
                                    ln2_mean,
                                    ln2_rstd,
                                    incoming_grad,
                                    residual_scale,
                                    grad_residual1,
                                    active_rows,
                                    kDim,
                                    nullptr),
                                label + ".ln2.backward_input_residual_add.with_stats");
                        }
                    }
                });
            } else {
                run_timed_stage("block_backward.ln2_residual.affine", [&]() {
                    run_layer_norm_backward_affine_accumulate(
                        tape.residual1,
                        grad_ln2,
                        ln2_mean,
                        ln2_rstd,
                        block.accum_grad_ln2_weight,
                        block.accum_grad_ln2_bias,
                        label + ".ln2.backward_affine.accumulate");
                });
                run_timed_stage("block_backward.ln2_residual.dinput", [&]() {
                    run_layer_norm_backward_input(tape.residual1, grad_ln2, block.ln2_weight, ln2_mean, ln2_rstd, grad_residual1_from_mlp, label + ".ln2.backward_input");
                });
                run_timed_stage("block_backward.ln2_residual.add", [&]() {
                    if (error.empty()) run(residual_add(incoming_grad, grad_residual1_from_mlp, residual_scale, grad_residual1, active_activation_elements, nullptr), label + ".mlp.residual.backward_add");
                });
            }
        });
        run_timed_stage("block_backward.attn_proj", [&]() {
            run_timed_stage("block_backward.attn_proj.dweight_bias", [&]() {
                if (packed_qkv_attention_enabled) {
                    if (error.empty()) {
                        auto* dweight_fn = dweight_first_microbatch_beta_zero_enabled
                            ? linear_backward_weight_bias_accumulate_bf16_bits_beta
                            : nullptr;
                        if (dweight_fn != nullptr) {
                            run(dweight_fn(
                                    active_packed_attn_out_bf16,
                                    grad_residual1,
                                    block.accum_grad_attn_proj_weight,
                                    block.accum_grad_attn_proj_bias,
                                    active_rows,
                                    kDim,
                                    kDim,
                                    dweight_beta,
                                    nullptr),
                                label + ".attn.out.backward_weight_bias.beta.packed_o_bf16_bits");
                        } else {
                            run(linear_backward_weight_bias_accumulate_bf16_bits(
                                    active_packed_attn_out_bf16,
                                    grad_residual1,
                                    block.accum_grad_attn_proj_weight,
                                    block.accum_grad_attn_proj_bias,
                                    active_rows,
                                    kDim,
                                    kDim,
                                    nullptr),
                                label + ".attn.out.backward_weight_bias.accumulate.packed_o_bf16_bits");
                        }
                    }
                } else {
                    if (error.empty()) run(linear_backward_weight_bias_accumulate_bf16(tape.attn_out, grad_residual1, block.accum_grad_attn_proj_weight, block.accum_grad_attn_proj_bias, active_rows, kDim, kDim, nullptr), label + ".attn.out.backward_weight_bias.accumulate.bf16");
                }
            });
            run_timed_stage("block_backward.attn_proj.dinput", [&]() {
                if (bf16_attention_grad_out_handoff_enabled) {
                    if (error.empty()) {
                        run(linear_backward_input_weight_bf16_to_bf16_bits(
                                grad_residual1,
                                block.attn_proj_weight_bf16,
                                active_attention_grad_out_bf16,
                                active_rows,
                                kDim,
                                kDim,
                                nullptr),
                            label + ".attn.out.backward_input.weight_bf16_to_bf16_bits");
                    }
                } else {
                    if (error.empty()) run(linear_backward_input_weight_bf16(grad_residual1, block.attn_proj_weight_bf16, grad_attn_out, active_rows, kDim, kDim, nullptr), label + ".attn.out.backward_input.weight_bf16");
                }
            });
        });
        run_timed_stage("block_backward.attn_sdpa", [&]() {
            run_timed_stage("block_backward.attn_sdpa.to_qkv", [&]() {
                if (packed_qkv_attention_enabled) {
                    if (error.empty()) {
                        if (stored_packed_attention != nullptr &&
                            stored_packed_attention->lse != nullptr) {
                            if (bf16_attention_grad_out_handoff_enabled) {
                                run(packed_attention_backward_to_qkv_bf16_bits_saved_lse_from_bf16_grad(
                                        active_qkv_bf16,
                                        active_packed_attn_out_bf16,
                                        stored_packed_attention->lse,
                                        active_attention_grad_out_bf16,
                                        active_qkv_grad_bf16,
                                        active_batch_size,
                                        kHeads,
                                        kHeads,
                                        seq_len,
                                        seq_len,
                                        kHeadDim,
                                        kHeadDim,
                                        attention_scale,
                                        true,
                                        false,
                                        false,
                                        0,
                                        0,
                                        0,
                                        0,
                                        nullptr),
                                    label + ".attn.sdpa.backward_packed_qkv_to_qkv_bf16_from_bf16_grad_saved_lse");
                            } else if (bf16_qkv_grad_handoff_enabled) {
                                run(packed_attention_backward_to_qkv_bf16_bits_saved_lse(
                                        active_qkv_bf16,
                                        active_packed_attn_out_bf16,
                                        stored_packed_attention->lse,
                                        grad_attn_out,
                                        active_qkv_grad_bf16,
                                        active_batch_size,
                                        kHeads,
                                        kHeads,
                                        seq_len,
                                        seq_len,
                                        kHeadDim,
                                        kHeadDim,
                                        attention_scale,
                                        true,
                                        false,
                                        false,
                                        0,
                                        0,
                                        0,
                                        0,
                                        nullptr),
                                    label + ".attn.sdpa.backward_packed_qkv_to_qkv_bf16_saved_lse");
                            } else {
                                run(packed_attention_backward_to_qkv_saved_lse(
                                        active_qkv_bf16,
                                        active_packed_attn_out_bf16,
                                        stored_packed_attention->lse,
                                        grad_attn_out,
                                        grad_qkv,
                                        active_batch_size,
                                        kHeads,
                                        kHeads,
                                        seq_len,
                                        seq_len,
                                        kHeadDim,
                                        kHeadDim,
                                        attention_scale,
                                        true,
                                        false,
                                        false,
                                        0,
                                        0,
                                        0,
                                        0,
                                        nullptr),
                                    label + ".attn.sdpa.backward_packed_qkv_to_qkv_saved_lse");
                            }
                        } else {
                            if (bf16_attention_grad_out_handoff_enabled) {
                                run(packed_attention_backward_to_qkv_bf16_bits_from_bf16_grad(
                                        active_qkv_bf16,
                                        active_packed_attn_out_bf16,
                                        active_attention_grad_out_bf16,
                                        active_qkv_grad_bf16,
                                        active_batch_size,
                                        kHeads,
                                        kHeads,
                                        seq_len,
                                        seq_len,
                                        kHeadDim,
                                        kHeadDim,
                                        attention_scale,
                                        true,
                                        false,
                                        false,
                                        0,
                                        0,
                                        0,
                                        0,
                                        nullptr),
                                    label + ".attn.sdpa.backward_packed_qkv_to_qkv_bf16_from_bf16_grad");
                            } else if (bf16_qkv_grad_handoff_enabled) {
                                run(packed_attention_backward_to_qkv_bf16_bits(
                                        active_qkv_bf16,
                                        active_packed_attn_out_bf16,
                                        grad_attn_out,
                                        active_qkv_grad_bf16,
                                        active_batch_size,
                                        kHeads,
                                        kHeads,
                                        seq_len,
                                        seq_len,
                                        kHeadDim,
                                        kHeadDim,
                                        attention_scale,
                                        true,
                                        false,
                                        false,
                                        0,
                                        0,
                                        0,
                                        0,
                                        nullptr),
                                    label + ".attn.sdpa.backward_packed_qkv_to_qkv_bf16");
                            } else {
                                run(packed_attention_backward_to_qkv(
                                        active_qkv_bf16,
                                        active_packed_attn_out_bf16,
                                        grad_attn_out,
                                        grad_qkv,
                                        active_batch_size,
                                        kHeads,
                                        kHeads,
                                        seq_len,
                                        seq_len,
                                        kHeadDim,
                                        kHeadDim,
                                        attention_scale,
                                        true,
                                        false,
                                        false,
                                        0,
                                        0,
                                        0,
                                        0,
                                        nullptr),
                                    label + ".attn.sdpa.backward_packed_qkv_to_qkv");
                            }
                        }
                        if (error.empty() && stored_packed_attention != nullptr) {
                            stored_packed_attention_backward_kernel_launches += 1;
                        }
                    }
                } else if (stored_attention != nullptr) {
                    if (error.empty()) {
                        run(attention_backward_to_qkv_from_saved_tk(
                                stored_attention->q,
                                stored_attention->k,
                                stored_attention->v,
                                stored_attention->o,
                                stored_attention->lse,
                                grad_attn_out,
                                grad_qkv,
                                active_batch_size,
                                kHeads,
                                kHeads,
                                seq_len,
                                seq_len,
                                kHeadDim,
                                kHeadDim,
                                attention_scale,
                                true,
                                false,
                                false,
                                0,
                                0,
                                0,
                                0,
                                nullptr),
                            label + ".attn.sdpa.backward_to_qkv_from_saved_tk_bf16");
                        if (error.empty()) stored_attention_backward_kernel_launches += 1;
                    }
                } else {
                    if (error.empty()) run(attention_backward_to_qkv_reuse_forward(grad_attn_out, grad_qkv, active_batch_size, kHeads, kHeads, seq_len, seq_len, kHeadDim, kHeadDim, attention_scale, true, false, false, 0, 0, 0, 0, nullptr), label + ".attn.sdpa.backward_to_qkv_reuse_forward_from_merged_grad");
                }
            });
        });
        run_timed_stage("block_backward.qkv", [&]() {
            run_timed_stage("block_backward.qkv.dweight_bias", [&]() {
                if (bf16_qkv_grad_handoff_enabled) {
                    if (error.empty()) {
                        if (bf16_qkv_dweight_enabled) {
                            const std::uint16_t* ln1_bf16_for_dweight =
                                ln1_bf16_qkv_forward_enabled ? tape.ln1_out_bf16 : nullptr;
                            if (ln1_bf16_for_dweight == nullptr) {
                                run(float32_to_bf16_bits(
                                        tape.ln1_out,
                                        active_qkv_bf16,
                                        active_activation_elements,
                                        nullptr),
                                    label + ".attn.qkv.ln1_out.to_bf16_bits");
                                ln1_bf16_for_dweight = active_qkv_bf16;
                            }
                            if (error.empty() && ln1_bf16_for_dweight != nullptr) {
                                if (bf16_block_dweight_staging_enabled &&
                                    block.accum_grad_qkv_weight_bf16 != nullptr) {
                                    run(linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_to_bf16_bits(
                                            ln1_bf16_for_dweight,
                                            active_qkv_grad_bf16,
                                            block.accum_grad_qkv_weight_bf16,
                                            block.accum_grad_qkv_bias,
                                            active_rows,
                                            kDim,
                                            kQkvDim,
                                            nullptr),
                                        label + ".attn.qkv.backward_weight_bias.accumulate.bf16_ln1_bf16_grad_to_bf16");
                                } else {
                                    auto* dweight_fn = dweight_first_microbatch_beta_zero_enabled
                                        ? linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_beta
                                        : nullptr;
                                    if (dweight_fn != nullptr) {
                                        run(dweight_fn(
                                                ln1_bf16_for_dweight,
                                                active_qkv_grad_bf16,
                                                block.accum_grad_qkv_weight,
                                                block.accum_grad_qkv_bias,
                                                active_rows,
                                                kDim,
                                                kQkvDim,
                                                dweight_beta,
                                                nullptr),
                                            label + ".attn.qkv.backward_weight_bias.beta.bf16_ln1_bf16_grad");
                                    } else {
                                        run(linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits(
                                                ln1_bf16_for_dweight,
                                                active_qkv_grad_bf16,
                                                block.accum_grad_qkv_weight,
                                                block.accum_grad_qkv_bias,
                                                active_rows,
                                                kDim,
                                                kQkvDim,
                                                nullptr),
                                            label + ".attn.qkv.backward_weight_bias.accumulate.bf16_ln1_bf16_grad");
                                    }
                                }
                            }
                        } else {
                            auto* dweight_fn = dweight_first_microbatch_beta_zero_enabled
                                ? linear_backward_weight_bias_accumulate_float32_bf16_bits_beta
                                : nullptr;
                            if (dweight_fn != nullptr) {
                                run(dweight_fn(
                                        tape.ln1_out,
                                        active_qkv_grad_bf16,
                                        block.accum_grad_qkv_weight,
                                        block.accum_grad_qkv_bias,
                                        active_rows,
                                        kDim,
                                        kQkvDim,
                                        dweight_beta,
                                        nullptr),
                                    label + ".attn.qkv.backward_weight_bias.beta.float32_bf16_grad");
                            } else {
                                run(linear_backward_weight_bias_accumulate_float32_bf16_bits(
                                        tape.ln1_out,
                                        active_qkv_grad_bf16,
                                        block.accum_grad_qkv_weight,
                                        block.accum_grad_qkv_bias,
                                        active_rows,
                                        kDim,
                                        kQkvDim,
                                        nullptr),
                                    label + ".attn.qkv.backward_weight_bias.accumulate.float32_bf16_grad");
                            }
                        }
                    }
                } else {
                    if (error.empty()) run(linear_backward_weight_bias_accumulate_bf16(tape.ln1_out, grad_qkv, block.accum_grad_qkv_weight, block.accum_grad_qkv_bias, active_rows, kDim, kQkvDim, nullptr), label + ".attn.qkv.backward_weight_bias.accumulate.bf16");
                }
            });
            run_timed_stage("block_backward.qkv.dinput", [&]() {
                if (bf16_qkv_grad_handoff_enabled) {
                    if (error.empty()) {
                        run(linear_backward_input_bf16_bits_weight_bf16(
                                active_qkv_grad_bf16,
                                block.qkv_weight_bf16,
                                grad_ln1,
                                active_rows,
                                kDim,
                                kQkvDim,
                                nullptr),
                            label + ".attn.qkv.backward_input.bf16_bits_weight_bf16");
                    }
                } else {
                    if (error.empty()) run(linear_backward_input_weight_bf16(grad_qkv, block.qkv_weight_bf16, grad_ln1, active_rows, kDim, kQkvDim, nullptr), label + ".attn.qkv.backward_input.weight_bf16");
                }
            });
        });
        run_timed_stage("block_backward.ln1_residual", [&]() {
            const bool use_fused_ln1_affine_residual =
                fuse_ln_backward_affine_residual_enabled && active_rows > 0 && kDim <= 1024;
            if (use_fused_ln1_affine_residual) {
                run_timed_stage("block_backward.ln1_residual.fused_affine_dinput_add", [&]() {
                    run_layer_norm_backward_affine_residual_add_accumulate(
                        block_input,
                        grad_ln1,
                        block.ln1_weight,
                        active_ln1_mean,
                        active_ln1_rstd,
                        grad_residual1,
                        residual_scale,
                        output_grad,
                        block.accum_grad_ln1_weight,
                        block.accum_grad_ln1_bias,
                        label + ".ln1.backward_affine_input_residual_add.accumulate");
                });
            } else if (fuse_ln_backward_residual_enabled) {
                run_timed_stage("block_backward.ln1_residual.affine", [&]() {
                    run_layer_norm_backward_affine_accumulate(block_input, grad_ln1, active_ln1_mean, active_ln1_rstd, block.accum_grad_ln1_weight, block.accum_grad_ln1_bias, label + ".ln1.backward_affine.accumulate");
                });
                run_timed_stage("block_backward.ln1_residual.dinput_add", [&]() {
                    if (error.empty()) {
                        run(layer_norm_backward_input_residual_add_with_stats(
                                block_input,
                                grad_ln1,
                                block.ln1_weight,
                                active_ln1_mean,
                                active_ln1_rstd,
                                grad_residual1,
                                residual_scale,
                                output_grad,
                                active_rows,
                                kDim,
                                nullptr),
                            label + ".ln1.backward_input_residual_add.with_stats");
                    }
                });
            } else {
                run_timed_stage("block_backward.ln1_residual.affine", [&]() {
                    run_layer_norm_backward_affine_accumulate(block_input, grad_ln1, active_ln1_mean, active_ln1_rstd, block.accum_grad_ln1_weight, block.accum_grad_ln1_bias, label + ".ln1.backward_affine.accumulate");
                });
                run_timed_stage("block_backward.ln1_residual.dinput", [&]() {
                    run_layer_norm_backward_input(block_input, grad_ln1, block.ln1_weight, active_ln1_mean, active_ln1_rstd, grad_x_from_attn, label + ".ln1.backward_input");
                });
                run_timed_stage("block_backward.ln1_residual.add", [&]() {
                    if (error.empty()) run(residual_add(grad_residual1, grad_x_from_attn, residual_scale, output_grad, active_activation_elements, nullptr), label + ".attn.residual.backward_add");
                });
            }
        });
        stage_end(stage_event, "block_backward");
    };

    auto forward_loss = [&](const std::string& label, bool compute_loss, bool preserve_block_outputs) -> double {
        const std::int64_t stage_event = stage_begin(label + ".model_forward");
        if (error.empty()) {
            if (direct_u16_token_ids_enabled) {
                run(token_embedding_u16(token_weight, token_ids_u16, token_out, active_rows, kDim, nullptr),
                    label + ".wte.forward.u16");
            } else {
                run(token_embedding(token_weight, token_ids, token_out, active_rows, kDim, nullptr),
                    label + ".wte.forward");
            }
        }
        if (error.empty()) run(position_embedding(position_weight, position_out, active_batch_size, seq_len, kDim, nullptr), label + ".wpe.forward");
        if (error.empty()) run(residual_add(token_out, position_out, residual_scale, x, active_activation_elements, nullptr), label + ".embedding.residual");
        const float* block_input = x;
        float* final_block_output = x;
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            TransformerBlockActivations& tape = block_tapes[0];
            StoredMlpActivations* fused_mlp_store =
                preserve_block_outputs && i < stored_mlp_activations.size() ? &stored_mlp_activations[i] : nullptr;
            StoredAttentionActivations* fused_attention_store =
                preserve_block_outputs && i < stored_attention_activations.size() ? &stored_attention_activations[i] : nullptr;
            StoredPackedAttentionActivations* fused_packed_attention_store =
                preserve_block_outputs && i < stored_packed_attention_activations.size()
                    ? &stored_packed_attention_activations[i]
                    : nullptr;
            std::uint16_t* fused_residual1_store =
                preserve_block_outputs && fuse_residual1_store_enabled && i < stored_residual1_activations.size()
                    ? stored_residual1_activations[i]
                    : nullptr;
            forward_block(
                blocks[i],
                tape,
                block_input,
                label + ".block" + std::to_string(i),
                true,
                true,
                fused_attention_store,
                fused_packed_attention_store,
                fused_mlp_store,
                fused_residual1_store,
                preserve_block_outputs && i + 1 < blocks.size() ? block_outputs[i] : nullptr);
            if (preserve_block_outputs && i + 1 < blocks.size()) {
                final_block_output = block_outputs[i];
                direct_block_output_write_count += 1;
            } else {
                final_block_output = tape.residual2;
            }
            if (error.empty() && preserve_block_outputs && i < stored_residual1_activations.size() &&
                (!fuse_residual1_store_enabled || !fuse_attention_residual_ln2_enabled)) {
                run(float32_to_bf16_bits(
                        tape.residual1,
                        stored_residual1_activations[i],
                        active_activation_elements,
                        nullptr),
                    label + ".block" + std::to_string(i) + ".residual1.store_bf16");
                if (error.empty()) {
                    stored_residual1_activation_store_kernel_launches += 1;
                }
            }
            if (error.empty() && preserve_block_outputs && i + 1 < blocks.size()) {
                if (fused_attention_store == nullptr) {
                    store_attention_activations(i);
                }
                if (fused_mlp_store == nullptr) {
                    store_mlp_activations(i, tape);
                }
                block_input = block_outputs[i];
            } else {
                block_input = tape.residual2;
            }
        }
        lnf_input = final_block_output;
        run_layer_norm(lnf_input, lnf_weight, lnf_bias, lnf_out, lnf_mean, lnf_rstd, label + ".ln_f.forward");
        stage_end(stage_event, label + ".model_forward");
        return (error.empty() && compute_loss) ? lm_head_forward_loss(label) : 0.0;
    };

    auto next_train_batch = [&]() -> bool {
        set_active_batch_size(batch_size);
        if (sampler.next_into(token_ids_pinned, targets_pinned, active_rows)) {
            upload_pinned_batch();
            return error.empty();
        }
        sampler.reset();
        epochs_completed += 1;
        if (sampler.next_into(token_ids_pinned, targets_pinned, active_rows)) {
            upload_pinned_batch();
            return error.empty();
        }
        error = "not enough train tokens to build one GPT-2 transformer/LM train batch";
        return false;
    };

    auto run_backward_microbatch = [&](bool record_train_loss, float accumulation_scale, bool dweight_accumulate) -> double {
        zero_gradients();
        const double train_loss_sum = forward_loss("train", record_train_loss, true);
        if (error.empty()) lm_head_backward(accumulation_scale, dweight_accumulate);
        const std::int64_t final_norm_event = stage_begin("final_norm_backward");
        run_layer_norm_backward_affine_accumulate(lnf_input, grad_lnf, lnf_mean, lnf_rstd, accum_grad_lnf_weight, accum_grad_lnf_bias, "ln_f.backward_affine.accumulate");
        run_layer_norm_backward_input(lnf_input, grad_lnf, lnf_weight, lnf_mean, lnf_rstd, grad_residual2, "ln_f.backward_input");
        stage_end(final_norm_event, "final_norm_backward");
        float* incoming_grad = grad_residual2;
        float* output_grad = grad_x;
        for (std::size_t reverse_index = blocks.size(); reverse_index > 0 && error.empty(); --reverse_index) {
            const std::size_t i = reverse_index - 1;
            TransformerBlockActivations& tape = block_tapes[0];
            const float* block_input = block_input_for(i);
            const std::uint16_t* stored_residual1_bf16 =
                i < stored_residual1_activations.size() ? stored_residual1_activations[i] : nullptr;
            if (reverse_index != blocks.size()) {
                const bool use_stored_mlp_activations = i < stored_mlp_activations.size();
                if (i < stored_attention_activations.size()) {
                    recompute_block_from_saved_attention(
                        blocks[i],
                        tape,
                        stored_attention_activations[i],
                        block_input,
                        !use_stored_mlp_activations,
                        "block" + std::to_string(i) + ".recompute_saved_attention");
                    if (error.empty()) stored_attention_restore_kernel_launches += 1;
                } else if (i < stored_packed_attention_activations.size()) {
                    recompute_block_from_saved_packed_attention(
                        blocks[i],
                        tape,
                        stored_packed_attention_activations[i],
                        stored_residual1_bf16,
                        block_input,
                        !use_stored_mlp_activations,
                        "block" + std::to_string(i) + ".recompute_saved_packed_attention");
                    if (error.empty()) stored_packed_attention_restore_blocks += 1;
                } else {
                    forward_block(
                        blocks[i],
                        tape,
                        block_input,
                        "block" + std::to_string(i) + ".recompute",
                        false,
                        !use_stored_mlp_activations,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr);
                }
            }
            const StoredMlpActivations* stored_mlp =
                i < stored_mlp_activations.size() ? &stored_mlp_activations[i] : nullptr;
            const StoredAttentionActivations* stored_attention =
                i < stored_attention_activations.size() ? &stored_attention_activations[i] : nullptr;
            const StoredPackedAttentionActivations* stored_packed_attention =
                i < stored_packed_attention_activations.size() ? &stored_packed_attention_activations[i] : nullptr;
            backward_block(
                blocks[i],
                tape,
                stored_mlp,
                stored_attention,
                stored_packed_attention,
                stored_residual1_bf16,
                block_input,
                incoming_grad,
                output_grad,
                dweight_accumulate,
                "block" + std::to_string(i));
            std::swap(incoming_grad, output_grad);
        }
        const std::int64_t embedding_backward_event = stage_begin("embedding_backward");
        if (error.empty()) {
            if (direct_u16_token_ids_enabled) {
                run(token_embedding_backward_weight_u16(
                        token_ids_u16,
                        incoming_grad,
                        accum_grad_token_weight,
                        active_rows,
                        kDim,
                        nullptr),
                    "wte.backward_weight.u16");
            } else {
                run(token_embedding_backward_weight(
                        token_ids,
                        incoming_grad,
                        accum_grad_token_weight,
                        active_rows,
                        kDim,
                        nullptr),
                    "wte.backward_weight");
            }
        }
        if (error.empty()) run(position_embedding_backward_accumulate(incoming_grad, accum_grad_position_weight, active_batch_size, seq_len, kDim, nullptr), "wpe.backward_weight.accumulate");
        stage_end(embedding_backward_event, "embedding_backward");
        return train_loss_sum;
    };

    auto forward_backward_update = [&](std::int64_t step, bool record_train_loss) -> double {
        zero_accumulated_gradients();
        double train_loss_sum = 0.0;
        const float accumulation_scale = 1.0f / static_cast<float>(grad_accum_steps);
        for (std::int64_t accum_index = 0; accum_index < grad_accum_steps && error.empty(); ++accum_index) {
            if (!next_train_batch()) {
                break;
            }
            const bool dweight_accumulate = accum_index != 0 || !dweight_first_microbatch_beta_zero_enabled;
            const double microbatch_loss_sum =
                run_backward_microbatch(record_train_loss, accumulation_scale, dweight_accumulate);
            if (record_train_loss && error.empty()) {
                train_loss_sum += microbatch_loss_sum;
            }
            if (error.empty()) {
                accumulate_gradients(accumulation_scale);
                train_microbatches_completed += 1;
            }
        }
        flush_bf16_block_dweight_staging();
        clip_gradients();

        const float bias_correction1 = 1.0f - std::pow(kBeta1, static_cast<float>(step));
        const float sqrt_bias_correction2 = std::sqrt(1.0f - std::pow(kBeta2, static_cast<float>(step)));
        if (error.empty()) {
            const std::int64_t stage_event = stage_begin("adamw_update");
            if (bf16_block_weight_param_update_enabled) {
                if (adamw_float_update_descriptor_count > 0) {
                    run(adamw_many_with_device_scale(
                            adamw_float_update_param_ptrs,
                            reinterpret_cast<const float* const*>(adamw_float_update_grad_ptrs),
                            grad_clip_scale,
                            adamw_float_update_avg_ptrs,
                            adamw_float_update_avg_sq_ptrs,
                            adamw_float_update_elements,
                            adamw_float_update_weight_decays,
                            adamw_float_update_descriptor_count,
                            adamw_float_update_max_elements,
                            static_cast<float>(cfg.learning_rate),
                            kBeta1,
                            kBeta2,
                            kEps,
                            bias_correction1,
                            sqrt_bias_correction2,
                            nullptr),
                        "adamw_many_with_device_scale.float_params");
                    if (error.empty()) {
                        adamw_kernel_launches += 1;
                        adamw_float_update_kernel_launches += 1;
                    }
                }
                if (error.empty() && adamw_bf16_param_descriptor_count > 0) {
                    run(adamw_many_with_device_scale_bf16_param(
                            adamw_bf16_param_ptrs,
                            reinterpret_cast<const float* const*>(adamw_bf16_param_grad_ptrs),
                            grad_clip_scale,
                            adamw_bf16_param_avg_ptrs,
                            adamw_bf16_param_avg_sq_ptrs,
                            adamw_bf16_param_elements,
                            adamw_bf16_param_weight_decays,
                            adamw_bf16_param_descriptor_count,
                            adamw_bf16_param_max_elements,
                            static_cast<float>(cfg.learning_rate),
                            kBeta1,
                            kBeta2,
                            kEps,
                            bias_correction1,
                            sqrt_bias_correction2,
                            nullptr),
                        "adamw_many_with_device_scale_bf16_param");
                    if (error.empty()) {
                        adamw_kernel_launches += 1;
                        adamw_bf16_param_kernel_launches += 1;
                    }
                }
                if (error.empty() && adamw_bf16_param_bf16_grad_descriptor_count > 0) {
                    run(adamw_many_with_device_scale_bf16_param_bf16_grad(
                            adamw_bf16_param_bf16_grad_param_ptrs,
                            reinterpret_cast<const std::uint16_t* const*>(adamw_bf16_param_bf16_grad_grad_ptrs),
                            grad_clip_scale,
                            adamw_bf16_param_bf16_grad_avg_ptrs,
                            adamw_bf16_param_bf16_grad_avg_sq_ptrs,
                            adamw_bf16_param_bf16_grad_elements,
                            adamw_bf16_param_bf16_grad_weight_decays,
                            adamw_bf16_param_bf16_grad_descriptor_count,
                            adamw_bf16_param_bf16_grad_max_elements,
                            static_cast<float>(cfg.learning_rate),
                            kBeta1,
                            kBeta2,
                            kEps,
                            bias_correction1,
                            sqrt_bias_correction2,
                            nullptr),
                        "adamw_many_with_device_scale_bf16_param_bf16_grad");
                    if (error.empty()) {
                        adamw_kernel_launches += 1;
                        adamw_bf16_param_bf16_grad_kernel_launches += 1;
                    }
                }
            } else if (fuse_adamw_bf16_shadow_refresh_enabled) {
                run(adamw_many_with_device_scale_bf16_shadow(
                        adamw_param_ptrs,
                        reinterpret_cast<const float* const*>(adamw_grad_ptrs),
                        grad_clip_scale,
                        adamw_avg_ptrs,
                        adamw_avg_sq_ptrs,
                        adamw_elements,
                        adamw_weight_decays,
                        adamw_bf16_shadow_offsets,
                        block_weight_bf16_arena,
                        adamw_descriptor_count,
                        adamw_max_elements,
                        static_cast<float>(cfg.learning_rate),
                        kBeta1,
                        kBeta2,
                        kEps,
                        bias_correction1,
                        sqrt_bias_correction2,
                        nullptr),
                    "adamw_many_with_device_scale_bf16_shadow");
            } else {
                run(adamw_many_with_device_scale(
                        adamw_param_ptrs,
                        reinterpret_cast<const float* const*>(adamw_grad_ptrs),
                        grad_clip_scale,
                        adamw_avg_ptrs,
                        adamw_avg_sq_ptrs,
                        adamw_elements,
                        adamw_weight_decays,
                        adamw_descriptor_count,
                        adamw_max_elements,
                        static_cast<float>(cfg.learning_rate),
                        kBeta1,
                        kBeta2,
                        kEps,
                        bias_correction1,
                        sqrt_bias_correction2,
                        nullptr),
                    "adamw_many_with_device_scale");
            }
            if (error.empty()) {
                if (!bf16_block_weight_param_update_enabled) {
                    adamw_kernel_launches += 1;
                }
                if (trainer_linear_bf16_cache_reset != nullptr) {
                    trainer_linear_bf16_cache_reset();
                }
                if (bf16_block_weight_param_update_enabled) {
                    // The block BF16 arena is now the authoritative parameter store.
                } else if (fuse_adamw_bf16_shadow_refresh_enabled) {
                    block_weight_bf16_fused_adamw_refresh_count += 1;
                } else {
                    refresh_block_weight_bf16("block_weight_bf16.post_adamw_refresh");
                }
                refresh_token_weight_bf16("token_weight_bf16.post_adamw_refresh");
            }
            stage_end(stage_event, "adamw_update");
        }
        return train_loss_sum;
    };

    neuralfn::native_train::SequentialTokenBatchSampler val_sampler(dataset.val_shards, seq_len, eval_batch_size);
    auto run_validation = [&](std::int64_t step) {
        if (!error.empty() || cfg.eval_every_steps <= 0 || cfg.eval_batches <= 0) {
            return;
        }
        set_active_batch_size(eval_batch_size);
        const auto validation_start_time = Clock::now();
        ValidationLossRecord record;
        record.step = step;
        for (std::int64_t batch_index = 0; batch_index < cfg.eval_batches; ++batch_index) {
            if (!val_sampler.next_into(token_ids_pinned, active_targets_pinned, active_rows)) {
                val_sampler.reset();
                if (!val_sampler.next_into(token_ids_pinned, active_targets_pinned, active_rows)) {
                    error = "not enough validation tokens to build one GPT-2 transformer/LM validation batch";
                    break;
                }
            }
            upload_pinned_batch();
            allocate_validation_mlp_float_scratch();
            const double loss_sum = forward_loss("validation", true, false);
            if (!error.empty()) {
                break;
            }
            record.batches += 1;
            record.tokens += active_rows;
            record.loss_sum += loss_sum;
        }
        if (error.empty() && record.batches > 0 && record.tokens > 0) {
            record.loss_mean = record.loss_sum / static_cast<double>(record.tokens);
            validation_losses.push_back(record);
        }
        validation_wall_ms += elapsed_ms(validation_start_time, Clock::now());
        set_active_batch_size(batch_size);
    };

    const auto train_loop_start_time = Clock::now();
    setup_wall_ms = elapsed_ms(total_start_time, train_loop_start_time);
    if (!cfg.startup_only) {
        for (std::int64_t step = 1; step <= cfg.max_steps && error.empty(); ++step) {
            const bool should_run_validation =
                cfg.eval_every_steps > 0 && (step % cfg.eval_every_steps) == 0;
            const bool should_record_train_loss = false;
            const double train_loss_sum = forward_backward_update(step, should_record_train_loss);
            if (error.empty()) {
                steps_completed = step;
                tokens_processed += effective_train_batch_tokens;
                if (should_record_train_loss) {
                    final_loss_sum = train_loss_sum;
                    final_loss_mean = final_loss_sum / static_cast<double>(effective_train_batch_tokens);
                    train_loss_eval_count += 1;
                    train_loss_last_step = step;
                }
                if (should_run_validation) {
                    run_validation(step);
                }
            }
        }
    }

    if (error.empty()) {
        run(cuda_device_synchronize(), "train_loop.complete");
    }
    const auto train_loop_end_time = Clock::now();
    train_loop_wall_ms = elapsed_ms(train_loop_start_time, train_loop_end_time);
    float sampled_token_weight = initial_token_weight_sample;
    if (error.empty()) {
        run(cuda_memcpy(&sampled_token_weight, token_weight, sizeof(float), kCudaMemcpyDeviceToHost),
            "token_weight.sample");
    }
    float sampled_clip_scale = 0.0f;
    if (error.empty()) {
        run(cuda_memcpy(&sampled_clip_scale, grad_clip_scale, sizeof(float), kCudaMemcpyDeviceToHost),
            "gradient_clip_scale.sample");
    }
    finalize_stage_timing();
    const double max_weight_delta = std::fabs(static_cast<double>(sampled_token_weight) - initial_token_weight_sample);
    passed = error.empty() &&
        ((cfg.startup_only && steps_completed == 0) ||
         (!cfg.startup_only && steps_completed == cfg.max_steps && max_weight_delta > 0.0));

    auto write_trained_checkpoint = [&]() {
        constexpr std::int32_t kCheckpointMagic = 20240326;
        constexpr std::int32_t kCheckpointVersion = 5;
        constexpr std::int64_t kCheckpointHeaderInts = 256;
        constexpr std::int64_t kCheckpointHeaderBytes = kCheckpointHeaderInts * 4;
        constexpr std::int64_t kBytesPerParam = 2;
        const fs::path output_dir = cfg.output_dir.empty() ? fs::path(default_output_dir()) : fs::path(cfg.output_dir);
        checkpoint_step = steps_completed;
        std::ostringstream checkpoint_name;
        checkpoint_name << "model_" << std::setw(8) << std::setfill('0') << checkpoint_step << ".bin";
        const fs::path checkpoint_path = output_dir / checkpoint_name.str();
        const fs::path done_marker = output_dir / ("DONE_" + checkpoint_name.str().substr(6, 8));
        checkpoint_path_json = checkpoint_path.string();
        done_marker_json = done_marker.string();
        const std::int64_t parameter_count =
            native_gpt2_parameter_count(seq_len, kPaddedVocab, trained_layers, kDim);
        checkpoint_expected_file_size = kCheckpointHeaderBytes + parameter_count * kBytesPerParam;

        try {
            fs::create_directories(output_dir);
        } catch (const std::exception& exc) {
            error = std::string("failed to create checkpoint directory: ") + exc.what();
            return;
        }

        std::vector<std::int32_t> header(static_cast<std::size_t>(kCheckpointHeaderInts), 0);
        header[0] = kCheckpointMagic;
        header[1] = kCheckpointVersion;
        header[2] = static_cast<std::int32_t>(seq_len);
        header[3] = static_cast<std::int32_t>(kVocab);
        header[4] = static_cast<std::int32_t>(trained_layers);
        header[5] = static_cast<std::int32_t>(kHeads);
        header[6] = static_cast<std::int32_t>(kDim);
        header[7] = static_cast<std::int32_t>(kPaddedVocab);

        std::ofstream out(checkpoint_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            error = "failed to open trained checkpoint for writing: " + checkpoint_path.string();
            return;
        }
        out.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(kCheckpointHeaderBytes));
        if (!out) {
            error = "failed to write trained checkpoint header: " + checkpoint_path.string();
            return;
        }

        if (bf16_block_weight_param_update_enabled) {
            for (std::size_t i = 0; i < blocks.size() && error.empty(); ++i) {
                TransformerBlockParams& block = blocks[i];
                const std::string prefix = "block" + std::to_string(i);
                run(bf16_bits_to_float32(
                        block.qkv_weight_bf16,
                        block.qkv_weight,
                        kQkvWeightElements,
                        nullptr),
                    prefix + ".attn.qkv.weight.sync_bf16_param_to_fp32_checkpoint");
                if (error.empty()) {
                    checkpoint_bf16_param_sync_kernel_launches += 1;
                }
                if (error.empty()) {
                    run(bf16_bits_to_float32(
                            block.attn_proj_weight_bf16,
                            block.attn_proj_weight,
                            kAttnProjWeightElements,
                            nullptr),
                        prefix + ".attn.proj.weight.sync_bf16_param_to_fp32_checkpoint");
                    if (error.empty()) {
                        checkpoint_bf16_param_sync_kernel_launches += 1;
                    }
                }
                if (error.empty()) {
                    run(bf16_bits_to_float32(
                            block.fc_weight_bf16,
                            block.fc_weight,
                            kFcWeightElements,
                            nullptr),
                        prefix + ".mlp.fc.weight.sync_bf16_param_to_fp32_checkpoint");
                    if (error.empty()) {
                        checkpoint_bf16_param_sync_kernel_launches += 1;
                    }
                }
                if (error.empty()) {
                    run(bf16_bits_to_float32(
                            block.mlp_proj_weight_bf16,
                            block.mlp_proj_weight,
                            kMlpProjWeightElements,
                            nullptr),
                        prefix + ".mlp.proj.weight.sync_bf16_param_to_fp32_checkpoint");
                    if (error.empty()) {
                        checkpoint_bf16_param_sync_kernel_launches += 1;
                    }
                }
            }
            if (!error.empty()) {
                return;
            }
        }

        struct CheckpointTensor {
            const float* device_ptr;
            std::int64_t elements;
            std::string name;
        };
        std::vector<CheckpointTensor> checkpoint_tensors;
        checkpoint_tensors.reserve(static_cast<std::size_t>(2 + trained_layers * 12 + 2));
        auto add_checkpoint_tensor = [&](const float* device_ptr, std::int64_t elements, std::string name) {
            if (elements < 0) {
                error = "bad checkpoint tensor element count for " + name;
                return;
            }
            checkpoint_tensors.push_back(CheckpointTensor{device_ptr, elements, std::move(name)});
        };

        add_checkpoint_tensor(token_weight, kTokenWeightElements, "wte.weight");
        add_checkpoint_tensor(position_weight, position_weight_elements, "wpe.weight");
        for (std::size_t i = 0; i < blocks.size() && error.empty(); ++i) {
            TransformerBlockParams& block = blocks[i];
            const std::string prefix = "block" + std::to_string(i);
            add_checkpoint_tensor(block.ln1_weight, kDim, prefix + ".ln1.weight");
            add_checkpoint_tensor(block.ln1_bias, kDim, prefix + ".ln1.bias");
            add_checkpoint_tensor(block.qkv_weight, kQkvWeightElements, prefix + ".attn.qkv.weight");
            add_checkpoint_tensor(block.qkv_bias, kQkvDim, prefix + ".attn.qkv.bias");
            add_checkpoint_tensor(block.attn_proj_weight, kAttnProjWeightElements, prefix + ".attn.proj.weight");
            add_checkpoint_tensor(block.attn_proj_bias, kDim, prefix + ".attn.proj.bias");
            add_checkpoint_tensor(block.ln2_weight, kDim, prefix + ".ln2.weight");
            add_checkpoint_tensor(block.ln2_bias, kDim, prefix + ".ln2.bias");
            add_checkpoint_tensor(block.fc_weight, kFcWeightElements, prefix + ".mlp.fc.weight");
            add_checkpoint_tensor(block.fc_bias, kHidden, prefix + ".mlp.fc.bias");
            add_checkpoint_tensor(block.mlp_proj_weight, kMlpProjWeightElements, prefix + ".mlp.proj.weight");
            add_checkpoint_tensor(block.mlp_proj_bias, kDim, prefix + ".mlp.proj.bias");
        }
        add_checkpoint_tensor(lnf_weight, kDim, "ln_f.weight");
        add_checkpoint_tensor(lnf_bias, kDim, "ln_f.bias");
        if (!error.empty()) {
            return;
        }

        std::vector<const float*> host_sources;
        std::vector<std::int64_t> host_elements;
        std::vector<std::int64_t> host_offsets;
        host_sources.reserve(checkpoint_tensors.size());
        host_elements.reserve(checkpoint_tensors.size());
        host_offsets.reserve(checkpoint_tensors.size());
        std::int64_t payload_offset = 0;
        std::int64_t checkpoint_max_tensor_elements = 0;
        for (const CheckpointTensor& tensor : checkpoint_tensors) {
            host_sources.push_back(tensor.device_ptr);
            host_elements.push_back(tensor.elements);
            host_offsets.push_back(payload_offset);
            checkpoint_max_tensor_elements = std::max(checkpoint_max_tensor_elements, tensor.elements);
            if (tensor.elements > std::numeric_limits<std::int64_t>::max() - payload_offset) {
                error = "checkpoint payload element count overflow at " + tensor.name;
                return;
            }
            payload_offset += tensor.elements;
        }
        if (payload_offset != parameter_count) {
            std::ostringstream message;
            message << "bad checkpoint payload layout: got " << payload_offset
                    << " elements, expected " << parameter_count;
            error = message.str();
            return;
        }
        if (payload_offset > static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t))) {
            error = "checkpoint payload byte size overflow";
            return;
        }
        if (checkpoint_bf16_device == nullptr) {
            void* raw_checkpoint_buffer = nullptr;
            const std::size_t checkpoint_buffer_bytes =
                sizeof(std::uint16_t) * static_cast<std::size_t>(payload_offset);
            const int alloc_status = device_malloc(&raw_checkpoint_buffer, checkpoint_buffer_bytes);
            if (alloc_status != 0) {
                error = cuda_error(alloc_status, "cudaMalloc checkpoint_bf16_device");
                return;
            }
            checkpoint_bf16_device = static_cast<std::uint16_t*>(raw_checkpoint_buffer);
            uint16_ptrs.push_back(checkpoint_bf16_device);
        }

        const std::size_t descriptor_count = host_sources.size();
        const std::size_t source_bytes = descriptor_count * sizeof(const float*);
        const std::size_t i64_bytes = descriptor_count * sizeof(std::int64_t);
        void* raw_sources_device = nullptr;
        void* raw_elements_device = nullptr;
        void* raw_offsets_device = nullptr;
        run(device_malloc(&raw_sources_device, source_bytes), "cudaMalloc checkpoint_sources_device");
        if (error.empty()) {
            run(device_malloc(&raw_elements_device, i64_bytes), "cudaMalloc checkpoint_elements_device");
        }
        if (error.empty()) {
            run(device_malloc(&raw_offsets_device, i64_bytes), "cudaMalloc checkpoint_offsets_device");
        }
        if (!error.empty()) {
            return;
        }
        descriptor_ptrs.push_back(raw_sources_device);
        descriptor_ptrs.push_back(raw_elements_device);
        descriptor_ptrs.push_back(raw_offsets_device);
        const float** checkpoint_sources_device = static_cast<const float**>(raw_sources_device);
        std::int64_t* checkpoint_elements_device = static_cast<std::int64_t*>(raw_elements_device);
        std::int64_t* checkpoint_offsets_device = static_cast<std::int64_t*>(raw_offsets_device);
        run(cuda_memcpy(
                checkpoint_sources_device,
                host_sources.data(),
                source_bytes,
                kCudaMemcpyHostToDevice),
            "cudaMemcpy checkpoint_sources_device");
        if (error.empty()) {
            run(cuda_memcpy(
                    checkpoint_elements_device,
                    host_elements.data(),
                    i64_bytes,
                    kCudaMemcpyHostToDevice),
                "cudaMemcpy checkpoint_elements_device");
        }
        if (error.empty()) {
            run(cuda_memcpy(
                    checkpoint_offsets_device,
                    host_offsets.data(),
                    i64_bytes,
                    kCudaMemcpyHostToDevice),
                "cudaMemcpy checkpoint_offsets_device");
        }
        if (!error.empty()) {
            return;
        }
        const int pack_status = float32_to_bf16_bits_many(
            checkpoint_sources_device,
            checkpoint_elements_device,
            checkpoint_offsets_device,
            checkpoint_bf16_device,
            static_cast<std::int64_t>(descriptor_count),
            checkpoint_max_tensor_elements,
            nullptr);
        if (pack_status != 0) {
            error = cuda_error(pack_status, "float32_to_bf16_bits_many checkpoint payload");
            return;
        }
        checkpoint_device_pack_kernel_launches += 1;

        std::vector<std::uint16_t> host_payload;
        try {
            host_payload.resize(static_cast<std::size_t>(payload_offset));
        } catch (const std::exception& exc) {
            error = std::string("failed to allocate host checkpoint payload: ") + exc.what();
            return;
        }
        const std::size_t payload_bytes = host_payload.size() * sizeof(std::uint16_t);
        const int copy_status = cuda_memcpy(
            host_payload.data(),
            checkpoint_bf16_device,
            payload_bytes,
            kCudaMemcpyDeviceToHost);
        if (copy_status != 0) {
            error = cuda_error(copy_status, "cudaMemcpy checkpoint payload");
            return;
        }
        checkpoint_tensor_count = static_cast<std::int64_t>(checkpoint_tensors.size());
        checkpoint_payload_elements = payload_offset;
        checkpoint_d2h_copy_count = 1;
        checkpoint_d2h_bytes = static_cast<std::int64_t>(payload_bytes);
        checkpoint_float32_d2h_bytes_elided =
            static_cast<std::int64_t>(sizeof(float) - sizeof(std::uint16_t)) * payload_offset;
        out.write(
            reinterpret_cast<const char*>(host_payload.data()),
            static_cast<std::streamsize>(payload_bytes));
        if (!out) {
            error = "failed to write bf16 checkpoint payload";
        }
        if (!error.empty()) {
            return;
        }
        out.close();
        if (!out) {
            error = "failed to finish trained checkpoint file: " + checkpoint_path.string();
            return;
        }
        checkpoint_actual_file_size =
            fs::exists(checkpoint_path) ? static_cast<std::int64_t>(fs::file_size(checkpoint_path)) : 0;
        if (checkpoint_actual_file_size != checkpoint_expected_file_size) {
            std::ostringstream message;
            message << "bad trained checkpoint size: got " << checkpoint_actual_file_size
                    << " bytes, expected " << checkpoint_expected_file_size;
            error = message.str();
            return;
        }
        std::ofstream done(done_marker, std::ios::trunc);
        done.close();
        if (!done) {
            error = "failed to write trained checkpoint DONE marker: " + done_marker.string();
            return;
        }
        checkpoint_written = true;
    };

    if (passed && cfg.write_checkpoint) {
        const auto checkpoint_start_time = Clock::now();
        write_trained_checkpoint();
        checkpoint_wall_ms = elapsed_ms(checkpoint_start_time, Clock::now());
        passed = error.empty() && checkpoint_written;
    }

    if (attention_row_launch_count != nullptr) {
        attention_forward_row_launches = attention_row_launch_count();
    }
    if (attention_forward_tk_launch_count != nullptr) {
        attention_forward_tk_launches = attention_forward_tk_launch_count();
    }
    if (attention_backward_tk_launch_count != nullptr) {
        attention_backward_tk_launches = attention_backward_tk_launch_count();
    }
    if (attention_tk_workspace_allocation_count_fn != nullptr) {
        attention_tk_workspace_allocations = attention_tk_workspace_allocation_count_fn();
    }
    if (attention_tk_workspace_element_capacity_fn != nullptr) {
        attention_tk_workspace_element_capacity = attention_tk_workspace_element_capacity_fn();
    }
    if (attention_tk_workspace_row_capacity_fn != nullptr) {
        attention_tk_workspace_row_capacity = attention_tk_workspace_row_capacity_fn();
    }
    if (attention_row_fallback_count != nullptr) {
        attention_forward_row_fallbacks = attention_row_fallback_count();
    }
    if (attention_scalar_launch_count != nullptr) {
        attention_forward_scalar_launches = attention_scalar_launch_count();
    }
    if (attention_row_last_error != nullptr) {
        attention_forward_row_last_error_code = attention_row_last_error();
    }
    if (attention_row_prelaunch_clear_error != nullptr) {
        attention_forward_row_prelaunch_clear_error_code = attention_row_prelaunch_clear_error();
    }
    if (attention_row_prelaunch_peek_error != nullptr) {
        attention_forward_row_prelaunch_peek_error_code = attention_row_prelaunch_peek_error();
    }
    if (attention_row_grid_x != nullptr) {
        attention_forward_row_grid_x = attention_row_grid_x();
    }
    if (attention_row_grid_y != nullptr) {
        attention_forward_row_grid_y = attention_row_grid_y();
    }
    if (attention_row_grid_z != nullptr) {
        attention_forward_row_grid_z = attention_row_grid_z();
    }
    if (attention_row_block_x != nullptr) {
        attention_forward_row_block_x = attention_row_block_x();
    }
    if (attention_row_attr_status != nullptr) {
        attention_forward_row_attr_status_code = attention_row_attr_status();
    }
    if (attention_row_attr_max_threads_per_block != nullptr) {
        attention_forward_row_attr_max_threads_per_block = attention_row_attr_max_threads_per_block();
    }
    if (attention_row_attr_num_regs != nullptr) {
        attention_forward_row_attr_num_regs = attention_row_attr_num_regs();
    }
    if (attention_row_attr_shared_size_bytes != nullptr) {
        attention_forward_row_attr_shared_size_bytes = attention_row_attr_shared_size_bytes();
    }
    if (attention_row_attr_const_size_bytes != nullptr) {
        attention_forward_row_attr_const_size_bytes = attention_row_attr_const_size_bytes();
    }
    if (attention_row_attr_local_size_bytes != nullptr) {
        attention_forward_row_attr_local_size_bytes = attention_row_attr_local_size_bytes();
    }
    if (trainer_linear_bf16_gemm_count_fn != nullptr) {
        linear_bf16_gemm_count = trainer_linear_bf16_gemm_count_fn();
    }
    if (trainer_linear_tk_gemm_count_fn != nullptr) {
        linear_tk_gemm_count = trainer_linear_tk_gemm_count_fn();
    }
    if (trainer_linear_tk_float_out_gemm_count_fn != nullptr) {
        linear_tk_float_out_gemm_count = trainer_linear_tk_float_out_gemm_count_fn();
    }
    if (trainer_linear_cublaslt_gemm_count_fn != nullptr) {
        linear_cublaslt_gemm_count = trainer_linear_cublaslt_gemm_count_fn();
    }
    if (trainer_linear_sgemm_count_fn != nullptr) {
        linear_sgemm_count = trainer_linear_sgemm_count_fn();
    }
    if (trainer_linear_bf16_a_pack_count_fn != nullptr) {
        linear_bf16_a_pack_count = trainer_linear_bf16_a_pack_count_fn();
    }
    if (trainer_linear_bf16_a_cache_hit_count_fn != nullptr) {
        linear_bf16_a_cache_hit_count = trainer_linear_bf16_a_cache_hit_count_fn();
    }
    if (trainer_linear_bf16_cache_reset_count_fn != nullptr) {
        linear_bf16_cache_reset_count = trainer_linear_bf16_cache_reset_count_fn();
    }
    if (trainer_linear_bf16_workspace_allocation_count_fn != nullptr) {
        linear_bf16_workspace_allocation_count = trainer_linear_bf16_workspace_allocation_count_fn();
    }
    if (trainer_linear_bf16_workspace_a_capacity_fn != nullptr) {
        linear_bf16_workspace_a_capacity = trainer_linear_bf16_workspace_a_capacity_fn();
    }
    if (trainer_linear_bf16_workspace_b_capacity_fn != nullptr) {
        linear_bf16_workspace_b_capacity = trainer_linear_bf16_workspace_b_capacity_fn();
    }
    if (trainer_linear_bf16_cached_a_capacity_fn != nullptr) {
        linear_bf16_cached_a_capacity = trainer_linear_bf16_cached_a_capacity_fn();
    }
    if (trainer_linear_bf16_cache_entry_count_fn != nullptr) {
        linear_bf16_cache_entry_count = trainer_linear_bf16_cache_entry_count_fn();
    }
    if (trainer_linear_shape_stats_count_fn != nullptr &&
        trainer_linear_shape_stats_entry_fn != nullptr) {
        const std::int64_t shape_stat_count = trainer_linear_shape_stats_count_fn();
        const std::int64_t capped_shape_stat_count = std::min<std::int64_t>(shape_stat_count, 256);
        linear_shape_stats.reserve(static_cast<std::size_t>(capped_shape_stat_count));
        for (std::int64_t index = 0; index < capped_shape_stat_count; ++index) {
            LinearShapeStat stat;
            if (trainer_linear_shape_stats_entry_fn(
                    index,
                    &stat.path,
                    &stat.m,
                    &stat.n,
                    &stat.k,
                    &stat.op_a,
                    &stat.op_b,
                    &stat.calls)) {
                linear_shape_stats.push_back(stat);
            }
        }
    }
    attention_forward_row_successes =
        std::max<std::int64_t>(0, attention_forward_row_launches - attention_forward_row_fallbacks);

    for (void* ptr : descriptor_ptrs) {
        device_free(ptr, "cudaFree transformer_lm_descriptor_buffer");
    }
    for (float* ptr : float_ptrs) {
        device_free(ptr, "cudaFree transformer_lm_buffer");
    }
    for (void* ptr : token_device_ptrs) {
        device_free(ptr, "cudaFree transformer_lm_token_device_arena");
    }
    for (std::int64_t* ptr : int_ptrs) {
        device_free(ptr, "cudaFree transformer_lm_i64_buffer");
    }
    for (std::uint16_t* ptr : uint16_ptrs) {
        device_free(ptr, "cudaFree transformer_lm_u16_buffer");
    }
    for (std::uint16_t* ptr : pinned_uint16_ptrs) {
        if (ptr != nullptr && cuda_free_host != nullptr) {
            const int status = cuda_free_host(ptr);
            if (status != 0 && error.empty()) {
                error = cuda_error(status, "cudaFreeHost transformer_lm_pinned_u16_buffer");
            }
        }
    }
    if (device_cuda_free_async_count > 0 && cuda_device_synchronize != nullptr) {
        const int status = cuda_device_synchronize();
        if (status != 0 && error.empty()) {
            error = cuda_error(status, "cudaDeviceSynchronize after cudaFreeAsync");
        }
    }
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }
    total_wall_ms = elapsed_ms(total_start_time, Clock::now());
    std::ostringstream setup_timing_json;
    setup_timing_json
        << "    \"setup_timing\": [\n";
    for (std::size_t i = 0; i < setup_timing_records.size(); ++i) {
        const SetupTimingRecord& record = setup_timing_records[i];
        setup_timing_json
            << "      {\"name\": \"" << json_escape(record.name) << "\", "
            << "\"total_ms\": " << record.total_ms << ", "
            << "\"count\": " << record.count << ", "
            << "\"avg_ms\": " << (record.count > 0 ? record.total_ms / static_cast<double>(record.count) : 0.0)
            << "}";
        if (i + 1 != setup_timing_records.size()) {
            setup_timing_json << ",";
        }
        setup_timing_json << "\n";
    }
    setup_timing_json << "    ],\n";
    std::ostringstream stage_timing_json;
    stage_timing_json
        << "    \"stage_timing_enabled\": " << (stage_timing_enabled ? "true" : "false") << ",\n"
        << "    \"stage_timing_max_events\": " << stage_timing_max_events << ",\n"
        << "    \"stage_timing_event_count\": " << stage_timing_event_count << ",\n"
        << "    \"stage_timing_dropped_event_count\": " << stage_timing_dropped_event_count << ",\n"
        << "    \"stage_timing\": [\n";
    for (std::size_t i = 0; i < stage_timing_records.size(); ++i) {
        const StageTimingRecord& record = stage_timing_records[i];
        stage_timing_json
            << "      {\"name\": \"" << json_escape(record.name) << "\", "
            << "\"total_ms\": " << record.total_ms << ", "
            << "\"count\": " << record.count << ", "
            << "\"avg_ms\": " << (record.count > 0 ? record.total_ms / static_cast<double>(record.count) : 0.0)
            << "}";
        if (i + 1 != stage_timing_records.size()) {
            stage_timing_json << ",";
        }
        stage_timing_json << "\n";
    }
    stage_timing_json << "    ]\n";

    auto linear_shape_path_name = [](int path) -> const char* {
        switch (path) {
            case 1:
                return "cublaslt";
            case 2:
                return "tk_bf16";
            case 3:
                return "tk_bf16_float_out";
            case 4:
                return "cublas_gemmex_bf16";
            case 5:
                return "cublas_sgemm";
            default:
                return "unknown";
        }
    };
    auto linear_shape_op_name = [](int op) -> const char* {
        switch (op) {
            case 0:
                return "N";
            case 1:
                return "T";
            case 2:
                return "C";
            default:
                return "?";
        }
    };
    std::ostringstream linear_shape_stats_json;
    linear_shape_stats_json << "  \"linear_shape_stats\": [\n";
    for (std::size_t i = 0; i < linear_shape_stats.size(); ++i) {
        const LinearShapeStat& stat = linear_shape_stats[i];
        linear_shape_stats_json
            << "    {\"path\": " << stat.path
            << ", \"path_name\": \"" << linear_shape_path_name(stat.path) << "\""
            << ", \"m\": " << stat.m
            << ", \"n\": " << stat.n
            << ", \"k\": " << stat.k
            << ", \"op_a\": " << stat.op_a
            << ", \"op_a_name\": \"" << linear_shape_op_name(stat.op_a) << "\""
            << ", \"op_b\": " << stat.op_b
            << ", \"op_b_name\": \"" << linear_shape_op_name(stat.op_b) << "\""
            << ", \"calls\": " << stat.calls
            << "}";
        if (i + 1 != linear_shape_stats.size()) {
            linear_shape_stats_json << ",";
        }
        linear_shape_stats_json << "\n";
    }
    linear_shape_stats_json << "  ],\n";

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(cfg.model_family) << "\",\n"
        << "  \"backend\": \"tile-cuda\",\n"
        << "  \"template_name\": \"" << json_escape(normalize_template_name(cfg.template_name)) << "\",\n"
        << "  \"resolved_native_template_name\": \"" << json_escape(resolved_native_template_name(cfg.template_name)) << "\",\n"
        << "  \"graph_file\": \"" << json_escape(cfg.graph_file) << "\",\n"
        << "  \"architecture_source\": \"" << json_escape(selected_architecture_source(cfg)) << "\",\n"
        << "  \"architecture_contract\": \"" << json_escape(dense_gpt_architecture_contract(cfg)) << "\",\n"
        << "  \"model_family_context_policy\": \"" << json_escape(model_family_context_policy(cfg)) << "\",\n"
        << "  \"native_cuda_activation\": \"" << json_escape(cfg.activation) << "\",\n"
        << "  \"selected_graph_support_status\": \"" << json_escape(selected_graph_support_status(cfg)) << "\",\n"
        << "  \"selected_graph_native_runnable\": " << (selected_graph_is_native_runnable(cfg) ? "true" : "false") << ",\n"
        << "  \"status\": \""
        << (passed ? (cfg.startup_only ? "native-transformer-lm-startup-ready" : "native-transformer-lm-trained")
                   : "native-transformer-lm-failed")
        << "\",\n"
        << "  \"timing\": {\n"
        << "    \"clock\": \"steady_clock_host_wall_ms\",\n"
        << "    \"setup_wall_ms\": " << setup_wall_ms << ",\n"
        << "    \"train_loop_wall_ms\": " << train_loop_wall_ms << ",\n"
        << "    \"validation_wall_ms\": " << validation_wall_ms << ",\n"
        << "    \"train_compute_wall_ms\": " << (train_loop_wall_ms - validation_wall_ms) << ",\n"
        << "    \"checkpoint_wall_ms\": " << checkpoint_wall_ms << ",\n"
        << "    \"total_wall_ms\": " << total_wall_ms << ",\n"
        << "    \"optimizer_steps_per_second\": " << (train_loop_wall_ms > 0.0 ? (static_cast<double>(steps_completed) * 1000.0 / train_loop_wall_ms) : 0.0) << ",\n"
        << "    \"train_tokens_per_second\": " << (train_loop_wall_ms > 0.0 ? (static_cast<double>(tokens_processed) * 1000.0 / train_loop_wall_ms) : 0.0) << ",\n"
        << setup_timing_json.str()
        << stage_timing_json.str()
        << "  },\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"cuda_module_loading\": \"" << json_escape(env_or_empty("CUDA_MODULE_LOADING")) << "\",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_preflight\": {\n"
        << "    \"checked\": " << (cuda_runtime_preflight_checked ? "true" : "false") << ",\n"
        << "    \"runtime_version\": " << cuda_runtime_version << ",\n"
        << "    \"runtime_version_string\": \"" << json_escape(cuda_version_string(cuda_runtime_version)) << "\",\n"
        << "    \"runtime_version_status\": " << cuda_runtime_version_status << ",\n"
        << "    \"driver_version\": " << cuda_driver_version << ",\n"
        << "    \"driver_version_string\": \"" << json_escape(cuda_version_string(cuda_driver_version)) << "\",\n"
        << "    \"driver_version_status\": " << cuda_driver_version_status << "\n"
        << "  },\n"
        << "  \"token_shards\": " << neuralfn::native_train::token_shard_dataset_json(dataset, &batch_plan) << ",\n"
        << "  \"batch_size\": " << batch_size << ",\n"
        << "  \"seq_len\": " << seq_len << ",\n"
        << "  \"rows\": " << rows << ",\n"
        << "  \"trained_layers\": " << trained_layers << ",\n"
        << "  \"target_layers\": " << target_layers << ",\n"
        << "  \"vocab\": " << kVocab << ",\n"
        << "  \"padded_vocab\": " << kPaddedVocab << ",\n"
        << "  \"lm_head_public_vocab_ce_enabled\": "
        << (lm_head_public_vocab_ce_enabled ? "true" : "false") << ",\n"
        << "  \"lm_head_softmax_vocab\": "
        << (lm_head_public_vocab_ce_enabled ? kVocab : kPaddedVocab) << ",\n"
        << "  \"lm_head_logit_row_stride\": " << kPaddedVocab << ",\n"
        << "  \"lm_head_padded_dlogits_zeroed\": "
        << (lm_head_public_vocab_ce_enabled ? "true" : "false") << ",\n"
        << "  \"model_dim\": " << kDim << ",\n"
        << "  \"hidden_dim\": " << kHidden << ",\n"
        << "  \"lm_head_row_chunk_size\": " << lm_head_chunk_rows << ",\n"
        << "  \"lm_head_row_chunk_count\": " << lm_head_chunk_count << ",\n"
        << "  \"loss_partial_count\": " << loss_partial_count << ",\n"
        << "  \"logit_workspace_elements\": " << logit_workspace_elements << ",\n"
        << "  \"grad_logit_workspace_elements\": 0,\n"
        << "  \"lm_head_training_logits_dtype\": \""
        << (lm_head_bf16_logits_enabled ? "bf16" : "float32") << "\",\n"
        << "  \"lm_head_training_dlogits_dtype\": \""
        << (lm_head_bf16_logits_enabled ? "bf16" : "float32") << "\",\n"
        << "  \"lm_head_loss_logits_dtype\": \""
        << (lm_head_bf16_loss_enabled ? "bf16" : "float32") << "\",\n"
        << "  \"lm_head_bf16_loss_enabled\": "
        << (lm_head_bf16_loss_enabled ? "true" : "false") << ",\n"
        << "  \"lm_head_bf16_logits_enabled\": "
        << (lm_head_bf16_logits_enabled ? "true" : "false") << ",\n"
        << "  \"lm_head_bf16_logit_elements\": " << lm_head_bf16_logit_elements << ",\n"
        << "  \"lm_head_bf16_logit_bytes\": " << lm_head_bf16_logit_bytes << ",\n"
        << "  \"lm_head_bf16_dweight_enabled\": "
        << (lm_head_bf16_dweight_enabled ? "true" : "false") << ",\n"
        << "  \"lm_head_prepack_bf16_hidden_enabled\": "
        << (lm_head_prepack_bf16_hidden_enabled ? "true" : "false") << ",\n"
        << "  \"lm_head_bf16_hidden_elements\": " << lm_head_bf16_hidden_elements << ",\n"
        << "  \"lm_head_bf16_hidden_bytes\": " << lm_head_bf16_hidden_bytes << ",\n"
        << "  \"lm_head_dweight_input_dtype\": \""
        << ((lm_head_prepack_bf16_hidden_enabled || lm_head_bf16_dweight_enabled)
                ? "bf16"
                : (lm_head_bf16_logits_enabled ? "float32" : "float32")) << "\",\n"
        << "  \"lm_head_dweight_strategy\": \""
        << (lm_head_prepack_bf16_hidden_enabled
                ? (dweight_first_microbatch_beta_zero_enabled
                       ? "full-final-norm-bf16-prepack-bf16-dlogit-dweight-first-write-then-accumulate"
                       : "full-final-norm-bf16-prepack-bf16-dlogit-dweight-accumulate")
                : (lm_head_bf16_dweight_enabled
                ? (dweight_first_microbatch_beta_zero_enabled
                       ? "chunked-final-norm-bf16-pack-bf16-dlogit-dweight-first-write-then-accumulate"
                       : "chunked-final-norm-bf16-pack-bf16-dlogit-dweight-accumulate")
                : (lm_head_bf16_logits_enabled
                       ? "float32-hidden-bf16-dlogit-dweight-accumulate"
                       : "float32-hidden-float32-dlogit-dweight-accumulate")))
        << "\",\n"
        << "  \"dweight_first_microbatch_beta_zero_enabled\": "
        << (dweight_first_microbatch_beta_zero_enabled ? "true" : "false") << ",\n"
        << "  \"dweight_first_microbatch_beta_strategy\": \""
        << (dweight_first_microbatch_beta_zero_enabled
                ? "first-gradient-accumulation-microbatch-uses-gemm-beta-zero"
                : "all-gradient-accumulation-microbatches-use-gemm-beta-one")
        << "\",\n"
        << "  \"lm_head_ce_backward_strategy\": \""
        << (lm_head_public_vocab_ce_enabled
                ? (lm_head_bf16_logits_enabled
                       ? "public-vocab-strided-fused-row-bf16-logits-dlogits"
                       : "public-vocab-strided-inplace-logits-dlogits-workspace")
                : (lm_head_bf16_logits_enabled
                       ? "padded-vocab-fused-row-bf16-logits-dlogits"
                       : "padded-vocab-inplace-logits-dlogits-workspace"))
        << "\",\n"
        << "  \"lm_head_grad_logits_workspace_allocated\": false,\n"
        << "  \"linear_backend_strategy\": \""
        << (lm_head_bf16_logits_enabled && linear_tk_gemm_count > 0 && linear_cublaslt_gemm_count > 0
                ? "block-bf16-cublaslt-shape-gated-lm-head-tk-sm120-default"
                : (lm_head_bf16_logits_enabled && linear_tk_gemm_count > 0
                ? "block-bf16-gemmex-lm-head-tk-sm120-fallback"
                : (lm_head_bf16_logits_enabled && linear_bf16_gemm_count > 0
                ? "block-and-lm-head-bf16-gemmex-fallback"
                : (linear_bf16_gemm_count > 0 && linear_cublaslt_gemm_count > 0
                ? "block-forward-dinput-dweight-bf16-lm-head-tf32-cublaslt-opt-in"
                : (linear_bf16_gemm_count > 0 && linear_sgemm_count > 0
                       ? "block-forward-dinput-dweight-bf16-lm-head-tf32-sgemm-default"
                : (linear_bf16_gemm_count > 0
                       ? "bf16-gemmex-float32-output"
                : (linear_cublaslt_gemm_count > 0
                      ? "tf32-cublaslt-optimized-opt-in"
                      : (linear_sgemm_count > 0 ? "tf32-sgemm-optimized" : "not-run"))))))))
        << "\",\n"
        << "  \"block_forward_linear_strategy\": \""
        << (linear_cublaslt_gemm_count > 0 ? "bf16-shadow-weight-shape-gated-cublaslt-forward"
                                           : "bf16-shadow-weight-gemmex-forward")
        << "\",\n"
        << "  \"block_backward_input_linear_strategy\": \""
        << (linear_cublaslt_gemm_count > 0 ? "bf16-shadow-weight-shape-gated-cublaslt-dinput"
                                           : "bf16-shadow-weight-gemmex-dinput")
        << "\",\n"
        << "  \"block_backward_mlp_proj_dgelu_fusion_enabled\": "
        << (fuse_mlp_proj_dgelu_enabled ? "true" : "false") << ",\n"
        << "  \"block_backward_bf16_mlp_grad_handoff_enabled\": "
        << (bf16_mlp_grad_handoff_enabled ? "true" : "false") << ",\n"
        << "  \"block_backward_mlp_dgelu_float_grad_elided\": "
        << (elide_mlp_dgelu_float_grad_enabled ? "true" : "false") << ",\n"
        << "  \"block_backward_mlp_proj_dgelu_strategy\": \""
        << (elide_mlp_dgelu_float_grad_enabled
                ? "tk-sm120-fused-dinput-dgelu-bf16-store-bf16-shadow-weight-bf16-grad-handoff-no-float-grad"
                : (bf16_mlp_grad_handoff_enabled
                ? "tk-sm120-fused-dinput-dgelu-bf16-store-bf16-shadow-weight-bf16-grad-handoff-float-grad"
                : (fuse_mlp_proj_dgelu_enabled
                       ? "tk-sm120-fused-dinput-dgelu-bf16-store-bf16-shadow-weight-float32-grad"
                       : "separate-dinput-plus-gelu")))
        << "\",\n"
        << "  \"block_backward_bf16_qkv_dweight_enabled\": "
        << (bf16_qkv_dweight_enabled ? "true" : "false") << ",\n"
        << "  \"block_backward_qkv_dweight_strategy\": \""
        << (bf16_qkv_dweight_enabled
                ? (dweight_first_microbatch_beta_zero_enabled
                       ? "packed-ln1-bf16-qkv-bf16-grad-dweight-bias-first-write-then-accumulate"
                       : "packed-ln1-bf16-qkv-bf16-grad-dweight-bias-accumulate")
                : (dweight_first_microbatch_beta_zero_enabled
                       ? "float32-ln1-bf16-qkv-grad-dweight-bias-first-write-then-accumulate"
                       : "float32-ln1-bf16-qkv-grad-dweight-bias-accumulate"))
        << "\",\n"
        << "  \"block_dweight_bf16_staging_enabled\": "
        << (bf16_block_dweight_staging_enabled ? "true" : "false") << ",\n"
        << "  \"block_dweight_bf16_staging_elements\": " << block_dweight_bf16_staging_elements << ",\n"
        << "  \"block_dweight_bf16_staging_bytes\": " << block_dweight_bf16_staging_bytes << ",\n"
        << "  \"block_dweight_bf16_staging_zero_count\": " << block_dweight_bf16_staging_zero_count << ",\n"
        << "  \"block_dweight_bf16_staging_convert_kernel_launches\": "
        << block_dweight_bf16_staging_convert_kernel_launches << ",\n"
        << "  \"block_dweight_bf16_staging_strategy\": \""
        << (bf16_block_dweight_staging_enabled
                ? (bf16_block_weight_param_update_enabled
                       ? "qkv-fc-bf16-dweight-staging-direct-bf16-param-adamw"
                       : "qkv-fc-bf16-dweight-staging-flush-to-float32")
                : "disabled-fp32-accumulation-default")
        << "\",\n"
        << "  \"block_backward_weight_linear_strategy\": \""
        << (linear_cublaslt_gemm_count > 0
                ? (dweight_first_microbatch_beta_zero_enabled
                       ? "shape-gated-bf16-cublaslt-dweight-bgrad-first-write-then-accumulate"
                       : "shape-gated-bf16-cublaslt-dweight-bgrad-accumulate")
                : "forced-bf16-gemmex-dweight-plus-bias-accumulate-fallback")
        << "\",\n"
        << "  \"non_block_forward_backward_linear_strategy\": \""
        << (lm_head_bf16_logits_enabled && linear_tk_gemm_count > 0
                ? "padded-lm-head-tk-sm120-bf16-gemm-default"
                : (lm_head_bf16_logits_enabled
                ? "padded-lm-head-bf16-gemmex-fallback"
                : (linear_cublaslt_gemm_count > 0
                       ? "padded-lm-head-tf32-cublaslt-optimized-opt-in"
                       : "padded-lm-head-tf32-sgemm-optimized-default")))
        << "\",\n"
        << "  \"lm_head_logits_linear_strategy\": \""
        << (lm_head_bf16_logits_enabled && linear_tk_gemm_count > 0
                ? "tk-sm120-bf16-gemm-default"
                : (lm_head_bf16_logits_enabled
                       ? "bf16-gemmex-fallback"
                       : "tf32-sgemm-or-cublaslt"))
        << "\",\n"
        << "  \"linear_bf16_gemm_count\": " << linear_bf16_gemm_count << ",\n"
        << "  \"linear_tk_gemm_count\": " << linear_tk_gemm_count << ",\n"
        << "  \"linear_tk_float_out_gemm_count\": " << linear_tk_float_out_gemm_count << ",\n"
        << "  \"linear_cublaslt_gemm_count\": " << linear_cublaslt_gemm_count << ",\n"
        << "  \"linear_cublaslt_descriptor_cache_enabled\": "
        << (linear_cublaslt_descriptor_cache_enabled ? "true" : "false") << ",\n"
        << "  \"linear_sgemm_count\": " << linear_sgemm_count << ",\n"
        << "  \"linear_bf16_a_pack_count\": " << linear_bf16_a_pack_count << ",\n"
        << "  \"linear_bf16_a_cache_hit_count\": " << linear_bf16_a_cache_hit_count << ",\n"
        << "  \"linear_bf16_a_cache_strategy\": \""
        << (linear_bf16_a_cache_hit_count > 0 ? "cached-first-gemm-operand-with-optimizer-reset" : "unused")
        << "\",\n"
        << "  \"linear_bf16_cache_reset_count\": " << linear_bf16_cache_reset_count << ",\n"
        << "  \"linear_bf16_workspace_allocation_strategy\": \""
        << (linear_bf16_workspace_allocation_count > 0 ? "cached-process-bf16-workspace" : "unused")
        << "\",\n"
        << "  \"linear_bf16_workspace_allocation_count\": " << linear_bf16_workspace_allocation_count << ",\n"
        << "  \"linear_bf16_workspace_a_capacity\": " << linear_bf16_workspace_a_capacity << ",\n"
        << "  \"linear_bf16_workspace_b_capacity\": " << linear_bf16_workspace_b_capacity << ",\n"
        << "  \"linear_bf16_cached_a_capacity\": " << linear_bf16_cached_a_capacity << ",\n"
        << "  \"linear_bf16_cache_entry_count\": " << linear_bf16_cache_entry_count << ",\n"
        << "  \"linear_shape_stats_enabled\": "
        << (!linear_shape_stats.empty() ? "true" : "false") << ",\n"
        << "  \"linear_shape_stats_count\": " << linear_shape_stats.size() << ",\n"
        << linear_shape_stats_json.str()
        << "  \"block_weight_bf16_shadow_strategy\": \""
        << (bf16_block_weight_param_update_enabled
                ? "persistent-bf16-primary-block-weight-adamw"
                : (fuse_adamw_bf16_shadow_refresh_enabled
                ? "persistent-fp32-master-bf16-shadow-fused-adamw-refresh"
	                   : "persistent-fp32-master-bf16-shadow-refresh-after-adamw"))
        << "\",\n"
        << "  \"block_weight_bf16_initialization_strategy\": \""
        << (direct_bf16_block_weight_init_enabled
                ? "direct-bf16-fill-many-values"
                : "float32-fill-then-bf16-pack")
        << "\",\n"
        << "  \"block_weight_bf16_primary_param_update_enabled\": "
        << (bf16_block_weight_param_update_enabled ? "true" : "false") << ",\n"
        << "  \"direct_bf16_block_weight_initialization_enabled\": "
        << (direct_bf16_block_weight_init_enabled ? "true" : "false") << ",\n"
        << "  \"block_weight_bf16_gradient_storage_strategy\": "
        << (adamw_bf16_param_bf16_grad_descriptor_count > 0
                ? "\"qkv-fc-bf16-accumulation-buffer\""
                : "\"float32-accumulation-buffer\"")
        << ",\n"
        << "  \"adamw_bf16_param_bf16_grad_kernel_loaded\": "
        << (adamw_many_with_device_scale_bf16_param_bf16_grad != nullptr ? "true" : "false") << ",\n"
        << "  \"block_weight_bf16_shadow_fused_adamw_refresh_enabled\": "
        << (fuse_adamw_bf16_shadow_refresh_enabled ? "true" : "false") << ",\n"
        << "  \"block_weight_bf16_shadow_elements\": " << block_weight_bf16_arena_elements << ",\n"
        << "  \"block_weight_bf16_shadow_bytes\": " << block_weight_bf16_arena_bytes << ",\n"
        << "  \"block_weight_bf16_shadow_descriptor_count\": " << block_weight_bf16_descriptor_count << ",\n"
        << "  \"block_weight_bf16_shadow_max_elements\": " << block_weight_bf16_max_elements << ",\n"
        << "  \"block_weight_bf16_refresh_count\": " << block_weight_bf16_refresh_count << ",\n"
        << "  \"block_weight_bf16_fused_adamw_refresh_count\": " << block_weight_bf16_fused_adamw_refresh_count << ",\n"
        << "  \"token_weight_bf16_shadow_enabled\": " << (token_weight_bf16_shadow_enabled ? "true" : "false") << ",\n"
        << "  \"token_weight_bf16_initial_refresh_fusion_enabled\": "
        << (fuse_token_weight_bf16_initial_refresh_enabled ? "true" : "false") << ",\n"
        << "  \"token_weight_bf16_refresh_count\": " << token_weight_bf16_refresh_count << ",\n"
        << "  \"block_weight_bf16_primary_param_update_count\": " << adamw_bf16_param_kernel_launches << ",\n"
        << "  \"block_weight_bf16_primary_param_bf16_grad_update_count\": "
        << adamw_bf16_param_bf16_grad_kernel_launches << ",\n"
        << "  \"attention_backend_strategy\": \""
        << (attention_forward_tk_launches > 0 || attention_backward_tk_launches > 0
                ? "tk-sm120-bf16-bridge"
                : "tile-row-float32")
        << "\",\n"
        << "  \"attention_forward_strategy\": \""
        << (packed_qkv_attention_enabled
                ? "tk-sm120-packed-qkv-bf16-flashattention"
                : (attention_forward_tk_launches > 0 ? "tk-sm120-bf16-flashattention-bridge" : "row-vector-tile-score-reuse"))
        << "\",\n"
        << "  \"attention_forward_tk_launch_count\": " << attention_forward_tk_launches << ",\n"
        << "  \"attention_tk_workspace_allocation_strategy\": \""
        << (attention_forward_tk_launches > 0 || attention_backward_tk_launches > 0
                ? "cached-process-bf16-workspace"
                : "unused")
        << "\",\n"
        << "  \"attention_tk_workspace_allocation_count\": " << attention_tk_workspace_allocations << ",\n"
        << "  \"attention_tk_workspace_element_capacity\": " << attention_tk_workspace_element_capacity << ",\n"
        << "  \"attention_tk_workspace_row_capacity\": " << attention_tk_workspace_row_capacity << ",\n"
        << "  \"attention_forward_row_count\": " << (rows * kHeads) << ",\n"
        << "  \"attention_forward_scalar_output_count\": " << (rows * kDim) << ",\n"
        << "  \"attention_forward_score_reuse_value_dim\": " << kAttentionForwardValueReuse << ",\n"
        << "  \"attention_forward_scalar_cta_elision_factor\": " << kAttentionForwardValueReuse << ",\n"
        << "  \"attention_forward_value_chunk_size\": " << kAttentionForwardValueReuse << ",\n"
        << "  \"attention_forward_scalar_launch_fallback_enabled\": true,\n"
        << "  \"attention_forward_row_launch_auto_disable_enabled\": true,\n"
        << "  \"attention_forward_row_launch_auto_disabled\": " << (attention_forward_row_fallbacks > 0 ? "true" : "false") << ",\n"
        << "  \"attention_forward_row_launch_count\": " << attention_forward_row_launches << ",\n"
        << "  \"attention_forward_row_launch_success_count\": " << attention_forward_row_successes << ",\n"
        << "  \"attention_forward_row_launch_fallback_count\": " << attention_forward_row_fallbacks << ",\n"
        << "  \"attention_forward_row_launch_last_error_code\": " << attention_forward_row_last_error_code << ",\n"
        << "  \"attention_forward_row_prelaunch_clear_error_code\": " << attention_forward_row_prelaunch_clear_error_code << ",\n"
        << "  \"attention_forward_row_prelaunch_peek_error_code\": " << attention_forward_row_prelaunch_peek_error_code << ",\n"
        << "  \"attention_forward_row_launch_grid_x\": " << attention_forward_row_grid_x << ",\n"
        << "  \"attention_forward_row_launch_grid_y\": " << attention_forward_row_grid_y << ",\n"
        << "  \"attention_forward_row_launch_grid_z\": " << attention_forward_row_grid_z << ",\n"
        << "  \"attention_forward_row_launch_block_x\": " << attention_forward_row_block_x << ",\n"
        << "  \"attention_forward_row_kernel_attr_status_code\": " << attention_forward_row_attr_status_code << ",\n"
        << "  \"attention_forward_row_kernel_attr_max_threads_per_block\": " << attention_forward_row_attr_max_threads_per_block << ",\n"
        << "  \"attention_forward_row_kernel_attr_num_regs\": " << attention_forward_row_attr_num_regs << ",\n"
        << "  \"attention_forward_row_kernel_attr_shared_size_bytes\": " << attention_forward_row_attr_shared_size_bytes << ",\n"
        << "  \"attention_forward_row_kernel_attr_const_size_bytes\": " << attention_forward_row_attr_const_size_bytes << ",\n"
        << "  \"attention_forward_row_kernel_attr_local_size_bytes\": " << attention_forward_row_attr_local_size_bytes << ",\n"
        << "  \"attention_forward_scalar_launch_count\": " << attention_forward_scalar_launches << ",\n"
        << "  \"packed_qkv_attention_enabled\": " << (packed_qkv_attention_enabled ? "true" : "false") << ",\n"
        << "  \"packed_qkv_attention_bf16_elements\": " << packed_qkv_attention_bf16_elements << ",\n"
        << "  \"packed_qkv_attention_bf16_bytes\": " << packed_qkv_attention_bf16_bytes << ",\n"
        << "  \"packed_qkv_float_attention_tape_elided\": "
        << (packed_qkv_float_attention_tape_elided ? "true" : "false") << ",\n"
        << "  \"packed_qkv_float_attention_tape_elements_elided\": "
        << packed_qkv_float_attention_tape_elements_elided << ",\n"
        << "  \"packed_qkv_float_attention_tape_bytes_elided\": "
        << (packed_qkv_float_attention_tape_elements_elided * static_cast<std::int64_t>(sizeof(float))) << ",\n"
        << "  \"qkv_forward_layout_strategy\": \""
        << (packed_qkv_attention_enabled ? "packed-qkv-bf16-no-split" : "fused-split-to-heads")
        << "\",\n"
        << "  \"qkv_forward_ln1_bf16_enabled\": "
        << (ln1_bf16_qkv_forward_enabled ? "true" : "false") << ",\n"
        << "  \"qkv_forward_layout_kernel_launches_per_block\": " << (packed_qkv_attention_enabled ? 0 : 1) << ",\n"
        << "  \"qkv_forward_layout_legacy_launches_per_block\": 4,\n"
        << "  \"qkv_forward_layout_launches_elided_per_block\": " << (packed_qkv_attention_enabled ? 4 : 3) << ",\n"
        << "  \"qkv_bias_layout_strategy\": \""
        << (packed_qkv_attention_enabled
                ? (fuse_qkv_bias_tk_gemm_enabled
                       ? "packed-qkv-bf16-bias-fused-tk-gemm"
                       : "packed-qkv-bf16-bias-inplace")
                : "fused-qkv-bias-split-to-heads")
        << "\",\n"
        << "  \"qkv_bias_fused_tk_gemm_enabled\": " << (fuse_qkv_bias_tk_gemm_enabled ? "true" : "false") << ",\n"
        << "  \"qkv_bias_layout_kernel_launches_per_block\": "
        << (packed_qkv_attention_enabled && fuse_qkv_bias_tk_gemm_enabled ? 0 : 1) << ",\n"
        << "  \"qkv_bias_layout_legacy_launches_per_block\": 2,\n"
        << "  \"qkv_bias_layout_launches_elided_per_block\": "
        << (packed_qkv_attention_enabled && fuse_qkv_bias_tk_gemm_enabled ? 2 : 1) << ",\n"
        << "  \"qkv_backward_layout_strategy\": \""
        << (packed_qkv_attention_enabled
                ? (bf16_qkv_grad_handoff_enabled ? "packed-qkv-bf16-gradient-handoff" : "packed-qkv-bf16-gradient-unpack")
                : "fused-heads-to-qkv")
        << "\",\n"
        << "  \"qkv_backward_layout_kernel_launches_per_block\": 1,\n"
        << "  \"qkv_backward_layout_legacy_launches_per_block\": 4,\n"
        << "  \"qkv_backward_layout_launches_elided_per_block\": 3,\n"
        << "  \"attention_backward_bf16_qkv_grad_handoff_enabled\": "
        << (bf16_qkv_grad_handoff_enabled ? "true" : "false") << ",\n"
        << "  \"attention_backward_bf16_grad_out_handoff_enabled\": "
        << (bf16_attention_grad_out_handoff_enabled ? "true" : "false") << ",\n"
        << "  \"attention_backward_grad_out_dtype\": \""
        << (bf16_attention_grad_out_handoff_enabled ? "bf16" : "float32") << "\",\n"
        << "  \"attention_backward_bf16_grad_out_scratch_elements\": "
        << attention_grad_out_bf16_elements << ",\n"
        << "  \"attention_backward_bf16_grad_out_scratch_bytes\": "
        << attention_grad_out_bf16_bytes << ",\n"
        << "  \"attention_backward_qkv_float_grad_scratch_elided\": "
        << (grad_qkv_float_scratch_elided ? "true" : "false") << ",\n"
        << "  \"attention_backward_qkv_float_grad_scratch_elements\": "
        << grad_qkv_float_scratch_elements << ",\n"
        << "  \"attention_backward_qkv_float_grad_scratch_bytes_elided\": "
        << grad_qkv_float_scratch_bytes_elided << ",\n"
        << "  \"attention_backward_qkv_float_grad_scratch_strategy\": \""
        << (grad_qkv_float_scratch_elided
                ? "elided-bf16-qkv-grad-handoff"
                : "float32-qkv-grad-expansion")
        << "\",\n"
        << "  \"attention_backward_direct_bf16_qkv_grad_scratch_enabled\": "
        << (direct_bf16_qkv_grad_scratch_enabled ? "true" : "false") << ",\n"
        << "  \"attention_backward_direct_bf16_qkv_grad_scratch_elements\": "
        << (direct_bf16_qkv_grad_scratch_enabled ? qkv_activation_elements : 0) << ",\n"
        << "  \"attention_backward_qkv_bridge_strategy\": \""
        << (packed_qkv_attention_enabled
                ? (bf16_attention_grad_out_handoff_enabled
                       ? "tk-sm120-packed-qkv-bf16-grad-out-direct-bf16-qkv-handoff"
                   : bf16_qkv_grad_handoff_enabled
                       ? (direct_bf16_qkv_grad_scratch_enabled
                              ? "tk-sm120-packed-qkv-direct-bf16-grad-scratch-handoff"
                              : "tk-sm120-packed-qkv-packed-bf16-grad-handoff")
                       : "tk-sm120-packed-qkv-packed-grad-bridge")
                : "fused-bf16-heads-to-row-qkv")
        << "\",\n"
        << "  \"attention_backward_qkv_bridge_kernel_launches_per_block\": " << (packed_qkv_attention_enabled ? 2 : 1) << ",\n"
        << "  \"attention_backward_qkv_bridge_legacy_launches_per_block\": 4,\n"
        << "  \"attention_backward_qkv_bridge_launches_elided_per_block\": 3,\n"
        << "  \"bf16_projection_residual_enabled\": "
        << (bf16_projection_residual_enabled ? "true" : "false") << ",\n"
        << "  \"attention_projection_input_strategy\": \""
        << (bf16_projection_residual_enabled
                ? (packed_qkv_attention_enabled
                       ? "packed-o-bf16-direct-gemm-bf16-residual-consumer"
                       : "float32-attention-output-bf16-gemm-bf16-residual-consumer")
                : (packed_qkv_attention_enabled ? "packed-o-bf16-direct-gemm" : "float32-attention-output-bf16-gemm"))
        << "\",\n"
        << "  \"attention_packed_output_unpack_strategy\": \""
        << (packed_qkv_attention_enabled ? "elided-direct-bf16-projection" : "not-packed")
        << "\",\n"
        << "  \"mlp_fc_bias_gelu_strategy\": \"fused-bias-preactivation-gelu\",\n"
        << "  \"mlp_fc_bias_gelu_kernel_launches_per_block\": 1,\n"
        << "  \"mlp_fc_bias_gelu_legacy_launches_per_block\": 2,\n"
        << "  \"mlp_fc_bias_gelu_launches_elided_per_block\": 1,\n"
        << "  \"mlp_proj_forward_activation_strategy\": \""
        << (bf16_projection_residual_enabled
                ? "fused-gelu-bf16-act-direct-bf16-output-gemm"
                : "fused-gelu-bf16-act-direct-gemm")
        << "\",\n"
        << "  \"mlp_forward_act_bf16_elements\": " << mlp_forward_act_bf16_elements << ",\n"
        << "  \"mlp_forward_act_bf16_bytes\": " << mlp_forward_act_bf16_bytes << ",\n"
        << "  \"projection_bf16_scratch_elements\": " << projection_bf16_scratch_elements << ",\n"
        << "  \"projection_bf16_scratch_bytes\": " << projection_bf16_scratch_bytes << ",\n"
        << "  \"projection_bias_residual_strategy\": \""
        << (bf16_projection_residual_enabled
                ? "fused-bf16-linear-bias-residual-add"
                : "fused-linear-bias-residual-add")
        << "\",\n"
        << "  \"fused_ln2_bf16_norm_float_store_elision_enabled\": "
        << (fused_ln2_bf16_norm_float_store_elision_enabled ? "true" : "false") << ",\n"
        << "  \"stored_mlp_ln2_bf16_float_store_elided_count\": "
        << stored_mlp_ln2_bf16_float_store_elided_count << ",\n"
        << "  \"stored_mlp_ln2_bf16_float_store_elided_elements\": "
        << stored_mlp_ln2_bf16_float_store_elided_elements << ",\n"
        << "  \"attention_residual_ln2_strategy\": \""
        << (fuse_attention_residual_ln2_enabled
                ? (bf16_projection_residual_enabled
                       ? (fused_ln2_bf16_norm_float_store_elision_enabled
                              ? "fused-bf16-linear-bias-residual-layernorm-bf16-norm-fp32-store-elided"
                              : "fused-bf16-linear-bias-residual-layernorm")
                       : "fused-linear-bias-residual-layernorm")
                : "disabled")
        << "\",\n"
        << "  \"attention_residual_ln2_kernel_launches_per_block\": "
        << (fuse_attention_residual_ln2_enabled ? 1 : 0) << ",\n"
        << "  \"attention_residual_ln2_legacy_launches_per_block\": 2,\n"
        << "  \"attention_residual_ln2_launches_elided_per_block\": "
        << (fuse_attention_residual_ln2_enabled ? 1 : 0) << ",\n"
        << "  \"projection_bias_residual_kernel_launches_per_block\": 2,\n"
        << "  \"projection_bias_residual_legacy_launches_per_block\": 4,\n"
        << "  \"projection_bias_residual_launches_elided_per_block\": 2,\n"
        << "  \"attention_backward_grad_layout_strategy\": \"merged-grad-out-direct\",\n"
        << "  \"attention_backward_grad_layout_kernel_launches_per_block\": 0,\n"
        << "  \"attention_backward_grad_layout_legacy_launches_per_block\": 1,\n"
        << "  \"attention_backward_grad_layout_launches_elided_per_block\": 1,\n"
        << "  \"attention_backward_strategy\": \""
        << (packed_qkv_attention_enabled
                ? (stored_packed_attention_backward_kernel_launches > 0
                       ? (bf16_attention_grad_out_handoff_enabled
                              ? "tk-sm120-packed-qkv-bf16-saved-activation-backward-bf16-grad-out-handoff"
                          : bf16_qkv_grad_handoff_enabled
                              ? (direct_bf16_qkv_grad_scratch_enabled
                                     ? "tk-sm120-packed-qkv-bf16-saved-activation-backward-direct-bf16-grad-scratch-handoff"
                                     : "tk-sm120-packed-qkv-bf16-saved-activation-backward-bf16-grad-handoff")
                              : "tk-sm120-packed-qkv-bf16-saved-activation-backward-bridge")
                       : (bf16_attention_grad_out_handoff_enabled
                              ? "tk-sm120-packed-qkv-bf16-backward-bf16-grad-out-handoff"
                          : bf16_qkv_grad_handoff_enabled
                              ? (direct_bf16_qkv_grad_scratch_enabled
                                     ? "tk-sm120-packed-qkv-bf16-backward-direct-bf16-grad-scratch-handoff"
                                     : "tk-sm120-packed-qkv-bf16-backward-bf16-grad-handoff")
                              : "tk-sm120-packed-qkv-bf16-backward-bridge"))
                : stored_attention_backward_kernel_launches > 0
                ? "tk-sm120-bf16-saved-forward-workspace-bridge"
                : attention_backward_tk_launches > 0
                ? "tk-sm120-bf16-reuse-forward-workspace-bridge"
                : "query-row-atomic-tile-score-reuse")
        << "\",\n"
        << "  \"attention_backward_reuses_forward_workspace\": "
        << (attention_backward_tk_launches > 0 ? "true" : "false") << ",\n"
        << "  \"attention_backward_uses_saved_forward_workspace\": "
        << (stored_attention_backward_kernel_launches > 0 || stored_packed_attention_backward_kernel_launches > 0
                ? "true"
                : "false")
        << ",\n"
        << "  \"attention_backward_recompute_forward_elided_per_block\": "
        << (attention_backward_tk_launches > 0 || stored_packed_attention_backward_kernel_launches > 0 ? 1 : 0)
        << ",\n"
        << "  \"attention_backward_tk_launch_count\": " << attention_backward_tk_launches << ",\n"
        << "  \"attention_backward_row_count\": " << (rows * kHeads) << ",\n"
        << "  \"attention_backward_scalar_output_count\": " << (rows * kDim * 3) << ",\n"
        << "  \"attention_backward_score_reuse_dim\": " << kAttentionBackwardDimReuse << ",\n"
        << "  \"attention_backward_scalar_cta_elision_factor\": " << (kAttentionBackwardDimReuse * 3) << ",\n"
        << "  \"train_batch_tokens\": " << cfg.train_batch_tokens << ",\n"
        << "  \"requested_train_batch_tokens\": " << requested_train_batch_tokens << ",\n"
        << "  \"microbatch_tokens\": " << microbatch_tokens << ",\n"
        << "  \"grad_accum_steps\": " << grad_accum_steps << ",\n"
        << "  \"effective_train_batch_tokens\": " << effective_train_batch_tokens << ",\n"
        << "  \"gradient_accumulation_strategy\": \"device-microbatch-average\",\n"
        << "  \"gradient_accumulation_scale\": " << (1.0 / static_cast<double>(grad_accum_steps)) << ",\n"
        << "  \"token_gradient_accumulation_strategy\": \"direct-device-accumulation-buffer\",\n"
        << "  \"token_gradient_scratch_buffer_allocated\": false,\n"
        << "  \"token_gradient_microbatch_full_copy_elided\": true,\n"
        << "  \"token_gradient_microbatch_zero_elided\": true,\n"
        << "  \"block_linear_weight_gradient_accumulation_strategy\": \"direct-device-accumulation-buffer\",\n"
        << "  \"block_linear_weight_gradient_scratch_buffers_allocated\": false,\n"
        << "  \"block_linear_weight_gradient_microbatch_full_copy_elided\": true,\n"
        << "  \"layer_norm_affine_gradient_accumulation_strategy\": \"direct-device-accumulation-buffer\",\n"
        << "  \"layer_norm_affine_gradient_scratch_buffers_allocated\": false,\n"
        << "  \"layer_norm_affine_gradient_microbatch_full_copy_elided\": true,\n"
        << "  \"linear_bias_gradient_accumulation_strategy\": \"direct-device-accumulation-buffer\",\n"
        << "  \"linear_bias_gradient_scratch_buffers_allocated\": false,\n"
        << "  \"linear_bias_gradient_microbatch_full_copy_elided\": true,\n"
        << "  \"position_gradient_accumulation_strategy\": \"direct-device-accumulation-buffer\",\n"
        << "  \"position_gradient_scratch_buffer_allocated\": false,\n"
        << "  \"position_gradient_microbatch_full_copy_elided\": true,\n"
        << "  \"position_gradient_microbatch_zero_elided\": true,\n"
        << "  \"mlp_activation_storage_strategy\": \""
        << (stored_mlp_activation_block_count > 0 ? "bf16-forward-store-direct-backward-opt-in" : "disabled")
        << "\",\n"
        << "  \"reuse_packed_ln2_fc_gelu_enabled\": "
        << (reuse_packed_ln2_fc_gelu_enabled ? "true" : "false") << ",\n"
        << "  \"fused_ln2_bf16_out_enabled\": "
        << (fused_ln2_bf16_out_enabled ? "true" : "false") << ",\n"
        << "  \"stored_mlp_ln2_bf16_prepack_strategy\": \""
        << (stored_mlp_activation_block_count > 0
                ? (reuse_packed_ln2_fc_gelu_enabled
                       ? (fused_ln2_bf16_out_enabled
                              ? "fused-attention-residual-ln2-bf16-store"
                              : "separate-float32-to-bf16-store")
                       : "not-prepacked")
                : "disabled")
        << "\",\n"
        << "  \"stored_mlp_forward_strategy\": \""
        << (stored_mlp_activation_block_count > 0
                ? (reuse_packed_ln2_fc_gelu_enabled
                       ? "tk-sm120-fused-fc-bias-gelu-prepacked-ln2-bf16-shadow-weight"
                       : "tk-sm120-fused-fc-bias-gelu-bf16-store-bf16-shadow-weight")
                : "disabled")
        << "\",\n"
        << "  \"stored_mlp_activation_blocks\": " << stored_mlp_activation_block_count << ",\n"
        << "  \"stored_mlp_activation_elements\": " << stored_mlp_activation_arena_elements << ",\n"
        << "  \"stored_mlp_activation_bytes\": " << stored_mlp_activation_arena_bytes << ",\n"
        << "  \"stored_mlp_layer_norm_stats_elements\": " << stored_mlp_norm_stats_elements << ",\n"
        << "  \"stored_mlp_layer_norm_stats_bytes\": " << stored_mlp_norm_stats_bytes << ",\n"
        << "  \"stored_mlp_ln2_bf16_fused_store_kernel_launches\": "
        << stored_mlp_ln2_bf16_fused_store_kernel_launches << ",\n"
        << "  \"stored_mlp_activation_store_kernel_launches\": " << stored_mlp_activation_store_kernel_launches << ",\n"
        << "  \"stored_mlp_activation_restore_kernel_launches\": " << stored_mlp_activation_restore_kernel_launches << ",\n"
        << "  \"stored_mlp_activation_backward_consumer_strategy\": \""
        << (stored_mlp_activation_block_count > 0
                ? (bf16_mlp_grad_handoff_enabled
                       ? "direct-bf16-dweight-dgelu-and-bf16-grad-handoff"
                       : "direct-bf16-dweight-and-gelu-backward")
                : "disabled")
        << "\",\n"
        << "  \"residual1_activation_storage_strategy\": \""
        << (stored_residual1_block_count > 0
                ? (bf16_residual1_ln_backward_enabled ? "bf16-forward-store-direct-ln-backward"
                                                       : "bf16-forward-store-recompute-restore")
                : "disabled")
        << "\",\n"
        << "  \"residual1_activation_store_strategy\": \""
        << (stored_residual1_block_count > 0
                ? (fuse_residual1_store_enabled ? "fused-attention-residual-layernorm-bf16-store"
                                                 : "separate-float32-to-bf16-store")
                : "disabled")
        << "\",\n"
        << "  \"stored_residual1_activation_blocks\": " << stored_residual1_block_count << ",\n"
        << "  \"stored_residual1_activation_elements\": " << stored_residual1_activation_arena_elements << ",\n"
        << "  \"stored_residual1_activation_bytes\": " << stored_residual1_activation_arena_bytes << ",\n"
        << "  \"stored_residual1_activation_store_kernel_launches\": "
        << stored_residual1_activation_store_kernel_launches << ",\n"
        << "  \"stored_residual1_activation_restore_kernel_launches\": "
        << stored_residual1_activation_restore_kernel_launches << ",\n"
        << "  \"residual1_backward_consumer_strategy\": \""
        << (stored_residual1_block_count > 0
                ? (bf16_residual1_ln_backward_enabled ? "bf16-layernorm-backward"
                                                       : "restore-float32-layernorm-backward")
                : "disabled")
        << "\",\n"
        << "  \"attention_activation_storage_strategy\": \""
        << (stored_attention_block_count > 0 ? "tk-bf16-direct-forward-store-saved-backward" : "disabled")
        << "\",\n"
        << "  \"stored_attention_activation_blocks\": " << stored_attention_block_count << ",\n"
        << "  \"stored_attention_bf16_elements\": " << stored_attention_bf16_arena_elements << ",\n"
        << "  \"stored_attention_bf16_bytes\": " << stored_attention_bf16_arena_bytes << ",\n"
        << "  \"stored_attention_lse_elements\": " << stored_attention_lse_arena_elements << ",\n"
        << "  \"stored_attention_lse_bytes\": " << stored_attention_lse_arena_bytes << ",\n"
        << "  \"stored_attention_store_kernel_launches\": " << stored_attention_store_kernel_launches << ",\n"
        << "  \"stored_attention_restore_kernel_launches\": " << stored_attention_restore_kernel_launches << ",\n"
        << "  \"stored_attention_backward_kernel_launches\": " << stored_attention_backward_kernel_launches << ",\n"
        << "  \"stored_attention_backward_consumer_strategy\": \""
        << (stored_attention_block_count > 0 ? "saved-tk-bf16-qkvo-lse-backward-to-qkv" : "disabled")
        << "\",\n"
        << "  \"packed_attention_activation_storage_strategy\": \""
        << (stored_packed_attention_block_count > 0 ? "packed-qkv-o-bf16-forward-store-direct-backward" : "disabled")
        << "\",\n"
        << "  \"stored_packed_attention_activation_blocks\": " << stored_packed_attention_block_count << ",\n"
        << "  \"stored_packed_attention_bf16_elements\": " << stored_packed_attention_bf16_arena_elements << ",\n"
        << "  \"stored_packed_attention_bf16_bytes\": " << stored_packed_attention_bf16_arena_bytes << ",\n"
        << "  \"stored_packed_attention_ln1_stats_enabled\": "
        << (store_packed_attention_ln1_stats_enabled ? "true" : "false") << ",\n"
        << "  \"stored_packed_attention_ln1_stats_blocks\": " << stored_packed_attention_ln1_stats_block_count << ",\n"
        << "  \"stored_packed_attention_ln1_stats_elements\": " << stored_packed_attention_ln1_stats_arena_elements << ",\n"
        << "  \"stored_packed_attention_ln1_stats_bytes\": " << stored_packed_attention_ln1_stats_arena_bytes << ",\n"
        << "  \"stored_packed_attention_lse_elements\": " << stored_packed_attention_lse_arena_elements << ",\n"
        << "  \"stored_packed_attention_lse_bytes\": " << stored_packed_attention_lse_arena_bytes << ",\n"
        << "  \"stored_packed_attention_lse_enabled\": "
        << (store_packed_attention_lse_enabled ? "true" : "false") << ",\n"
        << "  \"stored_packed_attention_store_blocks\": " << stored_packed_attention_store_blocks << ",\n"
        << "  \"stored_packed_attention_restore_blocks\": " << stored_packed_attention_restore_blocks << ",\n"
        << "  \"stored_packed_attention_backward_kernel_launches\": "
        << stored_packed_attention_backward_kernel_launches << ",\n"
        << "  \"stored_packed_attention_backward_consumer_strategy\": \""
        << (stored_packed_attention_block_count > 0
                ? (store_packed_attention_lse_enabled
                       ? "saved-packed-qkv-o-lse-bf16-backward-to-qkv"
                       : "saved-packed-qkv-o-workspace-lse-bf16-backward-to-qkv")
                : "disabled")
        << "\",\n"
        << "  \"max_steps\": " << cfg.max_steps << ",\n"
        << "  \"startup_only\": " << (cfg.startup_only ? "true" : "false") << ",\n"
        << "  \"eval_every_steps\": " << cfg.eval_every_steps << ",\n"
        << "  \"eval_batches\": " << cfg.eval_batches << ",\n"
        << "  \"lazy_validation_mlp_float_scratch_enabled\": "
        << (lazy_validation_mlp_float_scratch_enabled ? "true" : "false") << ",\n"
        << "  \"lazy_validation_mlp_float_scratch_elements\": "
        << lazy_validation_mlp_float_scratch_elements << ",\n"
        << "  \"lazy_validation_mlp_float_scratch_bytes\": "
        << lazy_validation_mlp_float_scratch_bytes << ",\n"
        << "  \"lazy_validation_mlp_float_scratch_cuda_malloc_count\": "
        << lazy_validation_mlp_float_scratch_cuda_malloc_count << ",\n"
        << "  \"train_loss_eval_count\": " << train_loss_eval_count << ",\n"
        << "  \"train_loss_last_step\": " << train_loss_last_step << ",\n"
        << "  \"train_loss_sparse\": false,\n"
        << "  \"train_loss_sampling\": \"disabled\",\n"
        << "  \"train_loss_on_validation_steps\": false,\n"
        << "  \"token_id_direct_u16_enabled\": " << (direct_u16_token_ids_enabled ? "true" : "false") << ",\n"
        << "  \"token_id_upload_strategy\": \""
        << (direct_u16_token_ids_enabled
                ? "uint16-pinned-async-h2d-direct-kernel-consumption"
                : "uint16-pinned-async-h2d-device-widen")
        << "\",\n"
        << "  \"token_id_host_staging\": \"pinned\",\n"
        << "  \"token_id_h2d_copy\": \"cudaMemcpyAsync-contiguous-arena\",\n"
        << "  \"token_id_h2d_copy_calls_per_microbatch\": 1,\n"
        << "  \"token_id_h2d_copy_calls_elided_per_microbatch\": 1,\n"
        << "  \"token_id_widen_strategy\": \""
        << (direct_u16_token_ids_enabled ? "elided-direct-u16-kernels" : "single-contiguous-arena-kernel")
        << "\",\n"
        << "  \"token_id_widen_kernel_launches_per_microbatch\": "
        << (direct_u16_token_ids_enabled ? 0 : 1) << ",\n"
        << "  \"token_id_widen_kernel_launches_elided_per_microbatch\": "
        << (direct_u16_token_ids_enabled ? 2 : 1) << ",\n"
        << "  \"token_batch_staging_strategy\": \"direct-sampler-to-pinned-arena\",\n"
        << "  \"token_batch_vector_materialization\": false,\n"
        << "  \"token_batch_vector_copy_to_pinned_elided\": true,\n"
        << "  \"token_id_host_validation\": false,\n"
        << "  \"token_buffer_allocation_strategy\": \"combined-arenas\",\n"
        << "  \"token_device_allocation_strategy\": \"single-device-arena\",\n"
        << "  \"device_allocator_strategy\": \""
        << (async_device_allocator_enabled ? "cudaMallocAsync-null-stream" : "cudaMalloc") << "\",\n"
        << "  \"device_cuda_malloc_async_requested\": "
        << (async_device_allocator_requested ? "true" : "false") << ",\n"
        << "  \"device_cuda_malloc_async_enabled\": "
        << (async_device_allocator_enabled ? "true" : "false") << ",\n"
        << "  \"device_cuda_malloc_async_symbol_loaded\": "
        << (cuda_malloc_async != nullptr ? "true" : "false") << ",\n"
        << "  \"device_cuda_free_async_symbol_loaded\": "
        << (cuda_free_async != nullptr ? "true" : "false") << ",\n"
        << "  \"device_cuda_malloc_async_count\": " << device_cuda_malloc_async_count << ",\n"
        << "  \"device_cuda_free_async_count\": " << device_cuda_free_async_count << ",\n"
        << "  \"device_cuda_malloc_async_fallback_count\": "
        << device_cuda_malloc_async_fallback_count << ",\n"
        << "  \"token_device_arena_cuda_malloc_count\": " << token_device_arena_cuda_malloc_count << ",\n"
        << "  \"token_device_arena_requested_bytes\": " << token_device_arena_requested_bytes << ",\n"
        << "  \"token_device_arena_bytes\": " << token_device_arena_bytes << ",\n"
        << "  \"token_device_arena_suballocation_count\": " << token_device_arena_suballocation_count << ",\n"
        << "  \"token_device_cuda_mallocs_elided\": " << token_device_cuda_mallocs_elided << ",\n"
        << "  \"token_i64_arena_cuda_malloc_count\": " << token_i64_arena_cuda_malloc_count << ",\n"
        << "  \"token_u16_device_arena_cuda_malloc_count\": " << token_u16_device_arena_cuda_malloc_count << ",\n"
        << "  \"token_u16_pinned_arena_cuda_host_alloc_count\": " << token_u16_pinned_arena_cuda_host_alloc_count << ",\n"
        << "  \"token_i64_arena_elements\": " << token_i64_arena_elements << ",\n"
        << "  \"token_u16_device_arena_elements\": " << token_u16_device_arena_elements << ",\n"
        << "  \"token_u16_pinned_arena_elements\": " << token_u16_pinned_arena_elements << ",\n"
        << "  \"token_weight_init_strategy\": \""
        << (legacy_mod17_token_weight_init_enabled
                ? (token_weight_bf16_initial_refresh_elided
                       ? "device-tile-mod17-deterministic-fused-bf16-shadow"
                       : "device-tile-mod17-deterministic")
                : (token_weight_bf16_initial_refresh_elided
                       ? "device-tile-power2-deterministic-fused-bf16-shadow"
                       : "device-tile-power2-deterministic"))
        << "\",\n"
        << "  \"token_weight_init_legacy_mod17_enabled\": "
        << (legacy_mod17_token_weight_init_enabled ? "true" : "false") << ",\n"
        << "  \"token_weight_bf16_initial_refresh_elided\": "
        << (token_weight_bf16_initial_refresh_elided ? "true" : "false") << ",\n"
        << "  \"token_weight_bf16_initial_refresh_fusion_enabled\": "
        << (fuse_token_weight_bf16_initial_refresh_enabled ? "true" : "false") << ",\n"
        << "  \"token_weight_host_materialization\": false,\n"
        << "  \"float_allocation_strategy\": \"single-arena\",\n"
        << "  \"float_allocation_cuda_malloc_count\": " << float_arena_cuda_malloc_count << ",\n"
        << "  \"float_allocation_request_count\": " << float_arena_requests.size() << ",\n"
        << "  \"float_arena_requested_elements\": " << float_arena_requested_elements << ",\n"
        << "  \"float_arena_allocated_elements\": " << float_arena_allocated_elements << ",\n"
        << "  \"uint16_allocation_strategy\": \""
        << (combined_uint16_arena_enabled ? "single-arena" : "per-buffer-cudaMalloc") << "\",\n"
        << "  \"uint16_allocation_cuda_malloc_count\": "
        << (combined_uint16_arena_enabled ? uint16_arena_cuda_malloc_count
                                          : static_cast<std::int64_t>(uint16_ptrs.size())) << ",\n"
        << "  \"uint16_allocation_request_count\": " << uint16_arena_requests.size() << ",\n"
        << "  \"uint16_arena_requested_elements\": " << uint16_arena_requested_elements << ",\n"
        << "  \"uint16_arena_allocated_elements\": " << uint16_arena_allocated_elements << ",\n"
        << "  \"uint16_arena_cuda_malloc_count\": " << uint16_arena_cuda_malloc_count << ",\n"
        << "  \"uint16_arena_suballocation_count\": " << uint16_arena_requests.size() << ",\n"
        << "  \"float_arena_zero_init_strategy\": \"" << startup_zero_init_strategy << "\",\n"
        << "  \"startup_cuda_memset_zero_enabled\": " << (startup_cuda_memset_zero_enabled ? "true" : "false") << ",\n"
        << "  \"startup_cuda_memset_zero_available\": " << (cuda_memset_async != nullptr ? "true" : "false") << ",\n"
        << "  \"float_arena_zero_fill_count\": " << float_arena_zero_fill_count << ",\n"
        << "  \"adamw_state_zero_fill_count\": " << adamw_state_zero_fill_count << ",\n"
        << "  \"startup_cuda_memset_zero_fill_count\": " << startup_cuda_memset_zero_fill_count << ",\n"
        << "  \"startup_tile_zero_fill_count\": " << startup_tile_zero_fill_count << ",\n"
        << "  \"adamw_state_zero_range_count\": " << adamw_state_zero_range_count << ",\n"
        << "  \"adamw_state_zero_range_elements\": " << adamw_state_zero_range_elements << ",\n"
        << "  \"startup_per_buffer_zero_fill_elided\": true,\n"
        << "  \"startup_per_buffer_zero_fill_launches_elided\": " << startup_per_buffer_zero_fill_launches_elided << ",\n"
        << "  \"descriptor_allocation_strategy\": \"single-device-arena\",\n"
        << "  \"descriptor_arena_cuda_malloc_count\": " << descriptor_arena_cuda_malloc_count << ",\n"
        << "  \"descriptor_arena_requested_bytes\": " << descriptor_arena_requested_bytes << ",\n"
        << "  \"descriptor_arena_bytes\": " << descriptor_arena_bytes << ",\n"
        << "  \"descriptor_arena_suballocation_count\": " << descriptor_arena_suballocation_count << ",\n"
        << "  \"descriptor_upload_strategy\": \"single-host-packed-arena-copy\",\n"
        << "  \"descriptor_arena_copy_count\": " << descriptor_arena_copy_count << ",\n"
        << "  \"descriptor_arena_copy_calls_elided\": " << descriptor_arena_copy_calls_elided << ",\n"
        << "  \"descriptor_cuda_mallocs_elided\": " << descriptor_cuda_mallocs_elided << ",\n"
        << "  \"parameter_initialization_strategy\": \""
        << (bf16_parameter_fill_descriptor_count > 0
                ? "split-float32-and-bf16-fill-many-values"
                : "fused-multi-buffer-fill-values")
        << "\",\n"
        << "  \"parameter_initialization_descriptor_count\": " << parameter_fill_descriptor_count << ",\n"
        << "  \"bf16_parameter_initialization_descriptor_count\": "
        << bf16_parameter_fill_descriptor_count << ",\n"
        << "  \"parameter_initialization_max_elements\": " << parameter_fill_max_elements << ",\n"
        << "  \"bf16_parameter_initialization_max_elements\": "
        << bf16_parameter_fill_max_elements << ",\n"
        << "  \"parameter_initialization_kernel_launches\": " << parameter_fill_kernel_launches << ",\n"
        << "  \"bf16_parameter_initialization_kernel_launches\": "
        << bf16_parameter_fill_kernel_launches << ",\n"
        << "  \"parameter_initialization_kernel_launches_per_startup\": "
        << ((parameter_fill_descriptor_count > 0 ? 1 : 0) +
            (bf16_parameter_fill_descriptor_count > 0 ? 1 : 0)) << ",\n"
        << "  \"parameter_initialization_per_buffer_launches_elided\": "
        << std::max<std::int64_t>(
               0,
               nonzero_parameter_fill_buffer_count +
                   nonzero_bf16_parameter_fill_buffer_count -
                   ((parameter_fill_descriptor_count > 0 ? 1 : 0) +
                    (bf16_parameter_fill_descriptor_count > 0 ? 1 : 0)))
        << ",\n"
        << "  \"steps_completed\": " << steps_completed << ",\n"
        << "  \"epochs_completed\": " << epochs_completed << ",\n"
        << "  \"train_microbatches_completed\": " << train_microbatches_completed << ",\n"
        << "  \"tokens_processed\": " << tokens_processed << ",\n"
        << "  \"weight_update_count\": " << (kGlobalParameterBuffers + kPerBlockParameterBuffers * trained_layers) << ",\n"
        << "  \"adamw_update_strategy\": \""
        << (bf16_block_weight_param_update_enabled
                ? "split-float32-and-bf16-param-multi-buffer-device-scale"
                : "fused-multi-buffer-device-scale")
        << "\",\n"
        << "  \"adamw_bf16_shadow_refresh_strategy\": \""
        << (bf16_block_weight_param_update_enabled
                ? "elided-bf16-primary-params"
                : (fuse_adamw_bf16_shadow_refresh_enabled ? "fused-adamw-shadow-write" : "separate-many-pack-after-adamw"))
        << "\",\n"
        << "  \"adamw_descriptor_count\": " << adamw_descriptor_count << ",\n"
        << "  \"adamw_float_update_descriptor_count\": " << adamw_float_update_descriptor_count << ",\n"
        << "  \"adamw_bf16_param_descriptor_count\": " << adamw_bf16_param_descriptor_count << ",\n"
        << "  \"adamw_bf16_param_bf16_grad_descriptor_count\": "
        << adamw_bf16_param_bf16_grad_descriptor_count << ",\n"
        << "  \"adamw_max_elements\": " << adamw_max_elements << ",\n"
        << "  \"adamw_float_update_max_elements\": " << adamw_float_update_max_elements << ",\n"
        << "  \"adamw_bf16_param_max_elements\": " << adamw_bf16_param_max_elements << ",\n"
        << "  \"adamw_bf16_param_bf16_grad_max_elements\": "
        << adamw_bf16_param_bf16_grad_max_elements << ",\n"
        << "  \"adamw_kernel_launches\": " << adamw_kernel_launches << ",\n"
        << "  \"adamw_float_update_kernel_launches\": " << adamw_float_update_kernel_launches << ",\n"
        << "  \"adamw_bf16_param_kernel_launches\": " << adamw_bf16_param_kernel_launches << ",\n"
        << "  \"adamw_bf16_param_bf16_grad_kernel_launches\": "
        << adamw_bf16_param_bf16_grad_kernel_launches << ",\n"
        << "  \"adamw_step_kernel_launches_per_optimizer_step\": "
        << (bf16_block_weight_param_update_enabled
                ? ((adamw_float_update_descriptor_count > 0 ? 1 : 0) +
                   (adamw_bf16_param_descriptor_count > 0 ? 1 : 0) +
                   (adamw_bf16_param_bf16_grad_descriptor_count > 0 ? 1 : 0))
                : (adamw_descriptor_count > 0 ? 1 : 0))
        << ",\n"
        << "  \"adamw_per_buffer_step_launches_elided\": "
        << std::max<std::int64_t>(0, (kGlobalParameterBuffers + kPerBlockParameterBuffers * trained_layers) - 1) << ",\n"
        << "  \"gradient_zero_strategy\": \"fused-multi-buffer-accumulation-zero\",\n"
        << "  \"gradient_cuda_memset_zero_enabled\": "
        << (gradient_cuda_memset_zero_enabled ? "true" : "false") << ",\n"
        << "  \"gradient_cuda_memset_zero_available\": "
        << (cuda_memset_async != nullptr ? "true" : "false") << ",\n"
        << "  \"gradient_zero_range_count\": " << gradient_zero_range_count << ",\n"
        << "  \"gradient_zero_range_elements\": " << gradient_zero_range_elements << ",\n"
        << "  \"gradient_zero_cuda_memset_count\": " << gradient_zero_cuda_memset_count << ",\n"
        << "  \"gradient_zero_tile_fill_count\": " << gradient_zero_tile_fill_count << ",\n"
        << "  \"accumulation_zero_kernel_launches\": " << accumulation_zero_kernel_launches << ",\n"
        << "  \"gradient_zero_kernel_launches_per_optimizer_step\": "
        << (((gradient_cuda_memset_zero_enabled && cuda_memset_async != nullptr && gradient_zero_range_count > 0)
                 ? gradient_zero_range_count
                 : (adamw_descriptor_count > 0 ? 1 : 0)) +
            (bf16_block_dweight_staging_enabled && block_dweight_bf16_staging_elements > 0 ? 1 : 0)) << ",\n"
        << "  \"gradient_zero_per_buffer_launches_elided\": "
        << std::max<std::int64_t>(0, (kGlobalParameterBuffers + kPerBlockParameterBuffers * trained_layers) - 1) << ",\n"
        << "  \"gradient_clip_strategy\": \"fused-multi-buffer-sumsq-device-scale\",\n"
        << "  \"gradient_clip_bf16_sumsq_kernel_loaded\": "
        << (sumsq_partials_many_bf16_bits != nullptr ? "true" : "false") << ",\n"
        << "  \"gradient_sumsq_kernel_launches\": " << gradient_sumsq_kernel_launches << ",\n"
        << "  \"gradient_sumsq_kernel_launches_per_optimizer_step\": "
        << (bf16_block_weight_param_update_enabled
                ? ((adamw_float_update_descriptor_count > 0 ? 1 : 0) +
                   (adamw_bf16_param_descriptor_count > 0 ? 1 : 0) +
                   (adamw_bf16_param_bf16_grad_descriptor_count > 0 ? 1 : 0))
                : (adamw_descriptor_count > 0 ? 1 : 0))
        << ",\n"
        << "  \"gradient_sumsq_per_buffer_launches_elided\": "
        << std::max<std::int64_t>(0, (kGlobalParameterBuffers + kPerBlockParameterBuffers * trained_layers) - 1) << ",\n"
        << "  \"block_state_layout\": {\n"
        << "    \"allocated_block_count\": " << trained_layers << ",\n"
        << "    \"target_block_count\": " << target_layers << ",\n"
        << "    \"activation_tape_count\": " << kActivationTapeCount << ",\n"
        << "    \"packed_qkv_float_attention_tape_elided\": "
        << (packed_qkv_float_attention_tape_elided ? "true" : "false") << ",\n"
        << "    \"packed_qkv_float_attention_tape_elements_elided\": "
        << packed_qkv_float_attention_tape_elements_elided << ",\n"
        << "    \"forward_row_qkv_scratch_allocated\": false,\n"
        << "    \"forward_row_qkv_scratch_buffers_elided\": 3,\n"
        << "    \"persistent_block_outputs\": " << persistent_block_output_count << ",\n"
        << "    \"persistent_block_output_write_strategy\": \"direct-residual2-output\",\n"
        << "    \"persistent_block_output_copy_elided_count\": " << direct_block_output_write_count << ",\n"
        << "    \"final_block_output_copy_elided\": true,\n"
        << "    \"validation_persistent_block_outputs\": 0,\n"
        << "    \"validation_block_output_copies_elided\": true,\n"
        << "    \"backward_recompute_blocks\": " << backward_recompute_block_count << ",\n"
        << "    \"final_block_backward_recompute_elided\": true,\n"
        << "    \"backward_recompute_mlp_fc_gelu_elided\": "
        << (stored_mlp_activation_block_count > 0 ? "true" : "false") << ",\n"
        << "    \"backward_recompute_attention_qkv_sdpa_elided\": "
        << (stored_attention_block_count > 0 || stored_packed_attention_block_count > 0 ? "true" : "false")
        << ",\n"
        << "    \"backward_recompute_attention_uses_saved_o\": "
        << (stored_attention_block_count > 0 || stored_packed_attention_block_count > 0 ? "true" : "false")
        << ",\n"
        << "    \"backward_recompute_mlp_projection_elided\": true,\n"
        << "    \"backward_recompute_final_residual_elided\": true,\n"
        << "    \"mlp_proj_backward_gelu_inplace\": true,\n"
        << "    \"mlp_proj_backward_grad_act_scratch_allocated\": false,\n"
        << "    \"activation_tape_strategy\": \""
        << (stored_packed_attention_block_count > 0 && stored_mlp_activation_block_count > 0
                ? "scratch-recompute-bf16-stored-packed-attention-and-mlp-direct-backward"
                : stored_packed_attention_block_count > 0
                ? "scratch-recompute-bf16-stored-packed-attention-direct-backward"
                : stored_attention_block_count > 0 && stored_mlp_activation_block_count > 0
                ? "scratch-recompute-bf16-stored-attention-and-mlp-direct-backward"
                : stored_mlp_activation_block_count > 0
                ? "scratch-recompute-bf16-stored-mlp-direct-backward-opt-in"
                : "scratch-recompute")
        << "\",\n"
        << "    \"per_block_parameter_buffers\": " << kPerBlockParameterBuffers << ",\n"
        << "    \"per_block_gradient_buffers\": " << kPerBlockGradientBuffers << ",\n"
        << "    \"per_block_direct_accum_gradient_buffers\": " << kPerBlockDirectAccumGradientBuffers << ",\n"
        << "    \"per_block_accum_gradient_buffers\": " << (kPerBlockGradientBuffers + kPerBlockDirectAccumGradientBuffers) << ",\n"
        << "    \"per_block_adamw_state_buffers\": " << kPerBlockAdamWStateBuffers << ",\n"
        << "    \"per_block_gradient_partials\": " << per_block_gradient_partial_count << ",\n"
        << "    \"global_gradient_partials\": " << global_gradient_partial_count << ",\n"
        << "    \"global_parameter_buffers\": " << kGlobalParameterBuffers << ",\n"
        << "    \"parameter_allocation_loop\": true,\n"
        << "    \"parameter_initialization_loop\": false,\n"
        << "    \"parameter_initialization_loop_elided\": true,\n"
        << "    \"parameter_initialization_strategy\": \""
        << (bf16_parameter_fill_descriptor_count > 0
                ? "split-float32-and-bf16-fill-many-values"
                : "fused-multi-buffer-fill-values")
        << "\",\n"
        << "    \"parameter_initialization_descriptor_count\": " << parameter_fill_descriptor_count << ",\n"
        << "    \"bf16_parameter_initialization_descriptor_count\": "
        << bf16_parameter_fill_descriptor_count << ",\n"
        << "    \"parameter_initialization_kernel_launches_per_startup\": "
        << ((parameter_fill_descriptor_count > 0 ? 1 : 0) +
            (bf16_parameter_fill_descriptor_count > 0 ? 1 : 0)) << ",\n"
        << "    \"parameter_initialization_per_buffer_launches_elided\": "
        << std::max<std::int64_t>(
               0,
               nonzero_parameter_fill_buffer_count +
                   nonzero_bf16_parameter_fill_buffer_count -
                   ((parameter_fill_descriptor_count > 0 ? 1 : 0) +
                    (bf16_parameter_fill_descriptor_count > 0 ? 1 : 0)))
        << ",\n"
        << "    \"startup_zero_init_strategy\": \"" << startup_zero_init_strategy << "\",\n"
        << "    \"startup_cuda_memset_zero_enabled\": " << (startup_cuda_memset_zero_enabled ? "true" : "false") << ",\n"
        << "    \"startup_cuda_memset_zero_available\": " << (cuda_memset_async != nullptr ? "true" : "false") << ",\n"
        << "    \"startup_arena_zero_fill_count\": " << float_arena_zero_fill_count << ",\n"
        << "    \"startup_adamw_state_zero_fill_count\": " << adamw_state_zero_fill_count << ",\n"
        << "    \"startup_cuda_memset_zero_fill_count\": " << startup_cuda_memset_zero_fill_count << ",\n"
        << "    \"startup_tile_zero_fill_count\": " << startup_tile_zero_fill_count << ",\n"
        << "    \"startup_adamw_state_zero_range_count\": " << adamw_state_zero_range_count << ",\n"
        << "    \"startup_adamw_state_zero_range_elements\": " << adamw_state_zero_range_elements << ",\n"
        << "    \"startup_per_buffer_zero_fill_elided\": true,\n"
        << "    \"startup_per_buffer_zero_fill_launches_elided\": " << startup_per_buffer_zero_fill_launches_elided << ",\n"
        << "    \"descriptor_allocation_strategy\": \"single-device-arena\",\n"
        << "    \"descriptor_arena_cuda_malloc_count\": " << descriptor_arena_cuda_malloc_count << ",\n"
        << "    \"descriptor_arena_suballocation_count\": " << descriptor_arena_suballocation_count << ",\n"
        << "    \"descriptor_upload_strategy\": \"single-host-packed-arena-copy\",\n"
        << "    \"descriptor_arena_copy_count\": " << descriptor_arena_copy_count << ",\n"
        << "    \"descriptor_arena_copy_calls_elided\": " << descriptor_arena_copy_calls_elided << ",\n"
        << "    \"descriptor_cuda_mallocs_elided\": " << descriptor_cuda_mallocs_elided << ",\n"
        << "    \"block0_duplicate_allocation_elided\": true,\n"
        << "    \"block0_duplicate_activation_allocation_elided\": true,\n"
        << "    \"block0_duplicate_parameter_initialization_elided\": true,\n"
        << "    \"block0_duplicate_adamw_state_zero_elided\": true,\n"
        << "    \"gradient_zero_loop\": false,\n"
        << "    \"gradient_zero_loop_elided\": true,\n"
        << "    \"gradient_zero_strategy\": \"fused-multi-buffer-accumulation-zero\",\n"
        << "    \"gradient_cuda_memset_zero_enabled\": "
        << (gradient_cuda_memset_zero_enabled ? "true" : "false") << ",\n"
        << "    \"gradient_cuda_memset_zero_available\": "
        << (cuda_memset_async != nullptr ? "true" : "false") << ",\n"
        << "    \"gradient_zero_range_count\": " << gradient_zero_range_count << ",\n"
        << "    \"gradient_zero_range_elements\": " << gradient_zero_range_elements << ",\n"
        << "    \"gradient_zero_cuda_memset_count\": " << gradient_zero_cuda_memset_count << ",\n"
        << "    \"gradient_zero_tile_fill_count\": " << gradient_zero_tile_fill_count << ",\n"
        << "    \"gradient_zeroed_buffer_count\": 0,\n"
        << "    \"gradient_zero_descriptor_count\": " << adamw_descriptor_count << ",\n"
        << "    \"gradient_zero_kernel_launches_per_optimizer_step\": "
        << (((gradient_cuda_memset_zero_enabled && cuda_memset_async != nullptr && gradient_zero_range_count > 0)
                 ? gradient_zero_range_count
                 : (adamw_descriptor_count > 0 ? 1 : 0)) +
            (bf16_block_dweight_staging_enabled && block_dweight_bf16_staging_elements > 0 ? 1 : 0)) << ",\n"
        << "    \"gradient_zero_per_buffer_launches_elided\": "
        << std::max<std::int64_t>(0, (kGlobalParameterBuffers + kPerBlockParameterBuffers * trained_layers) - 1) << ",\n"
        << "    \"gradient_accumulation_loop\": false,\n"
        << "    \"gradient_accumulation_buffers\": true,\n"
        << "    \"gradient_accumulation_copy_loop_elided\": true,\n"
        << "    \"gradient_accumulation_zero_strategy\": \"all-accumulation-buffers\",\n"
        << "    \"token_gradient_accumulation_direct\": true,\n"
        << "    \"token_gradient_scratch_buffer_allocated\": false,\n"
        << "    \"block_linear_weight_gradient_accumulation_direct\": true,\n"
        << "    \"block_dweight_bf16_staging_enabled\": "
        << (bf16_block_dweight_staging_enabled ? "true" : "false") << ",\n"
        << "    \"block_dweight_bf16_staging_elements\": " << block_dweight_bf16_staging_elements << ",\n"
        << "    \"block_dweight_bf16_staging_zero_count\": " << block_dweight_bf16_staging_zero_count << ",\n"
        << "    \"block_dweight_bf16_staging_convert_kernel_launches\": "
        << block_dweight_bf16_staging_convert_kernel_launches << ",\n"
        << "    \"block_linear_weight_gradient_scratch_buffers_allocated\": false,\n"
        << "    \"block_linear_weight_gradient_microbatch_full_copy_elided\": true,\n"
        << "    \"layer_norm_affine_gradient_accumulation_direct\": true,\n"
        << "    \"layer_norm_affine_gradient_scratch_buffers_allocated\": false,\n"
        << "    \"layer_norm_affine_gradient_microbatch_full_copy_elided\": true,\n"
        << "    \"linear_bias_gradient_accumulation_direct\": true,\n"
        << "    \"linear_bias_gradient_scratch_buffers_allocated\": false,\n"
        << "    \"linear_bias_gradient_microbatch_full_copy_elided\": true,\n"
        << "    \"position_gradient_accumulation_direct\": true,\n"
        << "    \"position_gradient_scratch_buffer_allocated\": false,\n"
        << "    \"position_gradient_microbatch_full_copy_elided\": true,\n"
        << "    \"layer_norm_backward_affine_strategy\": \"auto-chunked-atomic-accumulate\",\n"
        << "    \"layer_norm_stats_strategy\": \""
        << (layer_norm_stats_enabled ? "forward-store-mean-rstd-backward-reuse" : "backward-recompute") << "\",\n"
        << "    \"layer_norm_backward_reuses_forward_stats\": "
        << (layer_norm_stats_enabled ? "true" : "false") << ",\n"
        << "    \"layer_norm_stats_disabled_by_fused_residual_ln2\": "
        << (fuse_attention_residual_ln2_enabled && !layer_norm_stats_enabled ? "true" : "false") << ",\n"
        << "    \"layer_norm_backward_residual_fusion_enabled\": "
        << (fuse_ln_backward_residual_enabled ? "true" : "false") << ",\n"
        << "    \"layer_norm_backward_affine_residual_fusion_enabled\": "
        << (fuse_ln_backward_affine_residual_enabled ? "true" : "false") << ",\n"
        << "    \"layer_norm_backward_affine_residual_fused_kernel_launches\": "
        << layer_norm_backward_affine_residual_fused_kernel_launches << ",\n"
        << "    \"layer_norm_backward_residual_scratch_buffers_allocated\": "
        << (fuse_ln_backward_residual_enabled ? "false" : "true") << ",\n"
        << "    \"layer_norm_backward_residual_scratch_buffers_elided\": "
        << (fuse_ln_backward_residual_enabled ? 2 : 0) << ",\n"
        << "    \"layer_norm_backward_residual_scratch_elements_elided\": "
        << (fuse_ln_backward_residual_enabled ? activation_elements * 2 : 0) << ",\n"
        << "    \"layer_norm_backward_residual_strategy\": \""
        << (fuse_ln_backward_affine_residual_enabled
                ? "fused-affine-dinput-residual-add-with-forward-stats"
                : fuse_ln_backward_residual_enabled
                ? "fused-dinput-residual-add-with-forward-stats"
                : "separate-dinput-plus-residual-add")
        << "\",\n"
        << "    \"residual1_backward_consumer_strategy\": \""
        << (stored_residual1_block_count > 0
                ? (bf16_residual1_ln_backward_enabled ? "bf16-layernorm-backward"
                                                       : "restore-float32-layernorm-backward")
                : "disabled")
        << "\",\n"
        << "    \"gradient_clip_loop\": false,\n"
        << "    \"gradient_clip_loop_elided\": true,\n"
        << "    \"gradient_clip_strategy\": \"fused-multi-buffer-sumsq-device-scale\",\n"
        << "    \"gradient_clip_descriptor_count\": " << adamw_descriptor_count << ",\n"
        << "    \"gradient_clip_bf16_sumsq_kernel_loaded\": "
        << (sumsq_partials_many_bf16_bits != nullptr ? "true" : "false") << ",\n"
        << "    \"gradient_sumsq_kernel_launches_per_optimizer_step\": "
        << (bf16_block_weight_param_update_enabled
                ? ((adamw_float_update_descriptor_count > 0 ? 1 : 0) +
                   (adamw_bf16_param_descriptor_count > 0 ? 1 : 0) +
                   (adamw_bf16_param_bf16_grad_descriptor_count > 0 ? 1 : 0))
                : (adamw_descriptor_count > 0 ? 1 : 0))
        << ",\n"
        << "    \"gradient_sumsq_per_buffer_launches_elided\": "
        << std::max<std::int64_t>(0, (kGlobalParameterBuffers + kPerBlockParameterBuffers * trained_layers) - 1) << ",\n"
        << "    \"adamw_device_clip_scale_fused\": true,\n"
        << "    \"adamw_update_loop\": false,\n"
        << "    \"adamw_update_loop_elided\": true,\n"
        << "    \"adamw_update_strategy\": \""
        << (bf16_block_weight_param_update_enabled
                ? "split-float32-and-bf16-param-multi-buffer-device-scale"
                : "fused-multi-buffer-device-scale")
        << "\",\n"
        << "    \"adamw_bf16_shadow_refresh_strategy\": \""
        << (bf16_block_weight_param_update_enabled
                ? "elided-bf16-primary-params"
                : (fuse_adamw_bf16_shadow_refresh_enabled ? "fused-adamw-shadow-write" : "separate-many-pack-after-adamw"))
        << "\",\n"
        << "    \"block_weight_bf16_initialization_strategy\": \""
        << (direct_bf16_block_weight_init_enabled
                ? "direct-bf16-fill-many-values"
                : "float32-fill-then-bf16-pack")
        << "\",\n"
        << "    \"token_weight_bf16_shadow_enabled\": "
        << (token_weight_bf16_shadow_enabled ? "true" : "false") << ",\n"
        << "    \"token_weight_bf16_refresh_count\": " << token_weight_bf16_refresh_count << ",\n"
        << "    \"token_weight_bf16_initial_refresh_fusion_enabled\": "
        << (fuse_token_weight_bf16_initial_refresh_enabled ? "true" : "false") << ",\n"
        << "    \"token_weight_bf16_initial_refresh_elided\": "
        << (token_weight_bf16_initial_refresh_elided ? "true" : "false") << ",\n"
        << "    \"block_weight_bf16_primary_param_update_enabled\": "
        << (bf16_block_weight_param_update_enabled ? "true" : "false") << ",\n"
        << "    \"direct_bf16_block_weight_initialization_enabled\": "
        << (direct_bf16_block_weight_init_enabled ? "true" : "false") << ",\n"
        << "    \"block_weight_bf16_gradient_storage_strategy\": "
        << (adamw_bf16_param_bf16_grad_descriptor_count > 0
                ? "\"qkv-fc-bf16-accumulation-buffer\""
                : "\"float32-accumulation-buffer\"")
        << ",\n"
        << "    \"adamw_bf16_param_bf16_grad_kernel_loaded\": "
        << (adamw_many_with_device_scale_bf16_param_bf16_grad != nullptr ? "true" : "false") << ",\n"
        << "    \"adamw_descriptor_count\": " << adamw_descriptor_count << ",\n"
        << "    \"adamw_float_update_descriptor_count\": " << adamw_float_update_descriptor_count << ",\n"
        << "    \"adamw_bf16_param_descriptor_count\": " << adamw_bf16_param_descriptor_count << ",\n"
        << "    \"adamw_bf16_param_bf16_grad_descriptor_count\": "
        << adamw_bf16_param_bf16_grad_descriptor_count << ",\n"
        << "    \"adamw_step_kernel_launches_per_optimizer_step\": "
        << (bf16_block_weight_param_update_enabled
                ? ((adamw_float_update_descriptor_count > 0 ? 1 : 0) +
                   (adamw_bf16_param_descriptor_count > 0 ? 1 : 0) +
                   (adamw_bf16_param_bf16_grad_descriptor_count > 0 ? 1 : 0))
                : (adamw_descriptor_count > 0 ? 1 : 0))
        << ",\n"
        << "    \"adamw_per_buffer_step_launches_elided\": "
        << std::max<std::int64_t>(0, (kGlobalParameterBuffers + kPerBlockParameterBuffers * trained_layers) - 1) << ",\n"
        << "    \"checkpoint_export_loop\": " << (cfg.write_checkpoint ? "true" : "false") << ",\n"
        << "    \"activation_tape_loop\": true,\n"
        << "    \"forward_block_loop\": true,\n"
        << "    \"backward_block_loop\": true,\n"
        << "    \"residual_backward_fused\": true\n"
        << "  },\n"
        << "  \"gradient_partial_count\": " << gradient_partial_count << ",\n"
        << "  \"gradient_clip_norm\": " << kGradClipNorm << ",\n"
        << "  \"sample_gradient_clip_scale\": " << sampled_clip_scale << ",\n"
        << "  \"final_loss_sum\": " << final_loss_sum << ",\n"
        << "  \"final_loss_mean\": " << final_loss_mean << ",\n"
        << "  \"checkpoint\": {\n"
        << "    \"enabled\": " << (cfg.write_checkpoint ? "true" : "false") << ",\n"
        << "    \"checkpoint_written\": " << (checkpoint_written ? "true" : "false") << ",\n"
        << "    \"checkpoint_path\": \"" << json_escape(checkpoint_path_json) << "\",\n"
        << "    \"done_marker\": \"" << json_escape(done_marker_json) << "\",\n"
        << "    \"checkpoint_step\": " << checkpoint_step << ",\n"
        << "    \"version\": 5,\n"
        << "    \"precision\": \"bf16\",\n"
        << "    \"num_layers\": " << trained_layers << ",\n"
        << "    \"num_heads\": " << kHeads << ",\n"
        << "    \"channels\": " << kDim << ",\n"
        << "    \"padded_vocab\": " << kPaddedVocab << ",\n"
        << "    \"payload_pack_strategy\": \"device-many-float32-to-bf16-bits-contiguous\",\n"
        << "    \"payload_pack_kernel\": \"nfn_native_tile_float32_to_bf16_bits_many\",\n"
        << "    \"payload_copy_strategy\": \"single-contiguous-device-payload-d2h\",\n"
        << "    \"payload_cpu_bf16_conversion\": false,\n"
        << "    \"tensor_count\": " << checkpoint_tensor_count << ",\n"
        << "    \"payload_elements\": " << checkpoint_payload_elements << ",\n"
        << "    \"bf16_param_sync_kernel_launches\": " << checkpoint_bf16_param_sync_kernel_launches << ",\n"
        << "    \"device_pack_kernel_launches\": " << checkpoint_device_pack_kernel_launches << ",\n"
        << "    \"d2h_copy_count\": " << checkpoint_d2h_copy_count << ",\n"
        << "    \"d2h_bytes\": " << checkpoint_d2h_bytes << ",\n"
        << "    \"float32_d2h_bytes_elided\": " << checkpoint_float32_d2h_bytes_elided << ",\n"
        << "    \"expected_file_size\": " << checkpoint_expected_file_size << ",\n"
        << "    \"actual_file_size\": " << checkpoint_actual_file_size << "\n"
        << "  },\n"
        << "  \"validation\": {\n"
        << "    \"eval_every_steps\": " << cfg.eval_every_steps << ",\n"
        << "    \"eval_batches\": " << cfg.eval_batches << ",\n"
        << "    \"eval_batch_size\": " << eval_batch_size << ",\n"
        << "    \"eval_count\": " << validation_losses.size() << ",\n"
        << "    \"losses\": [\n";
    for (std::size_t i = 0; i < validation_losses.size(); ++i) {
        const ValidationLossRecord& record = validation_losses[i];
        std::cout
            << "      {\"step\": " << record.step
            << ", \"batches\": " << record.batches
            << ", \"tokens\": " << record.tokens
            << ", \"loss_sum\": " << record.loss_sum
            << ", \"loss_mean\": " << record.loss_mean << "}";
        if (i + 1 != validation_losses.size()) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout
        << "    ]\n"
        << "  },\n"
        << "  \"sample_updated_token_weight\": " << sampled_token_weight << ",\n"
        << "  \"max_weight_delta\": " << max_weight_delta << ",\n"
        << "  \"kernels\": [\n";
    for (std::size_t i = 0; i < required_symbols.size(); ++i) {
        std::cout << "    \"" << required_symbols[i] << "\"";
        if (i + 1 != required_symbols.size()) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout
        << "  ],\n"
        << "  \"missing_symbols\": [";
    for (std::size_t i = 0; i < missing_symbols.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << "\"" << missing_symbols[i] << "\"";
    }
    std::cout
        << "],\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";
    return passed ? 0 : 2;
}

int print_norm_residual_step_smoke_json(const Config& cfg, const char* program) {
    constexpr std::int64_t kRows = 2;
    constexpr std::int64_t kDim = 768;
    constexpr std::int64_t kActivationElements = kRows * kDim;
    constexpr float kResidualValue = 0.02f;
    constexpr float kResidualScale = 1.0f;
    constexpr float kLnWeight = 1.0f;
    constexpr float kLnBias = 0.0f;
    constexpr float kGradOut = 0.25f;
    constexpr float kLearningRate = 0.1f;
    constexpr float kBeta1 = 0.9f;
    constexpr float kBeta2 = 0.95f;
    constexpr float kEps = 1e-8f;
    constexpr float kNormEps = 1e-5f;
    constexpr float kWeightDecay = 0.1f;
    constexpr float kBiasCorrection1 = 1.0f - kBeta1;
    constexpr float kSqrtBiasCorrection2 = 0.22360679775f;
    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;

    std::vector<float> host_x(static_cast<std::size_t>(kActivationElements), 0.0f);
    for (std::int64_t i = 0; i < kActivationElements; ++i) {
        host_x[static_cast<std::size_t>(i)] = static_cast<float>((i % 17) - 8) * 0.01f;
    }

    const std::string tile_lib_path = cfg.tile_ops_lib.empty() ? default_tile_ops_lib(program) : cfg.tile_ops_lib;
    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(cfg);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool passed = false;
    double max_abs_sample = 0.0;
    double max_weight_delta = 0.0;
    std::string error;

    using FillFn = int (*)(float*, std::int64_t, float, void*);
    using GradientAccumulateFn = int (*)(float*, const float*, std::int64_t, float, void*);
    using ResidualAddFn = int (*)(const float*, const float*, const float*, float*, std::int64_t, void*);
    using LayerNormFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, float, void*);
    using LayerNormBackwardInputFn = int (*)(
        const float*, const float*, const float*, float*, std::int64_t, std::int64_t, float, void*);
    using LayerNormBackwardAffineFn = int (*)(
        const float*, const float*, float*, float*, std::int64_t, std::int64_t, float, void*);
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
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);

    void* tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    void* cuda_handle = nullptr;
    FillFn fill = nullptr;
    GradientAccumulateFn gradient_accumulate = nullptr;
    ResidualAddFn residual_add = nullptr;
    LayerNormFn layer_norm = nullptr;
    LayerNormBackwardInputFn layer_norm_backward_input = nullptr;
    LayerNormBackwardAffineFn layer_norm_backward_affine = nullptr;
    AdamWFn adamw = nullptr;
    CudaMallocFn cuda_malloc = nullptr;
    CudaFreeFn cuda_free = nullptr;
    CudaMemcpyFn cuda_memcpy = nullptr;
    CudaDeviceSynchronizeFn cuda_device_synchronize = nullptr;
    CudaGetErrorStringFn cuda_get_error_string = nullptr;

    auto cuda_error = [&](int code, const std::string& context) {
        std::ostringstream out;
        out << context << " failed with CUDA error " << code;
        if (cuda_get_error_string != nullptr) {
            const char* message = cuda_get_error_string(code);
            if (message != nullptr) {
                out << ": " << message;
            }
        }
        return out.str();
    };

    if (tile_handle == nullptr) {
        error = dl_last_error("dlopen tile ops failed");
    } else {
        tile_loaded = true;
        fill = load_symbol<FillFn>(tile_handle, "nfn_native_tile_fill_float32");
        gradient_accumulate = load_symbol<GradientAccumulateFn>(
            tile_handle, "nfn_native_tile_gradient_accumulate_float32");
        residual_add = load_symbol<ResidualAddFn>(tile_handle, "nfn_native_tile_scaled_residual_add_float32");
        layer_norm = load_symbol<LayerNormFn>(tile_handle, "nfn_native_tile_layer_norm_float32");
        layer_norm_backward_input = load_symbol<LayerNormBackwardInputFn>(
            tile_handle, "nfn_native_tile_layer_norm_backward_input_float32");
        layer_norm_backward_affine = load_symbol<LayerNormBackwardAffineFn>(
            tile_handle, "nfn_native_tile_layer_norm_backward_affine_float32");
        adamw = load_symbol<AdamWFn>(tile_handle, "nfn_native_tile_adamw_step_float32");
        if (fill == nullptr || gradient_accumulate == nullptr || residual_add == nullptr ||
            layer_norm == nullptr || layer_norm_backward_input == nullptr ||
            layer_norm_backward_affine == nullptr || adamw == nullptr) {
            error = dl_last_error("dlsym GPT-2 norm/residual-step kernels failed");
        }
    }

    if (error.empty()) {
        std::string runtime_error;
        for (const std::string& candidate : runtime_candidates) {
            cuda_lib_path = candidate;
            cuda_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cuda_handle != nullptr) {
                cuda_runtime_loaded = true;
                break;
            }
            runtime_error = dl_last_error("dlopen CUDA runtime failed");
        }
        if (cuda_handle == nullptr) {
            error = runtime_error.empty() ? "could not load CUDA runtime" : runtime_error;
        }
    }

    if (error.empty()) {
        cuda_malloc = load_symbol<CudaMallocFn>(cuda_handle, "cudaMalloc");
        cuda_free = load_symbol<CudaFreeFn>(cuda_handle, "cudaFree");
        cuda_memcpy = load_symbol<CudaMemcpyFn>(cuda_handle, "cudaMemcpy");
        cuda_device_synchronize = load_symbol<CudaDeviceSynchronizeFn>(cuda_handle, "cudaDeviceSynchronize");
        cuda_get_error_string = load_symbol<CudaGetErrorStringFn>(cuda_handle, "cudaGetErrorString");
        if (cuda_malloc == nullptr || cuda_free == nullptr || cuda_memcpy == nullptr || cuda_device_synchronize == nullptr) {
            error = "CUDA runtime is missing cudaMalloc/cudaFree/cudaMemcpy/cudaDeviceSynchronize";
        }
    }

    float* x = nullptr;
    float* residual_rhs = nullptr;
    float* residual_scale = nullptr;
    float* ln_weight = nullptr;
    float* ln_bias = nullptr;
    float* ln_out = nullptr;
    float* residual_out = nullptr;
    float* grad_out = nullptr;
    float* grad_ln = nullptr;
    float* grad_x = nullptr;
    float* grad_accum = nullptr;
    float* grad_ln_weight = nullptr;
    float* grad_ln_bias = nullptr;
    float* ln_weight_exp_avg = nullptr;
    float* ln_weight_exp_avg_sq = nullptr;
    float* ln_bias_exp_avg = nullptr;
    float* ln_bias_exp_avg_sq = nullptr;

    auto allocate = [&](float** ptr, std::int64_t elements, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        const int status = cuda_malloc(
            reinterpret_cast<void**>(ptr),
            sizeof(float) * static_cast<std::size_t>(elements));
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc " + name);
        }
    };
    allocate(&x, kActivationElements, "x");
    allocate(&residual_rhs, kActivationElements, "residual_rhs");
    allocate(&residual_scale, 1, "residual_scale");
    allocate(&ln_weight, kDim, "ln_weight");
    allocate(&ln_bias, kDim, "ln_bias");
    allocate(&ln_out, kActivationElements, "ln_out");
    allocate(&residual_out, kActivationElements, "residual_out");
    allocate(&grad_out, kActivationElements, "grad_out");
    allocate(&grad_ln, kActivationElements, "grad_ln");
    allocate(&grad_x, kActivationElements, "grad_x");
    allocate(&grad_accum, kActivationElements, "grad_accum");
    allocate(&grad_ln_weight, kDim, "grad_ln_weight");
    allocate(&grad_ln_bias, kDim, "grad_ln_bias");
    allocate(&ln_weight_exp_avg, kDim, "ln_weight_exp_avg");
    allocate(&ln_weight_exp_avg_sq, kDim, "ln_weight_exp_avg_sq");
    allocate(&ln_bias_exp_avg, kDim, "ln_bias_exp_avg");
    allocate(&ln_bias_exp_avg_sq, kDim, "ln_bias_exp_avg_sq");

    auto fill_buffer = [&](float* ptr, std::int64_t elements, float value, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        const int status = fill(ptr, elements, value, nullptr);
        if (status != 0) {
            error = cuda_error(status, "fill " + name);
        }
    };
    if (error.empty()) {
        const int status = cuda_memcpy(
            x,
            host_x.data(),
            sizeof(float) * static_cast<std::size_t>(kActivationElements),
            kCudaMemcpyHostToDevice);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy host-to-device x");
        }
    }
    fill_buffer(residual_rhs, kActivationElements, kResidualValue, "residual_rhs");
    fill_buffer(residual_scale, 1, kResidualScale, "residual_scale");
    fill_buffer(ln_weight, kDim, kLnWeight, "ln_weight");
    fill_buffer(ln_bias, kDim, kLnBias, "ln_bias");
    fill_buffer(grad_out, kActivationElements, kGradOut, "grad_out");
    fill_buffer(grad_accum, kActivationElements, 0.0f, "grad_accum");
    fill_buffer(grad_ln_weight, kDim, 0.0f, "grad_ln_weight");
    fill_buffer(grad_ln_bias, kDim, 0.0f, "grad_ln_bias");
    fill_buffer(ln_weight_exp_avg, kDim, 0.0f, "ln_weight_exp_avg");
    fill_buffer(ln_weight_exp_avg_sq, kDim, 0.0f, "ln_weight_exp_avg_sq");
    fill_buffer(ln_bias_exp_avg, kDim, 0.0f, "ln_bias_exp_avg");
    fill_buffer(ln_bias_exp_avg_sq, kDim, 0.0f, "ln_bias_exp_avg_sq");

    auto run = [&](int status, const std::string& name) {
        if (status != 0 && error.empty()) {
            error = cuda_error(status, name);
        }
    };
    if (error.empty()) {
        run(layer_norm(x, ln_weight, ln_bias, ln_out, kRows, kDim, kNormEps, nullptr), "layer_norm.forward");
    }
    if (error.empty()) {
        run(residual_add(x, residual_rhs, residual_scale, residual_out, kActivationElements, nullptr),
            "residual_add.forward");
    }
    if (error.empty()) {
        run(layer_norm_backward_affine(x, grad_out, grad_ln_weight, grad_ln_bias, kRows, kDim, kNormEps, nullptr),
            "layer_norm.backward_affine");
    }
    if (error.empty()) {
        run(layer_norm_backward_input(x, grad_out, ln_weight, grad_ln, kRows, kDim, kNormEps, nullptr),
            "layer_norm.backward_input");
    }
    if (error.empty()) {
        run(gradient_accumulate(grad_accum, grad_ln, kActivationElements, 1.0f, nullptr),
            "gradient_accumulate.layer_norm");
    }
    if (error.empty()) {
        run(gradient_accumulate(grad_accum, grad_out, kActivationElements, 1.0f, nullptr),
            "gradient_accumulate.residual");
    }
    if (error.empty()) {
        run(adamw(
                ln_weight,
                grad_ln_weight,
                ln_weight_exp_avg,
                ln_weight_exp_avg_sq,
                kDim,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                kWeightDecay,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr),
            "layer_norm.weight.adamw");
    }
    if (error.empty()) {
        run(adamw(
                ln_bias,
                grad_ln_bias,
                ln_bias_exp_avg,
                ln_bias_exp_avg_sq,
                kDim,
                kLearningRate,
                kBeta1,
                kBeta2,
                kEps,
                0.0f,
                kBiasCorrection1,
                kSqrtBiasCorrection2,
                nullptr),
            "layer_norm.bias.adamw");
    }
    if (error.empty()) {
        run(cuda_device_synchronize(), "cudaDeviceSynchronize");
    }

    auto copy_float_sample = [&](const float* device_ptr, std::int64_t offset, const std::string& name) -> float {
        float value = 0.0f;
        if (!error.empty()) {
            return value;
        }
        const int status = cuda_memcpy(&value, device_ptr + offset, sizeof(float), kCudaMemcpyDeviceToHost);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy sample " + name);
        }
        return value;
    };
    float sampled_ln_out = 0.0f;
    float sampled_residual_out = 0.0f;
    float sampled_grad_ln = 0.0f;
    float sampled_grad_accum = 0.0f;
    float sampled_grad_ln_weight = 0.0f;
    float sampled_grad_ln_bias = 0.0f;
    float sampled_ln_weight = 0.0f;
    float sampled_ln_bias = 0.0f;
    if (error.empty()) {
        sampled_ln_out = copy_float_sample(ln_out, 0, "ln_out");
        sampled_residual_out = copy_float_sample(residual_out, 0, "residual_out");
        sampled_grad_ln = copy_float_sample(grad_ln, 0, "grad_ln");
        sampled_grad_accum = copy_float_sample(grad_accum, 0, "grad_accum");
        sampled_grad_ln_weight = copy_float_sample(grad_ln_weight, 0, "grad_ln_weight");
        sampled_grad_ln_bias = copy_float_sample(grad_ln_bias, 0, "grad_ln_bias");
        sampled_ln_weight = copy_float_sample(ln_weight, 0, "ln_weight");
        sampled_ln_bias = copy_float_sample(ln_bias, 0, "ln_bias");
    }

    auto record_abs = [&](float value) {
        if (!std::isfinite(value)) {
            return false;
        }
        const double abs_value = std::fabs(static_cast<double>(value));
        if (abs_value > max_abs_sample) {
            max_abs_sample = abs_value;
        }
        return true;
    };
    if (error.empty()) {
        const bool finite_samples =
            record_abs(sampled_ln_out) && record_abs(sampled_residual_out) &&
            record_abs(sampled_grad_ln) && record_abs(sampled_grad_accum) &&
            record_abs(sampled_grad_ln_weight) && record_abs(sampled_grad_ln_bias) &&
            record_abs(sampled_ln_weight) && record_abs(sampled_ln_bias);
        max_weight_delta = std::fabs(static_cast<double>(sampled_ln_weight) - kLnWeight);
        const double bias_delta = std::fabs(static_cast<double>(sampled_ln_bias) - kLnBias);
        if (bias_delta > max_weight_delta) {
            max_weight_delta = bias_delta;
        }
        passed = finite_samples && max_abs_sample > 0.0 && max_weight_delta > 0.0;
        if (!passed) {
            std::ostringstream out_message;
            out_message << "GPT-2 norm/residual smoke did not produce finite nonzero samples and updates: sample="
                        << max_abs_sample << " weight_delta=" << max_weight_delta;
            error = out_message.str();
        }
    }

    auto free_device = [&](void* ptr, const std::string& name) {
        if (ptr != nullptr && cuda_free != nullptr) {
            const int status = cuda_free(ptr);
            if (status != 0 && error.empty()) {
                error = cuda_error(status, "cudaFree " + name);
            }
        }
    };
    free_device(x, "x");
    free_device(residual_rhs, "residual_rhs");
    free_device(residual_scale, "residual_scale");
    free_device(ln_weight, "ln_weight");
    free_device(ln_bias, "ln_bias");
    free_device(ln_out, "ln_out");
    free_device(residual_out, "residual_out");
    free_device(grad_out, "grad_out");
    free_device(grad_ln, "grad_ln");
    free_device(grad_x, "grad_x");
    free_device(grad_accum, "grad_accum");
    free_device(grad_ln_weight, "grad_ln_weight");
    free_device(grad_ln_bias, "grad_ln_bias");
    free_device(ln_weight_exp_avg, "ln_weight_exp_avg");
    free_device(ln_weight_exp_avg_sq, "ln_weight_exp_avg_sq");
    free_device(ln_bias_exp_avg, "ln_bias_exp_avg");
    free_device(ln_bias_exp_avg_sq, "ln_bias_exp_avg_sq");
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"" << json_escape(cfg.model_family) << "\",\n"
        << "  \"backend\": \"tile-cuda\",\n"
        << "  \"smoke\": \"norm_residual_step\",\n"
        << "  \"token_shards_resolved\": false,\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"rows\": " << kRows << ",\n"
        << "  \"model_dim\": " << kDim << ",\n"
        << "  \"kernels\": [\n"
        << "    \"nfn_native_tile_layer_norm_float32\",\n"
        << "    \"nfn_native_tile_scaled_residual_add_float32\",\n"
        << "    \"nfn_native_tile_layer_norm_backward_affine_float32\",\n"
        << "    \"nfn_native_tile_layer_norm_backward_input_float32\",\n"
        << "    \"nfn_native_tile_gradient_accumulate_float32\",\n"
        << "    \"nfn_native_tile_adamw_step_float32\"\n"
        << "  ],\n"
        << "  \"sample_ln_out\": " << sampled_ln_out << ",\n"
        << "  \"sample_residual_out\": " << sampled_residual_out << ",\n"
        << "  \"sample_grad_ln\": " << sampled_grad_ln << ",\n"
        << "  \"sample_grad_accum\": " << sampled_grad_accum << ",\n"
        << "  \"sample_grad_ln_weight\": " << sampled_grad_ln_weight << ",\n"
        << "  \"sample_grad_ln_bias\": " << sampled_grad_ln_bias << ",\n"
        << "  \"sample_updated_ln_weight\": " << sampled_ln_weight << ",\n"
        << "  \"sample_updated_ln_bias\": " << sampled_ln_bias << ",\n"
        << "  \"max_abs_sample\": " << max_abs_sample << ",\n"
        << "  \"max_weight_delta\": " << max_weight_delta << ",\n"
        << "  \"passed\": " << (passed ? "true" : "false");
    if (!error.empty()) {
        std::cout << ",\n"
                  << "  \"error\": \"" << json_escape(error) << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";

    return passed ? 0 : 2;
}

std::vector<std::string> build_command(const Config& cfg, const fs::path& train_shard, const fs::path& val_shard) {
    std::vector<std::string> args = {
        cfg.target,
        "-i", train_shard.string(),
        "-j", val_shard.string(),
        "-o", cfg.output_dir,
        "-v", std::to_string(cfg.eval_every_steps),
        "-s", std::to_string(cfg.sample_every_steps),
        "-g", std::to_string(cfg.generate_tokens),
        "-h", "0",
        "-b", std::to_string(cfg.batch_size),
        "-t", std::to_string(cfg.seq_len),
        "-d", std::to_string(cfg.train_batch_tokens),
        "-r", "0",
        "-z", "1",
        "-c", number_string(cfg.weight_decay),
        "-l", number_string(cfg.learning_rate),
        "-q", number_string(cfg.final_lr_fraction),
        "-u", std::to_string(cfg.warmup_steps),
        "-n", std::to_string(cfg.checkpoint_every_steps),
        "-y", "0",
        "-e", "d" + std::to_string(cfg.num_layers),
        "-af", cfg.activation,
        "-x", std::to_string(cfg.max_steps),
    };
    if (cfg.activation == "moa") {
        args.push_back("-ak");
        args.push_back(std::to_string(cfg.moa_interval));
    }
    return args;
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
    Config cfg;
    cfg.target = default_target();
    cfg.output_dir = default_output_dir();

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto value_after_equals = [&](const std::string& prefix) -> std::string {
            return arg.substr(prefix.size());
        };
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--model-family" || arg == "--base-model" || arg == "--model") {
            cfg.model_family = normalize_model_family(require_value(argc, argv, &i, arg));
        } else if (arg.rfind("--model-family=", 0) == 0) {
            cfg.model_family = normalize_model_family(value_after_equals("--model-family="));
        } else if (arg.rfind("--base-model=", 0) == 0) {
            cfg.model_family = normalize_model_family(value_after_equals("--base-model="));
        } else if (arg.rfind("--model=", 0) == 0) {
            cfg.model_family = normalize_model_family(value_after_equals("--model="));
        } else if (arg == "--tinystories") {
            cfg.dataset_alias = "roneneldan__TinyStories__TinyStoriesV2-GPT4";
        } else if (arg == "--dataset-alias" || arg == "--dataset-path") {
            cfg.dataset_alias = require_value(argc, argv, &i, arg);
        } else if (arg.rfind("--dataset-alias=", 0) == 0) {
            cfg.dataset_alias = value_after_equals("--dataset-alias=");
        } else if (arg.rfind("--dataset-path=", 0) == 0) {
            cfg.dataset_alias = value_after_equals("--dataset-path=");
        } else if (arg == "--target" || arg == "--native-cuda-executable") {
            cfg.target = require_value(argc, argv, &i, arg);
        } else if (arg.rfind("--target=", 0) == 0) {
            cfg.target = value_after_equals("--target=");
        } else if (arg == "--backend") {
            cfg.backend = normalize_backend(require_value(argc, argv, &i, arg));
        } else if (arg.rfind("--backend=", 0) == 0) {
            cfg.backend = normalize_backend(value_after_equals("--backend="));
        } else if (arg == "--tile-ops-lib") {
            cfg.tile_ops_lib = require_value(argc, argv, &i, arg);
        } else if (arg.rfind("--tile-ops-lib=", 0) == 0) {
            cfg.tile_ops_lib = value_after_equals("--tile-ops-lib=");
        } else if (arg == "--template-name" || arg == "--template" || arg == "--preset") {
            cfg.template_name = normalize_template_name(require_value(argc, argv, &i, arg));
            cfg.template_explicit = true;
        } else if (arg.rfind("--template-name=", 0) == 0) {
            cfg.template_name = normalize_template_name(value_after_equals("--template-name="));
            cfg.template_explicit = true;
        } else if (arg.rfind("--template=", 0) == 0) {
            cfg.template_name = normalize_template_name(value_after_equals("--template="));
            cfg.template_explicit = true;
        } else if (arg.rfind("--preset=", 0) == 0) {
            cfg.template_name = normalize_template_name(value_after_equals("--preset="));
            cfg.template_explicit = true;
        } else if (arg == "--graph-file" || arg == "--graph") {
            cfg.graph_file = require_value(argc, argv, &i, arg);
        } else if (arg.rfind("--graph-file=", 0) == 0) {
            cfg.graph_file = value_after_equals("--graph-file=");
        } else if (arg.rfind("--graph=", 0) == 0) {
            cfg.graph_file = value_after_equals("--graph=");
        } else if (arg == "--check-tile-ops") {
            cfg.backend = "tile-cuda";
            cfg.check_tile_ops = true;
        } else if (arg == "--smoke-tile-ops") {
            cfg.backend = "tile-cuda";
            cfg.smoke_tile_ops = true;
        } else if (arg == "--smoke-optimizer-step") {
            cfg.backend = "tile-cuda";
            cfg.smoke_optimizer_step = true;
        } else if (arg == "--smoke-lm-step") {
            cfg.backend = "tile-cuda";
            cfg.smoke_lm_step = true;
        } else if (arg == "--smoke-attention-step") {
            cfg.backend = "tile-cuda";
            cfg.smoke_attention_step = true;
        } else if (arg == "--smoke-mlp-step") {
            cfg.backend = "tile-cuda";
            cfg.smoke_mlp_step = true;
        } else if (arg == "--smoke-norm-residual-step") {
            cfg.backend = "tile-cuda";
            cfg.smoke_norm_residual_step = true;
        } else if (arg == "--smoke-transformer-block-step") {
            cfg.backend = "tile-cuda";
            cfg.smoke_transformer_block_step = true;
        } else if (arg == "--smoke-transformer-lm-step") {
            cfg.backend = "tile-cuda";
            cfg.smoke_transformer_lm_step = true;
        } else if (arg == "--smoke-embedding-lm-step") {
            cfg.backend = "tile-cuda";
            cfg.smoke_embedding_lm_step = true;
        } else if (arg == "--train-embedding-lm") {
            cfg.backend = "tile-cuda";
            cfg.train_embedding_lm = true;
        } else if (arg == "--train-transformer-lm") {
            cfg.backend = "tile-cuda";
            cfg.train_transformer_lm = true;
        } else if (arg == "--startup-only") {
            cfg.backend = "tile-cuda";
            cfg.startup_only = true;
            cfg.train_transformer_lm = true;
            cfg.write_checkpoint = false;
        } else if (arg == "--no-train-transformer-lm") {
            cfg.train_transformer_lm = false;
        } else if (arg == "--no-checkpoint" || arg == "--native-cuda-no-checkpoint") {
            cfg.write_checkpoint = false;
        } else if (arg == "--checkpoint" || arg == "--write-checkpoint" || arg == "--native-cuda-write-checkpoint") {
            cfg.write_checkpoint = true;
        } else if (arg == "--checkpoint-metadata-smoke") {
            cfg.backend = "tile-cuda";
            cfg.checkpoint_metadata_smoke = true;
        } else if (arg == "--cuda-runtime-lib") {
            cfg.cuda_runtime_lib = require_value(argc, argv, &i, arg);
        } else if (arg.rfind("--cuda-runtime-lib=", 0) == 0) {
            cfg.cuda_runtime_lib = value_after_equals("--cuda-runtime-lib=");
        } else if (arg == "--json-out" || arg == "--profile-json" || arg == "--stage-profile-json") {
            cfg.json_out_path = require_value(argc, argv, &i, arg);
        } else if (arg.rfind("--json-out=", 0) == 0) {
            cfg.json_out_path = value_after_equals("--json-out=");
        } else if (arg.rfind("--profile-json=", 0) == 0) {
            cfg.json_out_path = value_after_equals("--profile-json=");
        } else if (arg.rfind("--stage-profile-json=", 0) == 0) {
            cfg.json_out_path = value_after_equals("--stage-profile-json=");
        } else if (arg == "--print-plan" || arg == "--json") {
            cfg.print_plan = true;
        } else if (arg == "--output-dir" || arg == "--native-cuda-output-dir") {
            cfg.output_dir = require_value(argc, argv, &i, arg);
        } else if (arg.rfind("--output-dir=", 0) == 0) {
            cfg.output_dir = value_after_equals("--output-dir=");
        } else if (arg == "--eval-every-steps") {
            cfg.eval_every_steps = parse_int(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--eval-batches") {
            cfg.eval_batches = parse_int(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--eval-batch-size") {
            cfg.eval_batch_size = parse_int(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--lm-head-row-chunk-size" || arg == "--native-cuda-lm-head-row-chunk-size") {
            cfg.lm_head_row_chunk_size = parse_int(require_value(argc, argv, &i, arg), arg);
        } else if (arg.rfind("--lm-head-row-chunk-size=", 0) == 0) {
            cfg.lm_head_row_chunk_size = parse_int(value_after_equals("--lm-head-row-chunk-size="), "--lm-head-row-chunk-size");
        } else if (arg.rfind("--native-cuda-lm-head-row-chunk-size=", 0) == 0) {
            cfg.lm_head_row_chunk_size = parse_int(
                value_after_equals("--native-cuda-lm-head-row-chunk-size="),
                "--native-cuda-lm-head-row-chunk-size");
        } else if (arg == "--batch-size") {
            cfg.batch_size = parse_int(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--train-seq-len") {
            cfg.seq_len = parse_int(require_value(argc, argv, &i, arg), arg);
            cfg.seq_len_explicit = true;
        } else if (arg.rfind("--train-seq-len=", 0) == 0) {
            cfg.seq_len = parse_int(value_after_equals("--train-seq-len="), "--train-seq-len");
            cfg.seq_len_explicit = true;
        } else if (arg == "--train-batch-tokens") {
            cfg.train_batch_tokens = parse_int(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--learning-rate") {
            cfg.learning_rate = parse_double(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--final-lr-fraction") {
            cfg.final_lr_fraction = parse_double(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--weight-decay") {
            cfg.weight_decay = parse_double(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--warmup-steps") {
            cfg.warmup_steps = parse_int(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--max-steps") {
            cfg.max_steps = parse_int(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--num-layers") {
            cfg.num_layers = parse_int(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--native-cuda-checkpoint-every") {
            cfg.checkpoint_every_steps = parse_int(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--native-cuda-sample-every") {
            cfg.sample_every_steps = parse_int(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--native-cuda-generate-tokens") {
            cfg.generate_tokens = parse_int(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--activation" || arg == "--native-cuda-activation") {
            cfg.activation = lower_activation(require_value(argc, argv, &i, arg));
        } else if (arg.rfind("--activation=", 0) == 0) {
            cfg.activation = lower_activation(value_after_equals("--activation="));
        } else if (arg == "--moa-interval" || arg == "--native-cuda-moa-interval") {
            cfg.moa_interval = parse_int(require_value(argc, argv, &i, arg), arg);
        } else if (arg == "--allow-train-val-fallback" || arg == "--native-cuda-allow-train-val-fallback") {
            cfg.allow_train_as_val = true;
        } else if (arg == "--dry-run" || arg == "--native-cuda-dry-run") {
            cfg.dry_run = true;
        } else if (arg == "--print-command" || arg == "--native-cuda-print-command") {
            cfg.print_command = true;
        } else {
            std::cerr << "Unknown arg: " << arg << "\n";
            print_usage(argv[0]);
            return 2;
        }
    }

    const std::string model_selector = normalize_model_family(cfg.model_family);
    if (
        model_selector == "gpt3" &&
        !cfg.seq_len_explicit &&
        !cfg.template_explicit &&
        cfg.graph_file.empty() &&
        is_default_gpt_template(cfg)
    ) {
        cfg.seq_len = 2048;
    }
    cfg.model_family = canonical_dense_gpt_model_family(model_selector);
    cfg.activation = lower_activation(cfg.activation);
    cfg.template_name = normalize_template_name(cfg.template_name);
    apply_template_activation_defaults(cfg);
    if (!valid_activation(cfg.activation)) {
        std::cerr << "Invalid activation: " << cfg.activation << "\n";
        return 2;
    }
    cfg.backend = normalize_backend(cfg.backend);
    if (!valid_backend(cfg.backend)) {
        std::cerr << "Invalid backend: " << cfg.backend << "\n";
        return 2;
    }

    if (cfg.backend == "tile-cuda" && cfg.print_command) {
        print_invocation_command(argc, argv);
        return 0;
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

    ScopedStdoutRedirect stdout_redirect;
    if (!cfg.json_out_path.empty()) {
        std::string redirect_error;
        if (!stdout_redirect.open(cfg.json_out_path, &redirect_error)) {
            std::cerr << redirect_error << "\n";
            return 2;
        }
    }

    if (cfg.backend == "tile-cuda") {
        if (cfg.smoke_tile_ops) {
            return print_tile_ops_smoke_json(cfg, argv[0]);
        }
        if (cfg.smoke_optimizer_step) {
            return print_optimizer_smoke_json(cfg, argv[0]);
        }
        if (cfg.smoke_lm_step) {
            return print_lm_step_smoke_json(cfg, argv[0]);
        }
        if (cfg.smoke_attention_step) {
            return print_attention_step_smoke_json(cfg, argv[0]);
        }
        if (cfg.smoke_mlp_step) {
            return print_mlp_step_smoke_json(cfg, argv[0]);
        }
        if (cfg.smoke_norm_residual_step) {
            return print_norm_residual_step_smoke_json(cfg, argv[0]);
        }
        if (cfg.smoke_transformer_block_step) {
            return print_transformer_block_step_smoke_json(cfg, argv[0]);
        }
        if (cfg.check_tile_ops) {
            neuralfn::native_train::TokenShardDataset empty_dataset;
            const bool symbols_ok = print_tile_plan(cfg, empty_dataset, argv[0], true);
            return symbols_ok ? 0 : 2;
        }
    }

    neuralfn::native_train::TokenShardDataset dataset;
    try {
        dataset = neuralfn::native_train::resolve_token_shards(cfg.dataset_alias, cfg.allow_train_as_val);
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n";
        return 2;
    }

    if (cfg.backend == "tile-cuda") {
        if (cfg.smoke_transformer_lm_step) {
            return print_transformer_lm_step_smoke_json(cfg, dataset, argv[0]);
        }
        if (cfg.smoke_embedding_lm_step) {
            return print_embedding_lm_step_smoke_json(cfg, dataset, argv[0]);
        }
        if (cfg.checkpoint_metadata_smoke) {
            return write_checkpoint_metadata_smoke_json(cfg, dataset);
        }
        if (cfg.print_plan || cfg.dry_run) {
            print_tile_plan(cfg, dataset, argv[0], false);
            return 0;
        }
        if ((cfg.train_embedding_lm || cfg.train_transformer_lm) && !selected_graph_is_native_runnable(cfg)) {
            return print_selected_graph_unsupported_json(cfg, dataset);
        }
        if (cfg.train_embedding_lm) {
            return run_embedding_lm_training_json(cfg, dataset, argv[0]);
        }
        if (cfg.train_transformer_lm) {
            return run_transformer_lm_training_json(cfg, dataset, argv[0]);
        }
        std::cerr
            << "nfn_gpt_native_train: NeuralFn Tile CUDA dense GPT trainer loop is not implemented yet.\n"
            << "Use --train-transformer-lm for the current NeuralFn-owned Tile trainer path, or --print-plan / "
            << "--check-tile-ops to inspect the Tile trainer requirements.\n";
        return 2;
    }

    if (cfg.print_plan) {
        std::cout
            << "{\n"
            << "  \"model_family\": \"" << json_escape(cfg.model_family) << "\",\n"
            << "  \"backend\": \"llm-kittens\",\n"
            << "  \"status\": \"external-fast-path\",\n"
            << "  \"template_name\": \"" << json_escape(normalize_template_name(cfg.template_name)) << "\",\n"
            << "  \"resolved_native_template_name\": \"" << json_escape(resolved_native_template_name(cfg.template_name)) << "\",\n"
            << "  \"graph_file\": \"" << json_escape(cfg.graph_file) << "\",\n"
            << "  \"architecture_source\": \"" << json_escape(selected_architecture_source(cfg)) << "\",\n"
            << "  \"architecture_contract\": \"" << json_escape(dense_gpt_architecture_contract(cfg)) << "\",\n"
            << "  \"model_family_context_policy\": \"" << json_escape(model_family_context_policy(cfg)) << "\",\n"
            << "  \"dataset_path\": \"" << json_escape(dataset.dataset_path.string()) << "\",\n"
            << "  \"target\": \"" << json_escape(cfg.target) << "\"\n"
            << "}\n";
        return 0;
    }

    std::vector<std::string> command = build_command(cfg, dataset.train_shards[0].path, dataset.val_shards[0].path);
    if (cfg.print_command || cfg.dry_run) {
        print_command(command);
    }
    if (cfg.dry_run) {
        return 0;
    }
    return exec_command(command);
}
