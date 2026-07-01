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
#include <utility>
#include <vector>

#if defined(_WIN32)
#error "nfn_native_train currently targets POSIX execvp environments."
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

bool env_is_empty(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr || value[0] == '\0';
}

void setenv_default_if_empty(const char* name, const char* value) {
    if (value != nullptr && value[0] != '\0' && env_is_empty(name)) {
        setenv(name, value, 1);
    }
}

struct ModelEntry {
    std::string_view name;
    std::string_view status;
    std::string_view native_target;
    std::string_view transformer_lm_status;
    std::string_view token_lm_status;
    std::string_view geometry_status;
    std::string_view kernel_status;
    std::string_view trainer_loop_status;
    std::string_view notes;
};

constexpr ModelEntry MODEL_REGISTRY[] = {
    {
        "gpt",
        "implemented",
        "nfn_gpt_native_train",
        "native-transformer-lm",
        "not-applicable",
        "dense-gpt-template-geometry",
        "required-tile-symbols-present",
        "implemented",
        "Dense GPT aliases to the NeuralFn Tile-CUDA transformer-LM loop; template/custom graph selection decides the GPT architecture.",
    },
    {
        "gpt2",
        "implemented",
        "nfn_gpt_native_train",
        "native-transformer-lm",
        "not-applicable",
        "dense-gpt-template-geometry",
        "required-tile-symbols-present",
        "implemented",
        "GPT-2 is a dense GPT template selector on the NeuralFn Tile-CUDA transformer-LM loop; template/custom graph selection decides the effective architecture.",
    },
    {
        "gpt3",
        "implemented",
        "nfn_gpt_native_train",
        "native-transformer-lm",
        "not-applicable",
        "dense-gpt-template-geometry",
        "required-tile-symbols-present",
        "implemented",
        "GPT-3-style dense decoder training uses the same GPT native target; context/window and width come from the selected template or custom graph.",
    },
    {
        "gpt2-evo",
        "implemented",
        "nfn_gpt2_evo_native_train",
        "native-dense-gpt-layer-evo-delegate",
        "not-applicable",
        "dense-gpt2-compatible-layer-evo-delegate",
        "required-tile-symbols-present",
        "delegate-to-dense-gpt-loop",
        "GPT-2 evo is a model-aware native C++ preflight/delegate that dispatches dense GPT-2-compatible runs to the CUDA Tile transformer-LM loop with --layer-evo.",
    },
    {
        "nanogpt",
        "implemented",
        "nfn_gpt_native_train",
        "native-transformer-lm",
        "implemented",
        "dense-gpt-template-geometry",
        "required-tile-symbols-present",
        "implemented",
        "NanoGPT routes to the shared dense GPT target with --template-name nanogpt; the native loop now uses the selected 320-wide/5-head/5-layer dense GPT geometry. Pass --train-token-lm for the token-only native preflight.",
    },
    {
        "llama",
        "missing-native-trainer",
        "nfn_llama_native_train",
        "missing-native-trainer",
        "not-applicable",
        "requires-rope-swiglu-native-loop",
        "required-tile-symbols-present",
        "family-native-loop-missing",
        "LLaMA/RoPE/SwiGLU training needs a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "mixllama",
        "missing-native-trainer",
        "nfn_mixllama_native_train",
        "missing-native-trainer",
        "not-applicable",
        "requires-moe-routing-native-loop",
        "required-tile-symbols-present",
        "family-native-loop-missing",
        "MoE routing and expert kernels need a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "jepa",
        "missing-native-trainer",
        "nfn_jepa_native_train",
        "missing-native-trainer",
        "not-applicable",
        "requires-jepa-objective-native-loop",
        "required-tile-symbols-present",
        "family-native-loop-missing",
        "Semantic/JEPA objectives need a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "semantic-dense-jepa",
        "missing-native-trainer",
        "nfn_semantic_dense_jepa_native_train",
        "missing-native-trainer",
        "not-applicable",
        "requires-semantic-dense-jepa-native-loop",
        "required-tile-symbols-present",
        "family-native-loop-missing",
        "Semantic dense JEPA Evo training needs semantic target resolution, planner/projector/predictor wiring, semantic-alignment loss, latent MSE, and AR loss composition.",
    },
    {
        "moe-jepa-evo",
        "missing-native-trainer",
        "nfn_moe_jepa_evo_native_train",
        "missing-native-trainer",
        "not-applicable",
        "requires-moe-jepa-native-loop",
        "required-tile-symbols-present",
        "family-native-loop-missing",
        "MoE JEPA Evo training needs the standard MoE transformer loop plus JEPA target/projector/predictor and composite AR+JEPA+router loss wiring.",
    },
    {
        "auxfree-moe-jepa-evo",
        "missing-native-trainer",
        "nfn_moe_jepa_evo_native_train",
        "missing-native-trainer",
        "not-applicable",
        "requires-moe-jepa-native-loop",
        "required-tile-symbols-present",
        "family-native-loop-missing",
        "Aux-free MoE JEPA Evo shares the MoE+JEPA native trainer target and additionally needs aux-free load balancing integration.",
    },
    {
        "moe-jepa-evo-modern",
        "missing-native-trainer",
        "nfn_moe_jepa_evo_native_train",
        "missing-native-trainer",
        "not-applicable",
        "requires-moe-jepa-native-loop",
        "required-tile-symbols-present",
        "family-native-loop-missing",
        "Modern MoE JEPA Evo shares the MoE+JEPA native trainer target with modern-profile norm/position/MLP overlays.",
    },
    {
        "semantic-router-moe",
        "missing-native-trainer",
        "nfn_semantic_router_moe_native_train",
        "missing-native-trainer",
        "not-applicable",
        "requires-semantic-router-moe-native-loop",
        "required-tile-symbols-present",
        "family-native-loop-missing",
        "Semantic router MoE training needs a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "deepseek-v4",
        "missing-native-trainer",
        "nfn_deepseek_v4_native_train",
        "missing-native-trainer",
        "not-applicable",
        "requires-deepseek-sparse-moe-native-loop",
        "required-tile-symbols-present",
        "family-native-loop-missing",
        "DeepSeek-style sparse/MoE variants need a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "jamba",
        "missing-native-trainer",
        "nfn_jamba_native_train",
        "missing-native-trainer",
        "not-applicable",
        "requires-jamba-hybrid-mamba-native-loop",
        "required-tile-symbols-present",
        "family-native-loop-missing",
        "Jamba hybrid Mamba/transformer training needs a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "seq2seq",
        "missing-native-trainer",
        "nfn_seq2seq_native_train",
        "missing-native-trainer",
        "not-applicable",
        "requires-seq2seq-native-loop",
        "required-tile-symbols-present",
        "family-native-loop-missing",
        "Seq2seq encoder-decoder and cross-attention training needs a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "diffusion",
        "missing-native-trainer",
        "nfn_diffusion_native_train",
        "missing-native-trainer",
        "not-applicable",
        "requires-diffusion-native-loop",
        "required-tile-symbols-present",
        "family-native-loop-missing",
        "Diffusion objective training needs a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "ttt-llama",
        "missing-native-trainer",
        "nfn_ttt_llama_native_train",
        "missing-native-trainer",
        "not-applicable",
        "requires-ttt-native-loop",
        "required-tile-symbols-present",
        "family-native-loop-missing",
        "Test-time-training transformer variants need a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "hnet-lm",
        "missing-native-trainer",
        "nfn_hnet_lm_native_train",
        "missing-native-trainer",
        "not-applicable",
        "requires-hnet-byte-lm-native-loop",
        "required-tile-symbols-present",
        "family-native-loop-missing",
        "HNet byte-LM patching and merge training needs a dedicated native CUDA Tile C++ trainer.",
    },
    {
        "universal-llama",
        "missing-native-trainer",
        "nfn_universal_llama_native_train",
        "missing-native-trainer",
        "not-applicable",
        "requires-universal-transformer-native-loop",
        "required-tile-symbols-present",
        "family-native-loop-missing",
        "Universal transformer recurrent/halting training needs a dedicated native CUDA Tile C++ trainer.",
    },
};

constexpr std::string_view DEFAULT_TINYSTORIES_ALIAS = "roneneldan__TinyStories__TinyStoriesV2-GPT4";

struct NativeGptDefault {
    std::string_view flag;
    std::string_view env_a;
    std::string_view env_b;
    std::string_view env_c;
    std::string_view fallback;
};

constexpr NativeGptDefault NATIVE_GPT_QUALITY_DEFAULTS[] = {
    {"--eval-every-steps", "NFN_NATIVE_GPT_EVAL_EVERY_STEPS", "NFN_SM120_NATIVE_EVAL_EVERY_STEPS", "NFN_SM120_EVAL_EVERY_STEPS", "250"},
    {"--eval-batches", "NFN_NATIVE_GPT_EVAL_BATCHES", "NFN_SM120_NATIVE_EVAL_BATCHES", "NFN_SM120_EVAL_BATCHES", "20"},
    {"--native-cuda-sample-every", "NFN_NATIVE_GPT_SAMPLE_EVERY", "NFN_SM120_NATIVE_SAMPLE_EVERY", "NFN_SM120_SAMPLE_EVERY", "20000"},
    {"--native-cuda-generate-tokens", "NFN_NATIVE_GPT_GENERATE_TOKENS", "NFN_SM120_NATIVE_GENERATE_TOKENS", "NFN_SM120_GENERATE_TOKENS", "144"},
    {"--native-cuda-checkpoint-every", "NFN_NATIVE_GPT_CHECKPOINT_EVERY", "NFN_SM120_NATIVE_CHECKPOINT_EVERY", "NFN_SM120_CHECKPOINT_EVERY", "200"},
    {"--batch-size", "NFN_NATIVE_GPT_BATCH_SIZE", "NFN_SM120_NATIVE_BATCH_SIZE", "NFN_SM120_BATCH_SIZE", "64"},
    {"--train-seq-len", "NFN_NATIVE_GPT_TRAIN_SEQ_LEN", "NFN_SM120_NATIVE_TRAIN_SEQ_LEN", "NFN_SM120_TRAIN_SEQ_LEN", "1024"},
    {"--train-batch-tokens", "NFN_NATIVE_GPT_TRAIN_BATCH_TOKENS", "NFN_SM120_NATIVE_TRAIN_BATCH_TOKENS", "NFN_SM120_TRAIN_BATCH_TOKENS", "524288"},
    {"--learning-rate", "NFN_NATIVE_GPT_LEARNING_RATE", "NFN_SM120_NATIVE_LEARNING_RATE", "NFN_SM120_LEARNING_RATE", "0.0006"},
    {"--final-lr-fraction", "NFN_NATIVE_GPT_FINAL_LR_FRACTION", "NFN_SM120_NATIVE_FINAL_LR_FRACTION", "NFN_SM120_FINAL_LR_FRACTION", "0.0"},
    {"--weight-decay", "NFN_NATIVE_GPT_WEIGHT_DECAY", "NFN_SM120_NATIVE_WEIGHT_DECAY", "NFN_SM120_WEIGHT_DECAY", "0.1"},
    {"--beta1", "NFN_NATIVE_GPT_BETA1", "NFN_SM120_NATIVE_BETA1", "NFN_SM120_BETA1", "0.9"},
    {"--beta2", "NFN_NATIVE_GPT_BETA2", "NFN_SM120_NATIVE_BETA2", "NFN_SM120_BETA2", "0.95"},
    {"--adam-eps", "NFN_NATIVE_GPT_ADAM_EPS", "NFN_SM120_NATIVE_ADAM_EPS", "NFN_SM120_ADAM_EPS", "1e-8"},
    {"--grad-clip-norm", "NFN_NATIVE_GPT_GRAD_CLIP_NORM", "NFN_SM120_NATIVE_GRAD_CLIP_NORM", "NFN_SM120_GRAD_CLIP_NORM", "1.0"},
    {"--warmup-steps", "NFN_NATIVE_GPT_WARMUP_STEPS", "NFN_SM120_NATIVE_WARMUP_STEPS", "NFN_SM120_WARMUP_STEPS", "60"},
    {"--max-steps", "NFN_NATIVE_GPT_MAX_STEPS", "NFN_SM120_NATIVE_MAX_STEPS", "NFN_SM120_MAX_STEPS", "20000"},
};

std::string env_or_empty(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

std::string env_or_default(std::string_view env_a, std::string_view env_b, std::string_view env_c, std::string_view fallback) {
    for (std::string_view name : {env_a, env_b, env_c}) {
        const std::string value = env_or_empty(std::string(name).c_str());
        if (!value.empty()) {
            return value;
        }
    }
    return std::string(fallback);
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

bool has_forwarded_value_flag(const std::vector<std::string>& args, const std::string_view flag);
std::string forwarded_value_or_empty(const std::vector<std::string>& args, const std::string_view flag);

const ModelEntry* template_family_model(const std::vector<std::string>& args) {
    static constexpr std::pair<std::string_view, std::string_view> ALIASES[] = {
        {"llama", "llama"},
        {"llama-fast", "llama"},
        {"llama-fast-megakernel", "llama"},
        {"llama-modern", "llama"},
        {"modern-norms-llama", "llama"},
        {"ternary-b158", "llama"},
        {"ternary-b158-modern", "llama"},
        {"fp8-llama", "llama"},
        {"mxfp4-llama", "llama"},
        {"gemma3", "llama"},
        {"diff-transformer", "llama"},
        {"longctx-sparse-llama", "llama"},
        {"qwen3-longctx", "llama"},
        {"kv-pca-llama", "llama"},
        {"kv-pca-llama-modern", "llama"},
        {"mixllama", "mixllama"},
        {"mixllama-fast", "mixllama"},
        {"mixllama-fast-megakernel", "mixllama"},
        {"moe", "mixllama"},
        {"moe-modern", "mixllama"},
        {"deepseek-v3", "mixllama"},
        {"deepseek-v4", "deepseek-v4"},
        {"llm-jepa", "jepa"},
        {"llm-jepa-modern", "jepa"},
        {"dense-jepa-evo", "jepa"},
        {"dense-jepa-evo-modern", "jepa"},
        {"semantic-dense-jepa-evo", "semantic-dense-jepa"},
        {"semantic-dense-jepa-evo-modern", "semantic-dense-jepa"},
        {"dyt-geglu-semantic-dense-jepa-evo", "semantic-dense-jepa"},
        {"jepa-semantic-hybrid", "semantic-dense-jepa"},
        {"jepa-semantic-hybrid-modern", "semantic-dense-jepa"},
        {"jepa-semantic-hybrid-megakernel", "semantic-dense-jepa"},
        {"moe-jepa-evo", "moe-jepa-evo"},
        {"moe-jepa-evo-modern", "moe-jepa-evo"},
        {"auxfree-moe-jepa-evo", "moe-jepa-evo"},
        {"semantic-router-moe", "semantic-router-moe"},
        {"semantic-router-moe-modern", "semantic-router-moe"},
        {"semantic-router-moe-megakernel", "semantic-router-moe"},
        {"semantic-moe-jepa-evo", "semantic-router-moe"},
        {"semantic-moe-jepa-evo-modern", "semantic-router-moe"},
        {"diff-semantic-moe-jepa-evo", "semantic-router-moe"},
        {"jamba", "jamba"},
        {"jamba-modern", "jamba"},
        {"seq2seq", "seq2seq"},
        {"seq2seq-modern", "seq2seq"},
        {"diffusion", "diffusion"},
        {"diffusion-modern", "diffusion"},
        {"ttt-llama", "ttt-llama"},
        {"ttt-llama-modern", "ttt-llama"},
        {"hnet-lm", "hnet-lm"},
        {"hnet-lm-modern", "hnet-lm"},
        {"universal-llama", "universal-llama"},
        {"universal-llama-modern", "universal-llama"},
    };
    if (has_forwarded_value_flag(args, "--graph-file") || has_forwarded_value_flag(args, "--graph")) {
        return nullptr;
    }
    std::string template_name = forwarded_value_or_empty(args, "--template-name");
    if (template_name.empty()) {
        template_name = forwarded_value_or_empty(args, "--template");
    }
    if (template_name.empty()) {
        template_name = forwarded_value_or_empty(args, "--preset");
    }
    if (template_name.empty()) {
        return nullptr;
    }
    template_name = lower_model(template_name);
    for (const auto& alias : ALIASES) {
        if (alias.first == template_name) {
            return find_model(alias.second);
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

std::string forwarded_value_or_empty(const std::vector<std::string>& args, const std::string_view flag) {
    const std::string prefix(flag);
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == prefix && i + 1 < args.size()) {
            return args[i + 1];
        }
        if (arg.rfind(prefix + "=", 0) == 0) {
            return arg.substr(prefix.size() + 1);
        }
    }
    return {};
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
        "--list-templates",
        "--native-cuda-list-templates",
        "--print-plan",
        "--sample-token-batch",
        "--smoke-attention-step",
        "--smoke-dense-jepa-full-loop-step",
        "--smoke-dense-jepa-train-step",
        "--smoke-diffusion-denoise-step",
        "--smoke-diffusion-objective-step",
        "--smoke-diffusion-full-loop-step",
        "--smoke-embedding-lm-step",
        "--smoke-embedding-norm-step",
        "--smoke-family-layout-checkpoint-step",
        "--smoke-fused-qkv-attention-step",
        "--smoke-hnet-byte-patch-step",
        "--smoke-hnet-byte-patch-backward-step",
        "--smoke-hnet-byte-lm-loop-step",
        "--smoke-jamba-chunk-state-step",
        "--smoke-jamba-mamba-state-step",
        "--smoke-jamba-layer-schedule-step",
        "--smoke-jepa-ar-loss-step",
        "--smoke-jepa-projector-step",
        "--smoke-jepa-target-encoder-step",
        "--smoke-llama-attention-block-step",
        "--smoke-llama-lm-head-step",
        "--smoke-llama-loop",
        "--smoke-llama-token-lm-train-step",
        "--smoke-llama-composed-train-step",
        "--smoke-llama-full-loop-step",
        "--smoke-llama-packed-attention-step",
        "--smoke-llama-rope-attention-block-step",
        "--smoke-llama-rope-block-train-step",
        "--smoke-llama-train-step",
        "--smoke-lm-step",
        "--smoke-mlp-step",
        "--smoke-moe-route-expert-step",
        "--smoke-moe-transformer-block-step",
        "--smoke-moe-transformer-block-train-step",
        "--smoke-moe-transformer-lm-train-step",
        "--smoke-moe-full-loop-step",
        "--smoke-moe-jepa-loss-composition-step",
        "--smoke-norm-residual-step",
        "--smoke-optimizer-step",
        "--smoke-semantic-alignment-step",
        "--smoke-semantic-dense-jepa-train-step",
        "--smoke-semantic-router-moe-train-step",
        "--smoke-semantic-route-loss-step",
        "--smoke-seq2seq-cross-attention-step",
        "--smoke-seq2seq-full-encoder-decoder-loop-step",
        "--smoke-seq2seq-loss-composition-step",
        "--smoke-ttt-composite-inner-step",
        "--smoke-ttt-full-transformer-loop-step",
        "--smoke-ttt-linear-inner-step",
        "--smoke-universal-act-halt-step",
        "--smoke-universal-recurrent-step",
        "--smoke-universal-transformer-loop-step",
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

bool has_native_gpt_metadata_action(const std::vector<std::string>& args) {
    static constexpr std::string_view ACTION_FLAGS[] = {
        "--print-plan",
        "--list-templates",
        "--check-tile-ops",
        "--startup-only",
        "--smoke-tile-ops",
        "--smoke-dense-jepa-full-loop-step",
        "--smoke-dense-jepa-train-step",
        "--smoke-diffusion-denoise-step",
        "--smoke-diffusion-objective-step",
        "--smoke-diffusion-full-loop-step",
        "--smoke-family-layout-checkpoint-step",
        "--smoke-jepa-ar-loss-step",
        "--smoke-jepa-projector-step",
        "--smoke-hnet-byte-patch-step",
        "--smoke-hnet-byte-patch-backward-step",
        "--smoke-hnet-byte-lm-loop-step",
        "--smoke-jamba-chunk-state-step",
        "--smoke-jamba-mamba-state-step",
        "--smoke-jamba-layer-schedule-step",
        "--smoke-jepa-target-encoder-step",
        "--smoke-llama-attention-block-step",
        "--smoke-llama-lm-head-step",
        "--smoke-llama-loop",
        "--smoke-llama-token-lm-train-step",
        "--smoke-llama-composed-train-step",
        "--smoke-llama-full-loop-step",
        "--smoke-llama-packed-attention-step",
        "--smoke-llama-rope-attention-block-step",
        "--smoke-llama-rope-block-train-step",
        "--smoke-llama-train-step",
        "--smoke-moe-route-expert-step",
        "--smoke-moe-transformer-block-step",
        "--smoke-moe-transformer-block-train-step",
        "--smoke-moe-transformer-lm-train-step",
        "--smoke-moe-full-loop-step",
        "--smoke-moe-jepa-loss-composition-step",
        "--smoke-semantic-alignment-step",
        "--smoke-semantic-dense-jepa-train-step",
        "--smoke-semantic-router-moe-train-step",
        "--smoke-semantic-route-loss-step",
        "--smoke-seq2seq-cross-attention-step",
        "--smoke-seq2seq-full-encoder-decoder-loop-step",
        "--smoke-seq2seq-loss-composition-step",
        "--smoke-ttt-composite-inner-step",
        "--smoke-ttt-full-transformer-loop-step",
        "--smoke-ttt-linear-inner-step",
        "--smoke-universal-act-halt-step",
        "--smoke-universal-recurrent-step",
        "--smoke-universal-transformer-loop-step",
        "--smoke-nvfp4-pack",
        "--smoke-optimizer-step",
        "--smoke-lm-step",
        "--smoke-attention-step",
        "--smoke-mlp-step",
        "--smoke-norm-residual-step",
        "--smoke-transformer-block-step",
        "--smoke-transformer-lm-step",
        "--smoke-embedding-lm-step",
    };
    for (std::string_view flag : ACTION_FLAGS) {
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

void append_value_arg(std::vector<std::string>& args, std::string flag, std::string value) {
    args.push_back(std::move(flag));
    args.push_back(std::move(value));
}

void append_native_gpt_quality_defaults(std::vector<std::string>& args) {
    for (const NativeGptDefault& entry : NATIVE_GPT_QUALITY_DEFAULTS) {
        if (!has_forwarded_value_flag(args, entry.flag)) {
            append_value_arg(
                args,
                std::string(entry.flag),
                env_or_default(entry.env_a, entry.env_b, entry.env_c, entry.fallback));
        }
    }
    if (!has_forwarded_value_flag(args, "--activation") &&
        !has_forwarded_value_flag(args, "--native-cuda-activation")) {
        const bool moa_template = forwarded_value_or_empty(args, "--template-name") == "gpt2_moa";
        std::string activation = env_or_empty("NFN_NATIVE_GPT_ACTIVATION");
        if (activation.empty()) {
            activation = env_or_empty("NFN_SM120_ACTIVATION");
        }
        if (activation.empty()) {
            activation = moa_template ? "moa" : "gelu";
        }
        append_value_arg(args, "--native-cuda-activation", activation);
    }
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
                  << " geometry=" << entry.geometry_status
                  << " kernels=" << entry.kernel_status
                  << " loop=" << entry.trainer_loop_status << '\n';
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
            << "\", \"kernel_status\": \"" << entry.kernel_status
            << "\", \"trainer_loop_status\": \"" << entry.trainer_loop_status
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
        << "  --list-templates                Forward dense GPT template catalog lookup without dataset/CUDA setup\n"
        << "  --native-cuda-list-templates    Wrapper alias for --list-templates\n"
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
                "--native-cuda-list-templates",
                "--native-cuda-check-tile-ops",
                "--native-cuda-smoke-tile-ops",
                "--native-cuda-smoke-llama-loop",
                "--native-cuda-smoke-llama-attention-block-step",
                "--native-cuda-smoke-llama-rope-attention-block-step",
                "--native-cuda-smoke-llama-rope-block-train-step",
                "--native-cuda-smoke-llama-lm-head-step",
                "--native-cuda-smoke-llama-token-lm-train-step",
                "--native-cuda-smoke-llama-composed-train-step",
                "--native-cuda-smoke-llama-full-loop-step",
                "--native-cuda-smoke-llama-packed-attention-step",
                "--native-cuda-smoke-llama-train-step",
                "--native-cuda-smoke-dense-jepa-full-loop-step",
                "--native-cuda-smoke-dense-jepa-train-step",
                "--native-cuda-smoke-jepa-ar-loss-step",
                "--native-cuda-smoke-jepa-projector-step",
                "--native-cuda-smoke-jepa-target-encoder-step",
                "--native-cuda-smoke-diffusion-denoise-step",
                "--native-cuda-smoke-diffusion-objective-step",
                "--native-cuda-smoke-diffusion-full-loop-step",
                "--native-cuda-smoke-hnet-byte-patch-step",
                "--native-cuda-smoke-hnet-byte-patch-backward-step",
                "--native-cuda-smoke-hnet-byte-lm-loop-step",
                "--native-cuda-smoke-jamba-chunk-state-step",
                "--native-cuda-smoke-jamba-mamba-state-step",
                "--native-cuda-smoke-jamba-layer-schedule-step",
                "--native-cuda-smoke-family-layout-checkpoint-step",
                "--native-cuda-smoke-moe-route-expert-step",
                "--native-cuda-smoke-moe-transformer-block-step",
                "--native-cuda-smoke-moe-transformer-block-train-step",
                "--native-cuda-smoke-moe-transformer-lm-train-step",
                "--native-cuda-smoke-moe-full-loop-step",
                "--native-cuda-smoke-moe-jepa-loss-composition-step",
                "--native-cuda-smoke-semantic-alignment-step",
                "--native-cuda-smoke-semantic-dense-jepa-train-step",
                "--native-cuda-smoke-semantic-router-moe-train-step",
                "--native-cuda-smoke-semantic-route-loss-step",
                "--native-cuda-smoke-seq2seq-cross-attention-step",
                "--native-cuda-smoke-seq2seq-full-encoder-decoder-loop-step",
                "--native-cuda-smoke-seq2seq-loss-composition-step",
                "--native-cuda-smoke-ttt-composite-inner-step",
                "--native-cuda-smoke-ttt-full-transformer-loop-step",
                "--native-cuda-smoke-ttt-linear-inner-step",
                "--native-cuda-smoke-universal-act-halt-step",
                "--native-cuda-smoke-universal-recurrent-step",
                "--native-cuda-smoke-universal-transformer-loop-step",
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
                "--native-cuda-fast-startup",
                "--fast-startup",
            })) {
            if (arg == "--native-cuda-print-plan") {
                forwarded.push_back("--print-plan");
            } else if (arg == "--native-cuda-list-templates") {
                forwarded.push_back("--list-templates");
            } else if (arg == "--native-cuda-check-tile-ops") {
                forwarded.push_back("--check-tile-ops");
            } else if (arg == "--native-cuda-smoke-tile-ops") {
                forwarded.push_back("--smoke-tile-ops");
            } else if (arg == "--native-cuda-smoke-llama-loop") {
                forwarded.push_back("--smoke-llama-loop");
            } else if (arg == "--native-cuda-smoke-llama-attention-block-step") {
                forwarded.push_back("--smoke-llama-attention-block-step");
            } else if (arg == "--native-cuda-smoke-llama-rope-attention-block-step") {
                forwarded.push_back("--smoke-llama-rope-attention-block-step");
            } else if (arg == "--native-cuda-smoke-llama-rope-block-train-step") {
                forwarded.push_back("--smoke-llama-rope-block-train-step");
            } else if (arg == "--native-cuda-smoke-llama-lm-head-step") {
                forwarded.push_back("--smoke-llama-lm-head-step");
            } else if (arg == "--native-cuda-smoke-llama-token-lm-train-step") {
                forwarded.push_back("--smoke-llama-token-lm-train-step");
            } else if (arg == "--native-cuda-smoke-llama-composed-train-step") {
                forwarded.push_back("--smoke-llama-composed-train-step");
            } else if (arg == "--native-cuda-smoke-llama-full-loop-step") {
                forwarded.push_back("--smoke-llama-full-loop-step");
            } else if (arg == "--native-cuda-smoke-llama-packed-attention-step") {
                forwarded.push_back("--smoke-llama-packed-attention-step");
            } else if (arg == "--native-cuda-smoke-llama-train-step") {
                forwarded.push_back("--smoke-llama-train-step");
            } else if (arg == "--native-cuda-smoke-dense-jepa-full-loop-step") {
                forwarded.push_back("--smoke-dense-jepa-full-loop-step");
            } else if (arg == "--native-cuda-smoke-dense-jepa-train-step") {
                forwarded.push_back("--smoke-dense-jepa-train-step");
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
            } else if (arg == "--native-cuda-smoke-moe-route-expert-step") {
                forwarded.push_back("--smoke-moe-route-expert-step");
            } else if (arg == "--native-cuda-smoke-moe-transformer-block-step") {
                forwarded.push_back("--smoke-moe-transformer-block-step");
            } else if (arg == "--native-cuda-smoke-moe-transformer-block-train-step") {
                forwarded.push_back("--smoke-moe-transformer-block-train-step");
            } else if (arg == "--native-cuda-smoke-moe-transformer-lm-train-step") {
                forwarded.push_back("--smoke-moe-transformer-lm-train-step");
            } else if (arg == "--native-cuda-smoke-moe-full-loop-step") {
                forwarded.push_back("--smoke-moe-full-loop-step");
            } else if (arg == "--native-cuda-smoke-moe-jepa-loss-composition-step") {
                forwarded.push_back("--smoke-moe-jepa-loss-composition-step");
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
            } else if (arg == "--native-cuda-fast-startup" || arg == "--fast-startup") {
                forwarded.push_back("--fast-startup");
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
    bool template_routed_family = false;
    if (model_entry == nullptr) {
        std::vector<std::string> model_selector_args;
        append_value_arg(model_selector_args, "--template-name", model);
        model_entry = template_family_model(model_selector_args);
        template_routed_family = model_entry != nullptr;
    }
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

    if (
        (model_entry->name == std::string_view("gpt") ||
         model_entry->name == std::string_view("gpt2") ||
         model_entry->name == std::string_view("gpt3") ||
         model_entry->name == std::string_view("nanogpt")) &&
        !has_template_catalog_action(forwarded)
    ) {
        if (const ModelEntry* template_entry = template_family_model(forwarded); template_entry != nullptr) {
            model_entry = template_entry;
            template_routed_family = true;
        }
    }

    const bool dense_gpt =
        model_entry->name == std::string_view("gpt") ||
        model_entry->name == std::string_view("gpt2") ||
        model_entry->name == std::string_view("gpt3") ||
        model_entry->name == std::string_view("nanogpt");

    const bool nanogpt_token_lm =
        model_entry->name == std::string_view("nanogpt") &&
        has_forwarded_flag(forwarded, "--train-token-lm");
    const bool missing_family_native_target =
        model_entry->status == std::string_view("missing-native-trainer");

    if (dense_gpt && !has_native_train_action(forwarded)) {
        forwarded.push_back("--train-transformer-lm");
    }
    if (dense_gpt && !has_forwarded_value_flag(forwarded, "--backend")) {
        append_value_arg(forwarded, "--backend", "tile-cuda");
    }
    if (
        dense_gpt &&
        !has_template_catalog_action(forwarded) &&
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
    if (
        model_entry->name == std::string_view("gpt3") &&
        !has_forwarded_value_flag(forwarded, "--batch-size")
    ) {
        append_value_arg(forwarded, "--batch-size", "32");
    }
    if (dense_gpt && !has_native_gpt_metadata_action(forwarded)) {
        append_native_gpt_quality_defaults(forwarded);
    }
    if (missing_family_native_target && !has_template_or_graph_selector(forwarded)) {
        append_value_arg(forwarded, "--template-name", std::string(model_entry->name));
    }
    if (
        missing_family_native_target &&
        !has_any_forwarded_value_flag(forwarded, {"--dataset-alias", "--dataset-path"}) &&
        !has_forwarded_flag(forwarded, "--tinystories")
    ) {
        const std::string dataset_alias = env_or_empty("DATASET_ALIAS");
        append_value_arg(
            forwarded,
            "--dataset-alias",
            dataset_alias.empty() ? std::string(DEFAULT_TINYSTORIES_ALIAS) : dataset_alias);
    }
    if (missing_family_native_target && !template_routed_family && !has_native_train_action(forwarded)) {
        forwarded.push_back("--train-transformer-lm");
    }
    if (missing_family_native_target && !has_native_gpt_metadata_action(forwarded)) {
        append_native_gpt_quality_defaults(forwarded);
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
            setenv_default_if_empty("CUDA_VISIBLE_DEVICES", "0");
            setenv_default_if_empty("CUDA_DEVICE_MAX_CONNECTIONS", "1");
            setenv_default_if_empty("CUDA_MODULE_LOADING", "LAZY");
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
                "required-tile-symbols-present",
                "implemented",
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
            if (model_entry->name == std::string_view("gpt2-evo")) {
                return exec_command(command);
            }
            print_command(command);
            return 0;
        }
        return exec_command(command);
    }

    if (gpt_cli.empty()) {
        std::cerr << "No GPT native CLI configured.\n";
        return 2;
    }

    setenv_default_if_empty("CUDA_VISIBLE_DEVICES", "0");
    setenv_default_if_empty("CUDA_DEVICE_MAX_CONNECTIONS", "1");
    setenv_default_if_empty("CUDA_MODULE_LOADING", "LAZY");

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
