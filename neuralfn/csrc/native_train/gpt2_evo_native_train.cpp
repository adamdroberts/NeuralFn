#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <cerrno>
#include <cstring>
#include <unistd.h>

namespace {

struct Gpt2EvoPlan {
    std::string dataset_alias = "roneneldan__TinyStories__TinyStoriesV2-GPT4";
    std::string output = "artifacts/gpt2_evo.bin";
    std::string optimizer_profile = "adamw";
    std::string tile_activation_dtype = "nvfp4";
    std::string template_name = "gpt2";
    std::string graph_file;
    std::int64_t max_steps = 20000;
    std::int64_t train_seq_len = 1024;
    std::int64_t batch_size = 64;
    std::int64_t train_batch_tokens = 524288;
    std::int64_t eval_batches = 20;
    std::int64_t eval_batch_size = 64;
    std::int64_t eval_every_steps = 250;
    std::int64_t warmup_steps = 60;
    std::int64_t vocab_size = 50257;
    std::int64_t num_layers = 12;
    std::int64_t model_dim = 768;
    std::int64_t num_heads = 12;
    std::int64_t evo_layer_index = 6;
    std::int64_t evo_layer_interval = 10;
    std::int64_t evo_layer_population = 8;
    double learning_rate = 0.0006;
    double weight_decay = 0.1;
    double beta1 = 0.9;
    double beta2 = 0.95;
    double adam_eps = 1e-8;
    double grad_clip_norm = 1.0;
    double evo_layer_mutation_scale = 0.02;
    bool layer_evo_enabled = true;
    bool smoke_evo_kernels = false;
    std::string tile_ops_lib;
    std::string cuda_runtime_lib;
    std::vector<std::string> unparsed_args;
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

std::int64_t ceil_div(std::int64_t lhs, std::int64_t rhs) {
    return (lhs + rhs - 1) / rhs;
}

std::string resolve_tile_ops_lib(const Gpt2EvoPlan& plan, const char* program) {
    if (!plan.tile_ops_lib.empty()) {
        return plan.tile_ops_lib;
    }
    const char* env = std::getenv("NFN_NATIVE_TRAIN_TILE_OPS_LIB");
    if (env != nullptr && std::string_view(env).size() > 0) {
        return std::string(env);
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

std::string resolve_dense_gpt_cli(const char* program) {
    const char* env = std::getenv("NFN_NATIVE_GPT_CLI");
    if (env != nullptr && std::string_view(env).size() > 0) {
        return std::string(env);
    }
    std::filesystem::path exe_path(program);
    if (exe_path.has_parent_path()) {
        std::filesystem::path sibling = exe_path.parent_path() / "nfn_gpt_native_train";
        if (std::filesystem::exists(sibling)) {
            return sibling.string();
        }
    }
    std::filesystem::path build_path = std::filesystem::current_path() / "build" / "nfn_gpt_native_train";
    if (std::filesystem::exists(build_path)) {
        return build_path.string();
    }
    return "nfn_gpt_native_train";
}

std::string output_dir_for_dense_delegate(const Gpt2EvoPlan& plan) {
    if (plan.output.empty()) {
        return "artifacts";
    }
    std::filesystem::path output_path(plan.output);
    if (output_path.has_parent_path()) {
        return output_path.parent_path().string();
    }
    return "artifacts";
}

std::vector<std::string> cuda_runtime_candidates(const Gpt2EvoPlan& plan) {
    if (!plan.cuda_runtime_lib.empty()) {
        return {plan.cuda_runtime_lib};
    }
    const char* env = std::getenv("NFN_CUDA_RUNTIME_LIB");
    if (env != nullptr && std::string_view(env).size() > 0) {
        return {std::string(env)};
    }
    return {"libcudart.so", "libcudart.so.13", "libcudart.so.12"};
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
    return out.empty() ? "gpt2" : out;
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

bool selected_template_is_shipped(const Gpt2EvoPlan& plan) {
    const std::string name = normalize_template_name(plan.template_name);
    const std::vector<std::string>& presets = shipped_gpt_template_presets();
    return std::find(presets.begin(), presets.end(), name) != presets.end();
}

bool selected_template_is_dense_gpt2_compatible(const Gpt2EvoPlan& plan) {
    const std::string name = normalize_template_name(plan.template_name);
    return name == "gpt2" || name == "gpt2_megakernel" || name == "gpt2_moa";
}

bool custom_graph_file_exists(const Gpt2EvoPlan& plan) {
    if (plan.graph_file.empty()) {
        return false;
    }
    std::error_code ec;
    return std::filesystem::exists(plan.graph_file, ec) && !ec;
}

long long custom_graph_file_size_bytes(const Gpt2EvoPlan& plan) {
    if (plan.graph_file.empty()) {
        return -1;
    }
    std::error_code ec;
    if (!std::filesystem::is_regular_file(plan.graph_file, ec) || ec) {
        return -1;
    }
    const auto size = std::filesystem::file_size(plan.graph_file, ec);
    if (ec) {
        return -1;
    }
    return static_cast<long long>(size);
}

std::string selected_graph_support_status(const Gpt2EvoPlan& plan) {
    if (!plan.graph_file.empty()) {
        if (!custom_graph_file_exists(plan)) {
            return "custom-graph-file-missing";
        }
        return "custom-graph-native-trainer-missing";
    }
    if (!selected_template_is_shipped(plan)) {
        return "unknown-template";
    }
    return selected_template_is_dense_gpt2_compatible(plan) ? "native-dense-gpt-layer-evo-delegate"
                                                           : "template-native-trainer-missing";
}

std::int64_t dense_parameter_count(const Gpt2EvoPlan& plan) {
    const std::int64_t token = plan.vocab_size * plan.model_dim;
    const std::int64_t position = plan.train_seq_len * plan.model_dim;
    const std::int64_t per_block =
        2 * plan.model_dim +
        (3 * plan.model_dim * plan.model_dim) +
        (plan.model_dim * plan.model_dim) +
        2 * plan.model_dim +
        (4 * plan.model_dim * plan.model_dim) +
        (plan.model_dim * 4 * plan.model_dim);
    const std::int64_t final_norm = plan.model_dim;
    return token + position + plan.num_layers * per_block + final_norm;
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [native gpt2-evo options]\n\n"
        << "Compiled NeuralFn GPT-2 evo native training preflight.\n"
        << "This target parses the GPT-2 evo training contract in C++ and reports the\n"
        << "native CUDA Tile kernels still required for layer-evolution training.\n\n"
        << "Core options:\n"
        << "  --dataset-alias PATH_OR_ALIAS   Dataset alias or cached shard directory\n"
        << "  --tinystories                   Use TinyStoriesV2 GPT-4 alias\n"
        << "  --output PATH                   Native checkpoint/output path\n"
        << "  --template-name NAME            GPT template preset to select; aliases: --template, --preset\n"
        << "  --graph-file PATH               Custom NeuralFn graph JSON to select; alias: --graph\n"
        << "  --max-steps N                   Optimizer steps, default 20000\n"
        << "  --train-seq-len N               Sequence length, default 1024\n"
        << "  --batch-size N                  Microbatch rows, default 64\n"
        << "  --train-batch-tokens N          Effective tokens/step, default 524288\n"
        << "  --eval-every-steps N            Validation cadence, default 250\n"
        << "  --vocab-size N                  Vocabulary size, default 50257\n"
        << "  --num-layers N                  Transformer layers, default 12\n"
        << "  --model-dim N                   Width, default 768\n"
        << "  --num-heads N                   Attention heads, default 12\n"
        << "  --optimizer-profile adamw       Native optimizer profile; only adamw is accepted\n"
        << "  --tile-cuda-activation-dtype nvfp4|float32|none\n"
        << "  --evo-layer-index N             Evo-trained block index, default 6\n"
        << "  --evo-layer-interval N          Candidate search cadence, default 10\n"
        << "  --evo-layer-population N        Candidate population, default 8\n"
        << "  --evo-layer-mutation-scale X    Gaussian mutation scale, default 0.02\n"
        << "  --no-layer-evo                  Disable evo-layer metadata in the plan\n"
        << "  --tile-ops-lib PATH             Raw libnfn_native_train_tile_ops.so path for evo kernel smoke\n"
        << "  --cuda-runtime-lib PATH         CUDA runtime path for evo kernel smoke; env NFN_CUDA_RUNTIME_LIB also works\n"
        << "  --smoke-evo-kernels             Execute tiny mutate/select/adopt evo Tile kernels and exit\n"
        << "  --print-plan                    Print the native JSON plan and exit 0\n"
        << "  --dry-run                       Print the plan, then fail because training is not implemented\n"
        << "  --native-cuda-*                 Wrapper aliases are accepted for print-plan, smoke, and library paths\n";
}

void validate_plan(const Gpt2EvoPlan& plan) {
    if (plan.optimizer_profile != "adamw") {
        std::cerr << "--optimizer-profile must be adamw for the native GPT-2 evo trainer, got '"
                  << plan.optimizer_profile << "'\n";
        std::exit(2);
    }
    if (plan.tile_activation_dtype != "nvfp4" && plan.tile_activation_dtype != "float32" &&
        plan.tile_activation_dtype != "none") {
        std::cerr << "--tile-cuda-activation-dtype must be nvfp4, float32, or none\n";
        std::exit(2);
    }
    if (plan.train_seq_len <= 0 || plan.batch_size <= 0 || plan.train_batch_tokens <= 0 ||
        plan.max_steps <= 0 || plan.num_layers <= 0 || plan.model_dim <= 0 || plan.num_heads <= 0 ||
        plan.vocab_size <= 0) {
        std::cerr << "schedule and model dimensions must be positive\n";
        std::exit(2);
    }
    if (plan.model_dim % plan.num_heads != 0) {
        std::cerr << "--model-dim must be divisible by --num-heads for native GPT-2 evo\n";
        std::exit(2);
    }
    if (plan.layer_evo_enabled &&
        (plan.evo_layer_index < 0 || plan.evo_layer_index >= plan.num_layers ||
         plan.evo_layer_interval <= 0 || plan.evo_layer_population <= 0 ||
         plan.evo_layer_mutation_scale < 0.0)) {
        std::cerr << "evo layer index/cadence/population/mutation scale are outside the valid range\n";
        std::exit(2);
    }
}

void print_plan_json(const Gpt2EvoPlan& plan) {
    const std::int64_t microbatch_tokens = plan.batch_size * plan.train_seq_len;
    const std::int64_t grad_accum_steps = ceil_div(plan.train_batch_tokens, microbatch_tokens);
    const std::int64_t effective_tokens = grad_accum_steps * microbatch_tokens;
    const std::int64_t head_dim = plan.model_dim / plan.num_heads;
    const std::int64_t parameters = dense_parameter_count(plan);
    const std::int64_t evo_block_parameters =
        2 * plan.model_dim +
        (3 * plan.model_dim * plan.model_dim) +
        (plan.model_dim * plan.model_dim) +
        2 * plan.model_dim +
        (4 * plan.model_dim * plan.model_dim) +
        (plan.model_dim * 4 * plan.model_dim);
    const std::string support_status = selected_graph_support_status(plan);
    const bool native_runnable = plan.graph_file.empty() && selected_template_is_dense_gpt2_compatible(plan);
    const std::string status =
        support_status == "custom-graph-file-missing"
            ? support_status
            : (native_runnable ? "native-preflight-dense-gpt-layer-evo-delegate"
                               : "native-preflight-missing-evo-trainer");
    std::cout
        << "{\n"
        << "  \"model_family\": \"gpt2-evo\",\n"
        << "  \"status\": \"" << json_escape(status) << "\",\n"
        << "  \"template_name\": \"" << json_escape(normalize_template_name(plan.template_name)) << "\",\n"
        << "  \"graph_file\": \"" << json_escape(plan.graph_file) << "\",\n"
        << "  \"graph_file_exists\": " << (custom_graph_file_exists(plan) ? "true" : "false") << ",\n"
        << "  \"graph_file_size_bytes\": " << custom_graph_file_size_bytes(plan) << ",\n"
        << "  \"template_known\": " << (selected_template_is_shipped(plan) ? "true" : "false") << ",\n"
        << "  \"selected_graph_support_status\": \"" << json_escape(support_status) << "\",\n"
        << "  \"selected_graph_native_runnable\": " << (native_runnable ? "true" : "false") << ",\n"
        << "  \"shipped_template_catalog_count\": " << shipped_gpt_template_presets().size() << ",\n"
        << "  \"shipped_template_catalog\": [";
    const std::vector<std::string>& presets = shipped_gpt_template_presets();
    for (std::size_t i = 0; i < presets.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << "\"" << json_escape(presets[i]) << "\"";
    }
    std::cout
        << "],\n"
        << "  \"dataset_alias\": \"" << json_escape(plan.dataset_alias) << "\",\n"
        << "  \"output\": \"" << json_escape(plan.output) << "\",\n"
        << "  \"shape\": {\n"
        << "    \"vocab_size\": " << plan.vocab_size << ",\n"
        << "    \"num_layers\": " << plan.num_layers << ",\n"
        << "    \"model_dim\": " << plan.model_dim << ",\n"
        << "    \"num_heads\": " << plan.num_heads << ",\n"
        << "    \"head_dim\": " << head_dim << "\n"
        << "  },\n"
        << "  \"schedule\": {\n"
        << "    \"max_steps\": " << plan.max_steps << ",\n"
        << "    \"train_seq_len\": " << plan.train_seq_len << ",\n"
        << "    \"batch_size\": " << plan.batch_size << ",\n"
        << "    \"microbatch_tokens\": " << microbatch_tokens << ",\n"
        << "    \"requested_train_batch_tokens\": " << plan.train_batch_tokens << ",\n"
        << "    \"grad_accum_steps\": " << grad_accum_steps << ",\n"
        << "    \"effective_train_batch_tokens\": " << effective_tokens << ",\n"
        << "    \"eval_every_steps\": " << plan.eval_every_steps << ",\n"
        << "    \"eval_batches\": " << plan.eval_batches << ",\n"
        << "    \"eval_batch_size\": " << plan.eval_batch_size << ",\n"
        << "    \"warmup_steps\": " << plan.warmup_steps << "\n"
        << "  },\n"
        << "  \"optimizer\": {\n"
        << "    \"profile\": \"" << json_escape(plan.optimizer_profile) << "\",\n"
        << "    \"learning_rate\": " << plan.learning_rate << ",\n"
        << "    \"weight_decay\": " << plan.weight_decay << ",\n"
        << "    \"beta1\": " << plan.beta1 << ",\n"
        << "    \"beta2\": " << plan.beta2 << ",\n"
        << "    \"adam_eps\": " << plan.adam_eps << ",\n"
        << "    \"grad_clip_norm\": " << plan.grad_clip_norm << "\n"
        << "  },\n"
        << "  \"tile_cuda\": {\n"
        << "    \"activation_dtype\": \"" << json_escape(plan.tile_activation_dtype) << "\",\n"
        << "    \"strict_required\": true\n"
        << "  },\n"
        << "  \"layer_evo\": {\n"
        << "    \"enabled\": " << (plan.layer_evo_enabled ? "true" : "false") << ",\n"
        << "    \"layer_index\": " << plan.evo_layer_index << ",\n"
        << "    \"interval\": " << plan.evo_layer_interval << ",\n"
        << "    \"population\": " << plan.evo_layer_population << ",\n"
        << "    \"mutation_scale\": " << plan.evo_layer_mutation_scale << ",\n"
        << "    \"evo_block_parameters\": " << evo_block_parameters << ",\n"
        << "    \"forward_candidate_eval_enabled\": true,\n"
        << "    \"candidate_loss_source\": \"native-forward-loss-current-batch\",\n"
        << "    \"graph_editor_tensor_flow\": false\n"
        << "  },\n"
        << "  \"estimated_parameters\": " << parameters << ",\n"
        << "  \"available_native_kernels\": [\n"
        << "    \"cached uint16 token-shard dispatch through the dense GPT-2 native CLI\",\n"
        << "    \"AdamW optimizer profile and validation cadence parsed before Python/Torch import\",\n"
        << "    \"NVFP4 activation intent preserved in the compiled native plan\",\n"
        << "    \"template/custom graph selector parsed before graph-backed runtime import\",\n"
        << "    \"device-side evo candidate mutation, best-loss selection, and best-candidate adoption Tile ABI\",\n"
        << "    \"native CUDA forward-only candidate evaluation for current plus mutated evo-layer weights\",\n"
        << "    \"dense GPT native transformer trainer delegate with --layer-evo for GPT-2-compatible templates\"\n"
        << "  ],\n"
        << "  \"required_native_kernels\": [\n"
        << "    \"NVFP4 activation packing over projection and attention inputs in the native trainer\"\n"
        << "  ],\n"
        << "  \"unparsed_args\": [";
    for (std::size_t i = 0; i < plan.unparsed_args.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << "\"" << json_escape(plan.unparsed_args[i]) << "\"";
    }
    std::cout << "]\n}\n";
}

int print_evo_kernel_smoke_json(const Gpt2EvoPlan& plan, const char* program) {
    constexpr std::int64_t kElements = 4;
    constexpr std::int64_t kCandidateCount = 3;
    constexpr float kMutationScale = 0.0f;
    constexpr std::int64_t kSeed = 123;
    constexpr int kCudaMemcpyHostToDevice = 1;
    constexpr int kCudaMemcpyDeviceToHost = 2;

    const std::vector<float> host_base = {1.0f, -2.0f, 0.5f, 4.0f};
    const std::vector<float> host_losses = {3.0f, 1.25f, 2.0f};
    const std::string tile_lib_path = resolve_tile_ops_lib(plan, program);
    std::vector<std::string> runtime_candidates = cuda_runtime_candidates(plan);
    std::string cuda_lib_path = runtime_candidates.empty() ? "libcudart.so" : runtime_candidates.front();
    bool tile_loaded = false;
    bool cuda_runtime_loaded = false;
    bool mutate_loaded = false;
    bool select_loaded = false;
    bool adopt_loaded = false;
    bool passed = false;
    std::int64_t host_best_index = -1;
    float host_best_loss = 0.0f;
    double max_adopt_abs_error = 0.0;
    std::string error;

    using EvoMutateFn = int (*)(
        const float*, float*, std::int64_t, std::int64_t, float, std::int64_t, void*);
    using EvoSelectFn = int (*)(const float*, std::int64_t, std::int64_t*, float*, void*);
    using EvoAdoptFn = int (*)(
        const float*, const std::int64_t*, float*, std::int64_t, std::int64_t, void*);
    using CudaMallocFn = int (*)(void**, std::size_t);
    using CudaFreeFn = int (*)(void*);
    using CudaMemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using CudaDeviceSynchronizeFn = int (*)();
    using CudaGetErrorStringFn = const char* (*)(int);

    void* tile_handle = dlopen(tile_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    void* cuda_handle = nullptr;
    EvoMutateFn mutate = nullptr;
    EvoSelectFn select = nullptr;
    EvoAdoptFn adopt = nullptr;
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
        mutate = load_symbol<EvoMutateFn>(tile_handle, "nfn_native_tile_evo_mutate_candidates_float32");
        select = load_symbol<EvoSelectFn>(tile_handle, "nfn_native_tile_evo_select_best_loss_float32");
        adopt = load_symbol<EvoAdoptFn>(tile_handle, "nfn_native_tile_evo_adopt_candidate_float32");
        mutate_loaded = mutate != nullptr;
        select_loaded = select != nullptr;
        adopt_loaded = adopt != nullptr;
        if (mutate == nullptr || select == nullptr || adopt == nullptr) {
            error = dl_last_error("dlsym evo kernels failed");
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

    float* device_base = nullptr;
    float* device_candidates = nullptr;
    float* device_losses = nullptr;
    std::int64_t* device_best_index = nullptr;
    float* device_best_loss = nullptr;
    float* device_adopted = nullptr;
    auto allocate = [&](auto** ptr, std::size_t bytes, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_malloc(reinterpret_cast<void**>(ptr), bytes);
        if (status != 0) {
            error = cuda_error(status, "cudaMalloc " + name);
        }
    };
    allocate(&device_base, sizeof(float) * kElements, "base");
    allocate(&device_candidates, sizeof(float) * kElements * kCandidateCount, "candidates");
    allocate(&device_losses, sizeof(float) * kCandidateCount, "losses");
    allocate(&device_best_index, sizeof(std::int64_t), "best_index");
    allocate(&device_best_loss, sizeof(float), "best_loss");
    allocate(&device_adopted, sizeof(float) * kElements, "adopted");

    auto copy_to_device = [&](void* dest, const void* source, std::size_t bytes, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_memcpy(dest, source, bytes, kCudaMemcpyHostToDevice);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy host-to-device " + name);
        }
    };
    copy_to_device(device_base, host_base.data(), sizeof(float) * kElements, "base");
    copy_to_device(device_losses, host_losses.data(), sizeof(float) * kCandidateCount, "losses");

    if (error.empty()) {
        int status = mutate(
            device_base,
            device_candidates,
            kElements,
            kCandidateCount,
            kMutationScale,
            kSeed,
            nullptr);
        if (status != 0) {
            error = cuda_error(status, "nfn_native_tile_evo_mutate_candidates_float32");
        }
    }
    if (error.empty()) {
        int status = select(device_losses, kCandidateCount, device_best_index, device_best_loss, nullptr);
        if (status != 0) {
            error = cuda_error(status, "nfn_native_tile_evo_select_best_loss_float32");
        }
    }
    if (error.empty()) {
        int status = adopt(
            device_candidates,
            device_best_index,
            device_adopted,
            kElements,
            kCandidateCount,
            nullptr);
        if (status != 0) {
            error = cuda_error(status, "nfn_native_tile_evo_adopt_candidate_float32");
        }
    }
    if (error.empty()) {
        int status = cuda_device_synchronize();
        if (status != 0) {
            error = cuda_error(status, "cudaDeviceSynchronize");
        }
    }

    std::vector<float> host_adopted(static_cast<std::size_t>(kElements), 0.0f);
    auto copy_from_device = [&](void* dest, const void* source, std::size_t bytes, const std::string& name) {
        if (!error.empty()) {
            return;
        }
        int status = cuda_memcpy(dest, source, bytes, kCudaMemcpyDeviceToHost);
        if (status != 0) {
            error = cuda_error(status, "cudaMemcpy device-to-host " + name);
        }
    };
    copy_from_device(&host_best_index, device_best_index, sizeof(std::int64_t), "best_index");
    copy_from_device(&host_best_loss, device_best_loss, sizeof(float), "best_loss");
    copy_from_device(host_adopted.data(), device_adopted, sizeof(float) * kElements, "adopted");

    if (error.empty()) {
        for (std::int64_t i = 0; i < kElements; ++i) {
            max_adopt_abs_error = std::max(
                max_adopt_abs_error,
                std::fabs(static_cast<double>(host_adopted[static_cast<std::size_t>(i)]) -
                          static_cast<double>(host_base[static_cast<std::size_t>(i)])));
        }
        passed = host_best_index == 1 &&
                 std::fabs(static_cast<double>(host_best_loss) - 1.25) <= 1e-6 &&
                 max_adopt_abs_error <= 1e-6;
        if (!passed) {
            std::ostringstream out;
            out << "evo smoke failed: best_index=" << host_best_index
                << " best_loss=" << host_best_loss
                << " max_adopt_abs_error=" << max_adopt_abs_error;
            error = out.str();
        }
    }

    auto free_device = [&](void* ptr, const std::string& name) {
        if (ptr != nullptr && cuda_free != nullptr) {
            int status = cuda_free(ptr);
            if (status != 0 && error.empty()) {
                error = cuda_error(status, "cudaFree " + name);
            }
        }
    };
    free_device(device_base, "base");
    free_device(device_candidates, "candidates");
    free_device(device_losses, "losses");
    free_device(device_best_index, "best_index");
    free_device(device_best_loss, "best_loss");
    free_device(device_adopted, "adopted");
    if (cuda_handle != nullptr) {
        dlclose(cuda_handle);
    }
    if (tile_handle != nullptr) {
        dlclose(tile_handle);
    }

    std::cout
        << "{\n"
        << "  \"model_family\": \"gpt2-evo\",\n"
        << "  \"smoke\": \"evo_kernels\",\n"
        << "  \"tile_ops_library\": \"" << json_escape(tile_lib_path) << "\",\n"
        << "  \"cuda_runtime_library\": \"" << json_escape(cuda_lib_path) << "\",\n"
        << "  \"loaded\": " << (tile_loaded ? "true" : "false") << ",\n"
        << "  \"cuda_runtime_loaded\": " << (cuda_runtime_loaded ? "true" : "false") << ",\n"
        << "  \"mutate_kernel_loaded\": " << (mutate_loaded ? "true" : "false") << ",\n"
        << "  \"select_kernel_loaded\": " << (select_loaded ? "true" : "false") << ",\n"
        << "  \"adopt_kernel_loaded\": " << (adopt_loaded ? "true" : "false") << ",\n"
        << "  \"elements\": " << kElements << ",\n"
        << "  \"candidate_count\": " << kCandidateCount << ",\n"
        << "  \"mutation_scale\": " << kMutationScale << ",\n"
        << "  \"best_index\": " << host_best_index << ",\n"
        << "  \"best_loss\": " << host_best_loss << ",\n"
        << "  \"max_adopt_abs_error\": " << max_adopt_abs_error << ",\n"
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

Gpt2EvoPlan parse_args(int argc, char** argv, bool* print_plan, bool* dry_run) {
    Gpt2EvoPlan plan;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto value_for = [&](const std::string& flag) {
            return require_value(argc, argv, &i, flag);
        };
        auto after_equals = [&](std::string_view prefix) {
            return arg.substr(prefix.size());
        };
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (arg == "--print-plan" || arg == "--native-cuda-print-plan" || arg == "--json") {
            *print_plan = true;
            continue;
        }
        if (arg == "--dry-run" || arg == "--native-cuda-dry-run") {
            *dry_run = true;
            continue;
        }
        if (arg == "--smoke-evo-kernels" || arg == "--native-cuda-smoke-evo-kernels") {
            plan.smoke_evo_kernels = true;
            continue;
        }
        if (arg == "--tinystories") {
            plan.dataset_alias = "roneneldan__TinyStories__TinyStoriesV2-GPT4";
            continue;
        }
        if (arg == "--dataset-alias" || arg == "--dataset") {
            plan.dataset_alias = value_for(arg);
            continue;
        }
        if (arg.rfind("--dataset-alias=", 0) == 0) {
            plan.dataset_alias = after_equals("--dataset-alias=");
            continue;
        }
        if (arg == "--output") {
            plan.output = value_for(arg);
            continue;
        }
        if (arg.rfind("--output=", 0) == 0) {
            plan.output = after_equals("--output=");
            continue;
        }
        if (arg == "--template-name" || arg == "--template" || arg == "--preset") {
            plan.template_name = normalize_template_name(value_for(arg));
            continue;
        }
        if (arg.rfind("--template-name=", 0) == 0) {
            plan.template_name = normalize_template_name(after_equals("--template-name="));
            continue;
        }
        if (arg.rfind("--template=", 0) == 0) {
            plan.template_name = normalize_template_name(after_equals("--template="));
            continue;
        }
        if (arg.rfind("--preset=", 0) == 0) {
            plan.template_name = normalize_template_name(after_equals("--preset="));
            continue;
        }
        if (arg == "--graph-file" || arg == "--graph") {
            plan.graph_file = value_for(arg);
            continue;
        }
        if (arg.rfind("--graph-file=", 0) == 0) {
            plan.graph_file = after_equals("--graph-file=");
            continue;
        }
        if (arg.rfind("--graph=", 0) == 0) {
            plan.graph_file = after_equals("--graph=");
            continue;
        }
        if (arg == "--max-steps" || arg == "--iterations") {
            plan.max_steps = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--train-seq-len") {
            plan.train_seq_len = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--batch-size") {
            plan.batch_size = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--train-batch-tokens") {
            plan.train_batch_tokens = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--eval-batches") {
            plan.eval_batches = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--eval-batch-size") {
            plan.eval_batch_size = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--eval-every-steps") {
            plan.eval_every_steps = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--warmup-steps") {
            plan.warmup_steps = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--vocab-size") {
            plan.vocab_size = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--num-layers") {
            plan.num_layers = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--model-dim") {
            plan.model_dim = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--num-heads") {
            plan.num_heads = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--optimizer-profile") {
            plan.optimizer_profile = value_for(arg);
            continue;
        }
        if (arg == "--learning-rate") {
            plan.learning_rate = parse_f64(value_for(arg), arg);
            continue;
        }
        if (arg == "--weight-decay") {
            plan.weight_decay = parse_f64(value_for(arg), arg);
            continue;
        }
        if (arg == "--beta1") {
            plan.beta1 = parse_f64(value_for(arg), arg);
            continue;
        }
        if (arg == "--beta2") {
            plan.beta2 = parse_f64(value_for(arg), arg);
            continue;
        }
        if (arg == "--adam-eps") {
            plan.adam_eps = parse_f64(value_for(arg), arg);
            continue;
        }
        if (arg == "--grad-clip-norm") {
            plan.grad_clip_norm = parse_f64(value_for(arg), arg);
            continue;
        }
        if (arg == "--tile-cuda-activation-dtype") {
            plan.tile_activation_dtype = value_for(arg);
            continue;
        }
        if (arg == "--tile-ops-lib" || arg == "--native-cuda-tile-ops-lib") {
            plan.tile_ops_lib = value_for(arg);
            continue;
        }
        if (arg.rfind("--tile-ops-lib=", 0) == 0) {
            plan.tile_ops_lib = after_equals("--tile-ops-lib=");
            continue;
        }
        if (arg.rfind("--native-cuda-tile-ops-lib=", 0) == 0) {
            plan.tile_ops_lib = after_equals("--native-cuda-tile-ops-lib=");
            continue;
        }
        if (arg == "--cuda-runtime-lib" || arg == "--native-cuda-cuda-runtime-lib") {
            plan.cuda_runtime_lib = value_for(arg);
            continue;
        }
        if (arg.rfind("--cuda-runtime-lib=", 0) == 0) {
            plan.cuda_runtime_lib = after_equals("--cuda-runtime-lib=");
            continue;
        }
        if (arg.rfind("--native-cuda-cuda-runtime-lib=", 0) == 0) {
            plan.cuda_runtime_lib = after_equals("--native-cuda-cuda-runtime-lib=");
            continue;
        }
        if (arg == "--evo-layer-index") {
            plan.evo_layer_index = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--evo-layer-interval") {
            plan.evo_layer_interval = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--evo-layer-population") {
            plan.evo_layer_population = parse_i64(value_for(arg), arg);
            continue;
        }
        if (arg == "--evo-layer-mutation-scale") {
            plan.evo_layer_mutation_scale = parse_f64(value_for(arg), arg);
            continue;
        }
        if (arg == "--no-layer-evo") {
            plan.layer_evo_enabled = false;
            continue;
        }
        plan.unparsed_args.push_back(arg);
    }
    return plan;
}

int exec_dense_gpt_delegate(const Gpt2EvoPlan& plan, const char* program) {
    if (!plan.graph_file.empty() || !selected_template_is_dense_gpt2_compatible(plan)) {
        std::cerr
            << "nfn_gpt2_evo_native_train: selected template or graph is not runnable by the dense GPT layer-evo delegate.\n"
            << "Use --print-plan to inspect the native support status.\n";
        return 2;
    }

    std::vector<std::string> args;
    args.push_back(resolve_dense_gpt_cli(program));
    args.push_back("--backend");
    args.push_back("tile-cuda");
    args.push_back("--train-transformer-lm");
    args.push_back("--template-name");
    args.push_back(plan.template_name);
    args.push_back("--dataset-alias");
    args.push_back(plan.dataset_alias);
    args.push_back("--output-dir");
    args.push_back(output_dir_for_dense_delegate(plan));
    args.push_back("--max-steps");
    args.push_back(std::to_string(plan.max_steps));
    args.push_back("--train-seq-len");
    args.push_back(std::to_string(plan.train_seq_len));
    args.push_back("--batch-size");
    args.push_back(std::to_string(plan.batch_size));
    args.push_back("--train-batch-tokens");
    args.push_back(std::to_string(plan.train_batch_tokens));
    args.push_back("--eval-batches");
    args.push_back(std::to_string(plan.eval_batches));
    args.push_back("--eval-batch-size");
    args.push_back(std::to_string(plan.eval_batch_size));
    args.push_back("--eval-every-steps");
    args.push_back(std::to_string(plan.eval_every_steps));
    args.push_back("--warmup-steps");
    args.push_back(std::to_string(plan.warmup_steps));
    args.push_back("--learning-rate");
    args.push_back(std::to_string(plan.learning_rate));
    args.push_back("--weight-decay");
    args.push_back(std::to_string(plan.weight_decay));
    if (!plan.tile_ops_lib.empty()) {
        args.push_back("--tile-ops-lib");
        args.push_back(plan.tile_ops_lib);
    }
    if (!plan.cuda_runtime_lib.empty()) {
        args.push_back("--cuda-runtime-lib");
        args.push_back(plan.cuda_runtime_lib);
    }
    if (plan.layer_evo_enabled) {
        args.push_back("--layer-evo");
        args.push_back("--evo-layer-index");
        args.push_back(std::to_string(plan.evo_layer_index));
        args.push_back("--evo-layer-interval");
        args.push_back(std::to_string(plan.evo_layer_interval));
        args.push_back("--evo-layer-population");
        args.push_back(std::to_string(plan.evo_layer_population));
        args.push_back("--evo-layer-mutation-scale");
        args.push_back(std::to_string(plan.evo_layer_mutation_scale));
    } else {
        args.push_back("--no-layer-evo");
    }
    args.insert(args.end(), plan.unparsed_args.begin(), plan.unparsed_args.end());

    std::vector<char*> delegate_argv;
    delegate_argv.reserve(args.size() + 1);
    for (std::string& arg : args) {
        delegate_argv.push_back(arg.data());
    }
    delegate_argv.push_back(nullptr);
    execvp(args.front().c_str(), delegate_argv.data());
    std::cerr << "nfn_gpt2_evo_native_train: failed to exec dense GPT delegate '"
              << args.front() << "': " << std::strerror(errno) << "\n";
    return 127;
}

}  // namespace

int main(int argc, char** argv) {
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) {
        setenv("CUDA_MODULE_LOADING", "LAZY", 0);
    }
    bool print_plan = false;
    bool dry_run = false;
    Gpt2EvoPlan plan = parse_args(argc, argv, &print_plan, &dry_run);
    validate_plan(plan);
    if (plan.smoke_evo_kernels) {
        return print_evo_kernel_smoke_json(plan, argv[0]);
    }
    if (print_plan || dry_run) {
        print_plan_json(plan);
    }
    if (print_plan && !dry_run) {
        return 0;
    }
    if (dry_run) {
        return 0;
    }
    return exec_dense_gpt_delegate(plan, argv[0]);
}
