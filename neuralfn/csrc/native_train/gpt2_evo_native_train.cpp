#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Gpt2EvoPlan {
    std::string dataset_alias = "roneneldan__TinyStories__TinyStoriesV2-GPT4";
    std::string output = "artifacts/gpt2_evo.bin";
    std::string optimizer_profile = "adamw";
    std::string tile_activation_dtype = "nvfp4";
    std::int64_t max_steps = 20000;
    std::int64_t train_seq_len = 1024;
    std::int64_t batch_size = 64;
    std::int64_t train_batch_tokens = 524288;
    std::int64_t eval_batches = 20;
    std::int64_t eval_batch_size = 64;
    std::int64_t eval_every_steps = 250;
    std::int64_t warmup_steps = 60;
    std::int64_t vocab_size = 1024;
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
        << "  --max-steps N                   Optimizer steps, default 20000\n"
        << "  --train-seq-len N               Sequence length, default 1024\n"
        << "  --batch-size N                  Microbatch rows, default 64\n"
        << "  --train-batch-tokens N          Effective tokens/step, default 524288\n"
        << "  --eval-every-steps N            Validation cadence, default 250\n"
        << "  --vocab-size N                  Vocabulary size, default 1024\n"
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
        << "  --print-plan                    Print the native JSON plan and exit 0\n"
        << "  --dry-run                       Print the plan, then fail because training is not implemented\n";
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
    std::cout
        << "{\n"
        << "  \"model_family\": \"gpt2-evo\",\n"
        << "  \"status\": \"native-preflight-missing-evo-trainer\",\n"
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
        << "    \"evo_block_parameters\": " << evo_block_parameters << "\n"
        << "  },\n"
        << "  \"estimated_parameters\": " << parameters << ",\n"
        << "  \"available_native_kernels\": [\n"
        << "    \"cached uint16 token-shard dispatch through the dense GPT-2 native CLI\",\n"
        << "    \"AdamW optimizer profile and validation cadence parsed before Python/Torch import\",\n"
        << "    \"NVFP4 activation intent preserved in the compiled native plan\"\n"
        << "  ],\n"
        << "  \"required_native_kernels\": [\n"
        << "    \"native dense GPT-2 forward/backward trainer loop that can expose evo block state\",\n"
        << "    \"forward-only candidate evaluation for current plus mutated evo-layer weights\",\n"
        << "    \"device-side candidate loss reduction and best-candidate selection\",\n"
        << "    \"device-side gaussian mutation of the evo block using deterministic per-step seeds\",\n"
        << "    \"copy/adopt best evo block candidate without host graph-editor tensor flow\",\n"
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
        if (arg == "--print-plan" || arg == "--json") {
            *print_plan = true;
            continue;
        }
        if (arg == "--dry-run" || arg == "--native-cuda-dry-run") {
            *dry_run = true;
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

}  // namespace

int main(int argc, char** argv) {
    bool print_plan = false;
    bool dry_run = false;
    Gpt2EvoPlan plan = parse_args(argc, argv, &print_plan, &dry_run);
    validate_plan(plan);
    if (print_plan || dry_run) {
        print_plan_json(plan);
    }
    if (print_plan && !dry_run) {
        return 0;
    }
    std::cerr
        << "nfn_gpt2_evo_native_train: native CUDA Tile trainer for gpt2-evo is not implemented yet.\n"
        << "The C++ preflight parsed the GPT-2 evo plan; implement the required evo-layer kernels printed by "
        << "--print-plan before production training.\n";
    return 2;
}
