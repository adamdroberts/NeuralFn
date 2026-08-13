#include "tile_ops.h"
#include "token_shards.h"
#include "../native_gpt2/resident_sha256.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

bool valid_sha256(std::string_view value);

constexpr int kHostToDevice = 1;
constexpr int kDeviceToHost = 2;
constexpr int kDeviceToDevice = 3;

struct Geometry {
    std::int64_t max_sequence = 131072;
    std::int64_t vocab = 202048;
    std::int64_t layers = 52;
    std::int64_t dim = 6656;
    std::int64_t intermediate = 19968;
    std::int64_t query_heads = 32;
    std::int64_t kv_heads = 2;
    std::int64_t head_dim = 128;
    std::int64_t window = 2048;
    float rope_theta = 500000.0f;
    float norm_eps = 1.0e-5f;
    float post_norm_eps = 1.0e-8f;
    float q_scale = 3.87f;
    float output_multiplier = 0.19611613513818404f;
    float softcap = 20.0f;

    std::int64_t query_width() const { return query_heads * head_dim; }
    std::int64_t kv_width() const { return kv_heads * head_dim; }
    bool local(std::int64_t layer) const { return layer % 4 != 3; }

    void validate() const {
        if (max_sequence <= 0 || vocab <= 1 || layers <= 0 || layers % 4 != 0 ||
            dim <= 0 || intermediate <= 0 || query_heads <= 0 || kv_heads <= 0 ||
            query_heads % kv_heads != 0 || head_dim <= 0 || head_dim % 2 != 0 ||
            window <= 0 || window > max_sequence || !std::isfinite(rope_theta) ||
            !(rope_theta > 0.0f) || !std::isfinite(norm_eps) || !(norm_eps > 0.0f) ||
            !std::isfinite(post_norm_eps) || !(post_norm_eps > 0.0f) ||
            !std::isfinite(q_scale) || !(q_scale > 0.0f) ||
            !std::isfinite(output_multiplier) || !(output_multiplier > 0.0f) ||
            !std::isfinite(softcap) || !(softcap > 0.0f)) {
            throw std::runtime_error("invalid Muse Glimmer training geometry");
        }
    }
};

struct Options {
    Geometry geometry;
    std::string checkpoint;
    std::string checkpoint_sha256;
    std::string reference_checkpoint;
    std::string reference_checkpoint_sha256;
    std::string reward_checkpoint;
    std::string reward_checkpoint_sha256;
    std::string dataset;
    std::string output_dir = "artifacts/muse-glimmer-native";
    std::string resume;
    std::string graph_fingerprint;
    std::string objective = "ar";
    std::string adapter = "none";
    // Empty means the authenticated BF16 source (or the existing NF4 QLoRA
    // conversion).  The two public values select Meta's immutable GGUF
    // K-Quant artifacts.  "test" is accepted only with --tiny-geometry so
    // the packed trainer can be exercised without a 17/20-GB fixture.
    std::string kquant_profile;
    std::vector<std::string> lora_targets{
        "q_proj", "k_proj", "v_proj", "o_proj", "attn_gate_proj",
        "gate_proj", "up_proj", "down_proj"};
    std::string chat_template_sha256;
    std::string tile_ops_lib;
    std::string cuda_runtime_lib;
    std::string nccl_lib;
    std::string distributed_id_file;
    std::int64_t pipeline_parallel_size = 1;
    std::int64_t pipeline_parallel_rank = 0;
    std::int64_t cuda_device = 0;
    std::int64_t distributed_timeout_seconds = 120;
    std::int64_t distributed_reserve_bytes = 2LL * 1024LL * 1024LL * 1024LL;
    std::int64_t max_steps = 1;
    std::int64_t batch_size = 1;
    std::int64_t sequence_length = 128;
    std::int64_t activation_checkpoint_interval = 4;
    std::int64_t checkpoint_every_steps = 1000;
    float learning_rate = 1.0e-5f;
    float beta1 = 0.9f;
    float beta2 = 0.95f;
    float adam_eps = 1.0e-8f;
    float weight_decay = 0.1f;
    float max_grad_norm = 1.0f;
    float dpo_beta = 0.1f;
    float dpo_label_smoothing = 0.0f;
    std::string dpo_loss_type = "sigmoid";
    float kl_coef = 0.1f;
    float ppo_clip = 0.2f;
    float ppo_value_coefficient = 0.5f;
    float ppo_entropy_coefficient = 0.0f;
    float gae_gamma = 1.0f;
    float gae_lambda = 0.95f;
    float rollout_temperature = 1.0f;
    std::int64_t rollout_length = 64;
    std::int64_t ppo_epochs_per_rollout = 4;
    std::int64_t ppo_minibatch_size = 4;
    std::int64_t rollout_top_k = 64;
    std::uint64_t rollout_seed = 20260813ULL;
    std::vector<std::int32_t> eos_token_ids{200001, 200008};
    std::int64_t lora_rank = 8;
    float lora_alpha = 16.0f;
    float lora_dropout = 0.0f;
    std::uint64_t lora_seed = 20260813ULL;
    std::uint64_t reward_head_seed = 20260813ULL;
    std::uint64_t ppo_value_head_seed = 20260813ULL;
    std::int64_t qlora_group_size = 64;
    bool allow_train_as_validation = false;
    bool kernel_check = false;
    bool print_layout = false;
    bool distributed_plan = false;
    bool tiny_geometry = false;
};

std::string require_value(int argc, char** argv, int* index, const char* flag) {
    if (*index + 1 >= argc) throw std::runtime_error(std::string(flag) + " requires a value");
    return argv[++*index];
}

std::int64_t parse_i64(const std::string& value, const char* label) {
    std::size_t consumed = 0;
    const long long parsed = std::stoll(value, &consumed);
    if (consumed != value.size()) throw std::runtime_error(std::string(label) + " is not an integer");
    return static_cast<std::int64_t>(parsed);
}

float parse_float(const std::string& value, const char* label) {
    std::size_t consumed = 0;
    const float parsed = std::stof(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed))
        throw std::runtime_error(std::string(label) + " is not finite");
    return parsed;
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " --checkpoint MODEL.bf16 --checkpoint-sha256 SHA256 --dataset PATH [options]\n"
        << "Exact no-Python Muse Glimmer native C++/CUDA pretraining/SFT core.\n\n"
        << "  --kernel-check                  Validate the complete Glimmer training ABI\n"
        << "  --print-parameter-layout        Print the ordered 627-tensor production table\n"
        << "  --tile-ops-lib PATH             Native Tile-CUDA shared library\n"
        << "  --cuda-runtime-lib PATH         CUDA runtime shared library\n"
        << "  --pipeline-parallel-size N --pipeline-parallel-rank N\n"
        << "  --cuda-device N --nccl-lib PATH --distributed-id-file PATH\n"
        << "  --distributed-timeout-seconds N\n"
        << "  --distributed-reserve-bytes N  Per-rank free-VRAM safety reserve\n"
        << "  --print-distributed-plan        Validate/print stage and byte placement\n"
        << "  --max-steps N --batch-size N --sequence-length N\n"
        << "  --objective {ar,sft,dpo,reward_model,ppo}  Pretraining or post-training objective\n"
        << "  --adapter {none,lora,qlora}    Full update, LoRA, or frozen NF4-base LoRA\n"
        << "  --kquant-profile {k-quant-17gb,k-quant-dynamic}\n"
        << "                                   Frozen packed GGUF base for LoRA (no whole-model dequant)\n"
        << "  --reference-checkpoint PATH --reference-checkpoint-sha256 SHA256 (DPO)\n"
        << "  --dpo-beta F --dpo-label-smoothing F --dpo-loss-type {sigmoid,hinge,ipo}\n"
        << "  --reward-head-seed N           Deterministic reward-head initialization\n"
        << "  --reward-checkpoint DIR --reward-checkpoint-sha256 MANIFEST_SHA256 (PPO)\n"
        << "  --rollout-length N --ppo-epochs-per-rollout N --ppo-minibatch-size N\n"
        << "  --kl-coef F --ppo-clip F --ppo-vf-coef F --ppo-ent-coef F\n"
        << "  --gae-gamma F --gae-lambda F --rollout-temperature F --rollout-top-k N\n"
        << "  --rollout-seed N --eos-token-ids LIST (PPO)\n"
        << "  --ppo-value-head-seed N        Deterministic PPO value-head initialization\n"
        << "  --lora-targets LIST            Comma-separated exact Glimmer projection roles\n"
        << "  --lora-rank N --lora-alpha F --lora-dropout F --lora-seed N\n"
        << "  --qlora-group-size 64          Canonical self-contained NF4 group layout\n"
        << "  --chat-template-sha256 SHA256  Required exact ATEM hash for SFT\n"
        << "  --activation-checkpoint-interval N\n"
        << "  --learning-rate F --weight-decay F --max-grad-norm F\n"
        << "  --output-dir PATH --resume-from-checkpoint PATH\n"
        << "  --tiny-geometry D,F,QH,KVH,HD,L,V,W,MAX (test/oracle only)\n";
}

Options parse_options(int argc, char** argv) {
    Options out;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { print_usage(argv[0]); std::exit(0); }
        if (arg == "--checkpoint") out.checkpoint = require_value(argc, argv, &i, "--checkpoint");
        else if (arg == "--checkpoint-sha256") out.checkpoint_sha256 = require_value(argc, argv, &i, "--checkpoint-sha256");
        else if (arg == "--reference-checkpoint") out.reference_checkpoint = require_value(argc, argv, &i, "--reference-checkpoint");
        else if (arg == "--reference-checkpoint-sha256") out.reference_checkpoint_sha256 = require_value(argc, argv, &i, "--reference-checkpoint-sha256");
        else if (arg == "--reward-checkpoint") out.reward_checkpoint = require_value(argc, argv, &i, "--reward-checkpoint");
        else if (arg == "--reward-checkpoint-sha256") out.reward_checkpoint_sha256 = require_value(argc, argv, &i, "--reward-checkpoint-sha256");
        else if (arg == "--dataset") out.dataset = require_value(argc, argv, &i, "--dataset");
        else if (arg == "--output-dir") out.output_dir = require_value(argc, argv, &i, "--output-dir");
        else if (arg == "--resume-from-checkpoint") out.resume = require_value(argc, argv, &i, "--resume-from-checkpoint");
        else if (arg == "--graph-fingerprint") out.graph_fingerprint = require_value(argc, argv, &i, "--graph-fingerprint");
        else if (arg == "--objective") out.objective = require_value(argc, argv, &i, "--objective");
        else if (arg == "--adapter") out.adapter = require_value(argc, argv, &i, "--adapter");
        else if (arg == "--kquant-profile") out.kquant_profile =
            require_value(argc, argv, &i, "--kquant-profile");
        else if (arg == "--lora-targets") {
            out.lora_targets.clear();
            const std::string raw = require_value(argc, argv, &i, "--lora-targets");
            std::size_t start = 0;
            while (start <= raw.size()) {
                const std::size_t comma = raw.find(',', start);
                const std::string value = raw.substr(start, comma - start);
                if (!value.empty()) out.lora_targets.push_back(value);
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
        }
        else if (arg == "--lora-rank") out.lora_rank = parse_i64(require_value(argc, argv, &i, "--lora-rank"), "LoRA rank");
        else if (arg == "--lora-alpha") out.lora_alpha = parse_float(require_value(argc, argv, &i, "--lora-alpha"), "LoRA alpha");
        else if (arg == "--lora-dropout") out.lora_dropout = parse_float(require_value(argc, argv, &i, "--lora-dropout"), "LoRA dropout");
        else if (arg == "--lora-seed") {
            const std::string value = require_value(argc, argv, &i, "--lora-seed");
            std::size_t consumed = 0;
            out.lora_seed = std::stoull(value, &consumed);
            if (consumed != value.size()) throw std::runtime_error("LoRA seed is not an integer");
        }
        else if (arg == "--reward-head-seed") {
            const std::string value = require_value(argc, argv, &i, "--reward-head-seed");
            std::size_t consumed = 0;
            out.reward_head_seed = std::stoull(value, &consumed);
            if (consumed != value.size())
                throw std::runtime_error("reward-head seed is not an integer");
        }
        else if (arg == "--ppo-value-head-seed") {
            const std::string value = require_value(argc, argv, &i, "--ppo-value-head-seed");
            std::size_t consumed = 0;
            out.ppo_value_head_seed = std::stoull(value, &consumed);
            if (consumed != value.size())
                throw std::runtime_error("PPO value-head seed is not an integer");
        }
        else if (arg == "--qlora-group-size") out.qlora_group_size = parse_i64(
            require_value(argc, argv, &i, "--qlora-group-size"), "QLoRA group size");
        else if (arg == "--chat-template-sha256") out.chat_template_sha256 = require_value(argc, argv, &i, "--chat-template-sha256");
        else if (arg == "--tile-ops-lib") out.tile_ops_lib = require_value(argc, argv, &i, "--tile-ops-lib");
        else if (arg == "--cuda-runtime-lib") out.cuda_runtime_lib = require_value(argc, argv, &i, "--cuda-runtime-lib");
        else if (arg == "--nccl-lib") out.nccl_lib = require_value(argc, argv, &i, "--nccl-lib");
        else if (arg == "--distributed-id-file") out.distributed_id_file = require_value(argc, argv, &i, "--distributed-id-file");
        else if (arg == "--pipeline-parallel-size") out.pipeline_parallel_size = parse_i64(require_value(argc, argv, &i, "--pipeline-parallel-size"), "pipeline parallel size");
        else if (arg == "--pipeline-parallel-rank") out.pipeline_parallel_rank = parse_i64(require_value(argc, argv, &i, "--pipeline-parallel-rank"), "pipeline parallel rank");
        else if (arg == "--cuda-device") out.cuda_device = parse_i64(require_value(argc, argv, &i, "--cuda-device"), "CUDA device");
        else if (arg == "--distributed-timeout-seconds") out.distributed_timeout_seconds = parse_i64(require_value(argc, argv, &i, "--distributed-timeout-seconds"), "distributed timeout");
        else if (arg == "--distributed-reserve-bytes") out.distributed_reserve_bytes = parse_i64(require_value(argc, argv, &i, "--distributed-reserve-bytes"), "distributed reserve bytes");
        else if (arg == "--max-steps") out.max_steps = parse_i64(require_value(argc, argv, &i, "--max-steps"), "max steps");
        else if (arg == "--batch-size") out.batch_size = parse_i64(require_value(argc, argv, &i, "--batch-size"), "batch size");
        else if (arg == "--sequence-length") out.sequence_length = parse_i64(require_value(argc, argv, &i, "--sequence-length"), "sequence length");
        else if (arg == "--activation-checkpoint-interval") out.activation_checkpoint_interval = parse_i64(require_value(argc, argv, &i, "--activation-checkpoint-interval"), "activation checkpoint interval");
        else if (arg == "--checkpoint-every-steps") out.checkpoint_every_steps = parse_i64(require_value(argc, argv, &i, "--checkpoint-every-steps"), "checkpoint interval");
        else if (arg == "--learning-rate") out.learning_rate = parse_float(require_value(argc, argv, &i, "--learning-rate"), "learning rate");
        else if (arg == "--weight-decay") out.weight_decay = parse_float(require_value(argc, argv, &i, "--weight-decay"), "weight decay");
        else if (arg == "--max-grad-norm") out.max_grad_norm = parse_float(require_value(argc, argv, &i, "--max-grad-norm"), "max grad norm");
        else if (arg == "--dpo-beta") out.dpo_beta = parse_float(require_value(argc, argv, &i, "--dpo-beta"), "DPO beta");
        else if (arg == "--dpo-label-smoothing") out.dpo_label_smoothing = parse_float(require_value(argc, argv, &i, "--dpo-label-smoothing"), "DPO label smoothing");
        else if (arg == "--dpo-loss-type") out.dpo_loss_type = require_value(argc, argv, &i, "--dpo-loss-type");
        else if (arg == "--kl-coef") out.kl_coef = parse_float(require_value(argc, argv, &i, "--kl-coef"), "KL coefficient");
        else if (arg == "--ppo-clip") out.ppo_clip = parse_float(require_value(argc, argv, &i, "--ppo-clip"), "PPO clip");
        else if (arg == "--ppo-vf-coef") out.ppo_value_coefficient = parse_float(require_value(argc, argv, &i, "--ppo-vf-coef"), "PPO value coefficient");
        else if (arg == "--ppo-ent-coef") out.ppo_entropy_coefficient = parse_float(require_value(argc, argv, &i, "--ppo-ent-coef"), "PPO entropy coefficient");
        else if (arg == "--gae-gamma") out.gae_gamma = parse_float(require_value(argc, argv, &i, "--gae-gamma"), "GAE gamma");
        else if (arg == "--gae-lambda") out.gae_lambda = parse_float(require_value(argc, argv, &i, "--gae-lambda"), "GAE lambda");
        else if (arg == "--rollout-temperature") out.rollout_temperature = parse_float(require_value(argc, argv, &i, "--rollout-temperature"), "rollout temperature");
        else if (arg == "--rollout-length") out.rollout_length = parse_i64(require_value(argc, argv, &i, "--rollout-length"), "rollout length");
        else if (arg == "--ppo-epochs-per-rollout") out.ppo_epochs_per_rollout = parse_i64(require_value(argc, argv, &i, "--ppo-epochs-per-rollout"), "PPO epochs per rollout");
        else if (arg == "--ppo-minibatch-size") out.ppo_minibatch_size = parse_i64(require_value(argc, argv, &i, "--ppo-minibatch-size"), "PPO minibatch size");
        else if (arg == "--rollout-top-k") out.rollout_top_k = parse_i64(require_value(argc, argv, &i, "--rollout-top-k"), "rollout top-k");
        else if (arg == "--rollout-seed") {
            const std::string value = require_value(argc, argv, &i, "--rollout-seed");
            std::size_t consumed = 0;
            out.rollout_seed = std::stoull(value, &consumed);
            if (consumed != value.size())
                throw std::runtime_error("rollout seed is not an integer");
        }
        else if (arg == "--eos-token-ids") {
            out.eos_token_ids.clear();
            const std::string raw = require_value(argc, argv, &i, "--eos-token-ids");
            std::size_t start = 0;
            while (start <= raw.size()) {
                const std::size_t comma = raw.find(',', start);
                const std::string value = raw.substr(start, comma - start);
                if (!value.empty()) {
                    const auto parsed = parse_i64(value, "EOS token ID");
                    if (parsed < 0 || parsed > std::numeric_limits<std::int32_t>::max())
                        throw std::runtime_error("EOS token ID is outside int32");
                    out.eos_token_ids.push_back(static_cast<std::int32_t>(parsed));
                }
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
        }
        else if (arg == "--allow-train-as-validation") out.allow_train_as_validation = true;
        else if (arg == "--kernel-check") out.kernel_check = true;
        else if (arg == "--print-parameter-layout") out.print_layout = true;
        else if (arg == "--print-distributed-plan") out.distributed_plan = true;
        else if (arg == "--tiny-geometry") {
            out.tiny_geometry = true;
            std::string raw = require_value(argc, argv, &i, "--tiny-geometry");
            std::vector<std::int64_t> values;
            std::size_t start = 0;
            while (start <= raw.size()) {
                const std::size_t comma = raw.find(',', start);
                values.push_back(parse_i64(raw.substr(start, comma - start), "tiny geometry"));
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
            if (values.size() != 9) throw std::runtime_error("--tiny-geometry needs D,F,QH,KVH,HD,L,V,W,MAX");
            out.geometry.dim = values[0]; out.geometry.intermediate = values[1];
            out.geometry.query_heads = values[2]; out.geometry.kv_heads = values[3];
            out.geometry.head_dim = values[4]; out.geometry.layers = values[5];
            out.geometry.vocab = values[6]; out.geometry.window = values[7];
            out.geometry.max_sequence = values[8];
        } else {
            throw std::runtime_error("unknown Muse Glimmer native trainer option: " + arg);
        }
    }
    out.geometry.validate();
    if (out.max_steps <= 0 || out.batch_size <= 0 || out.sequence_length <= 0 ||
        out.sequence_length > out.geometry.max_sequence ||
        out.activation_checkpoint_interval <= 0 || out.learning_rate < 0.0f ||
        out.checkpoint_every_steps <= 0 ||
        out.weight_decay < 0.0f || !(out.max_grad_norm > 0.0f)) {
        throw std::runtime_error("invalid Muse Glimmer training options");
    }
    if (out.objective != "ar" && out.objective != "sft" &&
        out.objective != "dpo" && out.objective != "reward_model" &&
        out.objective != "ppo") {
        throw std::runtime_error(
            "--objective must be ar, sft, dpo, reward_model, or ppo");
    }
    if (out.adapter != "none" && out.adapter != "lora" && out.adapter != "qlora") {
        throw std::runtime_error("--adapter must be none, lora, or qlora");
    }
    if (
        out.pipeline_parallel_size <= 0 || out.pipeline_parallel_rank < 0 ||
        out.pipeline_parallel_rank >= out.pipeline_parallel_size ||
        out.pipeline_parallel_size > out.geometry.layers || out.cuda_device < 0 ||
        out.distributed_timeout_seconds <= 0 || out.distributed_reserve_bytes < 0) {
        throw std::runtime_error("invalid Muse Glimmer pipeline-parallel options");
    }
    if (out.pipeline_parallel_size > 1) {
        if (out.objective != "ar" && out.objective != "sft") {
            throw std::runtime_error(
                "pipeline-parallel Muse Glimmer training currently supports AR or SFT");
        }
        if (out.adapter != "none" || !out.kquant_profile.empty()) {
            throw std::runtime_error(
                "pipeline-parallel Muse Glimmer training requires a full BF16 update");
        }
        if (out.distributed_id_file.empty() && !out.distributed_plan) {
            throw std::runtime_error(
                "pipeline-parallel training requires --distributed-id-file");
        }
    } else if (
        out.pipeline_parallel_rank != 0 || !out.distributed_id_file.empty() ||
        !out.nccl_lib.empty()) {
        throw std::runtime_error(
            "distributed options require --pipeline-parallel-size greater than one");
    }
    if (!out.kquant_profile.empty() &&
        out.kquant_profile != "k-quant-17gb" &&
        out.kquant_profile != "k-quant-dynamic" &&
        out.kquant_profile != "test") {
        throw std::runtime_error(
            "--kquant-profile must be k-quant-17gb or k-quant-dynamic");
    }
    if (out.kquant_profile == "test" && !out.tiny_geometry) {
        throw std::runtime_error("the test K-Quant profile requires --tiny-geometry");
    }
    if (!out.kquant_profile.empty() && out.kquant_profile != "test" &&
        out.tiny_geometry) {
        throw std::runtime_error(
            "official K-Quant profiles require the production Muse Glimmer geometry");
    }
    if (!out.kquant_profile.empty() &&
        (out.adapter != "lora" ||
         (out.objective != "sft" && out.objective != "dpo"))) {
        throw std::runtime_error(
            "K-Quant is an immutable base and currently requires LoRA with SFT or DPO; "
            "PPO remains gated until the frozen reward artifact has a packed-base contract");
    }
    const std::unordered_set<std::string> allowed_lora_targets{
        "q_proj", "k_proj", "v_proj", "o_proj", "attn_gate_proj",
        "gate_proj", "up_proj", "down_proj"};
    std::unordered_set<std::string> unique_lora_targets;
    for (const auto& target : out.lora_targets) {
        if (!allowed_lora_targets.contains(target))
            throw std::runtime_error("unsupported Muse Glimmer LoRA target: " + target);
        if (!unique_lora_targets.insert(target).second)
            throw std::runtime_error("duplicate Muse Glimmer LoRA target: " + target);
    }
    if (out.adapter != "none" &&
        ((out.objective != "sft" && out.objective != "dpo" &&
          out.objective != "ppo") ||
         out.lora_targets.empty())) {
        throw std::runtime_error(
            "native Muse Glimmer LoRA/QLoRA requires SFT, DPO, or PPO and at least one target");
    }
    if (out.adapter != "none" &&
        (out.lora_rank <= 0 || !(out.lora_alpha > 0.0f) ||
         out.lora_dropout < 0.0f || out.lora_dropout >= 1.0f)) {
        throw std::runtime_error("invalid Muse Glimmer LoRA rank/alpha/dropout");
    }
    if (out.adapter == "qlora" && out.qlora_group_size != 64) {
        throw std::runtime_error("native Muse Glimmer QLoRA requires group size 64");
    }
    if (out.adapter == "none" &&
        (out.lora_rank != 8 || out.lora_alpha != 16.0f || out.lora_dropout != 0.0f ||
         out.lora_seed != 20260813ULL || out.qlora_group_size != 64)) {
        throw std::runtime_error("LoRA tuning parameters require --adapter lora or qlora");
    }
    if ((out.objective == "sft" || out.objective == "dpo" ||
         out.objective == "reward_model" || out.objective == "ppo") &&
        !valid_sha256(out.chat_template_sha256)) {
        throw std::runtime_error(
            "--chat-template-sha256 is required for post-training");
    }
    if (out.objective == "ar" && !out.chat_template_sha256.empty()) {
        throw std::runtime_error(
            "--chat-template-sha256 is only valid for post-training");
    }
    if (out.objective == "dpo" || out.objective == "ppo") {
        if (out.reference_checkpoint.empty() ||
            !valid_sha256(out.reference_checkpoint_sha256)) {
            throw std::runtime_error(
                "DPO/PPO requires --reference-checkpoint and "
                "--reference-checkpoint-sha256");
        }
        if (!out.kquant_profile.empty() &&
            (out.reference_checkpoint_sha256 != out.checkpoint_sha256 ||
             fs::path(out.reference_checkpoint) != fs::path(out.checkpoint))) {
            throw std::runtime_error(
                "K-Quant DPO/PPO requires the reference to be the identical frozen packed base");
        }
    }
    if (out.objective == "dpo") {
        if (!(out.dpo_beta > 0.0f) || out.dpo_label_smoothing < 0.0f ||
            out.dpo_label_smoothing > 1.0f ||
            (out.dpo_loss_type != "sigmoid" && out.dpo_loss_type != "hinge" &&
             out.dpo_loss_type != "ipo")) {
            throw std::runtime_error("invalid native Muse Glimmer DPO configuration");
        }
    } else if ((out.objective != "ppo" &&
                (!out.reference_checkpoint.empty() ||
                 !out.reference_checkpoint_sha256.empty())) ||
               out.dpo_beta != 0.1f || out.dpo_label_smoothing != 0.0f ||
               out.dpo_loss_type != "sigmoid") {
        throw std::runtime_error("DPO options require --objective dpo");
    }
    if (out.objective == "ppo") {
        if (out.reward_checkpoint.empty() ||
            !valid_sha256(out.reward_checkpoint_sha256)) {
            throw std::runtime_error(
                "PPO requires --reward-checkpoint and "
                "--reward-checkpoint-sha256");
        }
        if (!(out.kl_coef >= 0.0f) || !(out.ppo_clip > 0.0f) ||
            out.ppo_clip >= 1.0f || !(out.ppo_value_coefficient >= 0.0f) ||
            !(out.ppo_entropy_coefficient >= 0.0f) ||
            out.gae_gamma < 0.0f || out.gae_gamma > 1.0f ||
            out.gae_lambda < 0.0f || out.gae_lambda > 1.0f ||
            !(out.rollout_temperature > 0.0f) || out.rollout_length <= 0 ||
            out.rollout_length >= out.sequence_length ||
            out.ppo_epochs_per_rollout <= 0 || out.ppo_minibatch_size <= 0 ||
            out.ppo_minibatch_size != out.batch_size || out.rollout_top_k < 0 ||
            out.rollout_top_k > out.geometry.vocab || out.eos_token_ids.empty() ||
            std::any_of(
                out.eos_token_ids.begin(), out.eos_token_ids.end(),
                [&](std::int32_t token) {
                    return token < 0 || token >= out.geometry.vocab;
                })) {
            throw std::runtime_error("invalid native Muse Glimmer PPO configuration");
        }
    } else if (!out.reward_checkpoint.empty() ||
               !out.reward_checkpoint_sha256.empty() || out.kl_coef != 0.1f ||
               out.ppo_clip != 0.2f || out.ppo_value_coefficient != 0.5f ||
               out.ppo_entropy_coefficient != 0.0f || out.gae_gamma != 1.0f ||
               out.gae_lambda != 0.95f || out.rollout_temperature != 1.0f ||
               out.rollout_length != 64 || out.ppo_epochs_per_rollout != 4 ||
               out.ppo_minibatch_size != 4 || out.rollout_top_k != 64 ||
               out.rollout_seed != 20260813ULL ||
               out.ppo_value_head_seed != 20260813ULL ||
               out.eos_token_ids != std::vector<std::int32_t>{200001, 200008}) {
        throw std::runtime_error("PPO options require --objective ppo");
    }
    if (out.objective != "reward_model" &&
        out.reward_head_seed != 20260813ULL) {
        throw std::runtime_error(
            "--reward-head-seed requires --objective reward_model");
    }
    return out;
}

template <typename Fn>
Fn symbol(void* handle, const char* name) {
    void* raw = handle == nullptr ? nullptr : dlsym(handle, name);
    if (raw == nullptr) throw std::runtime_error(std::string("missing required symbol: ") + name);
    return reinterpret_cast<Fn>(raw);
}

struct Runtime {
    using SetDeviceFn = int (*)(int);
    using MallocFn = int (*)(void**, std::size_t);
    using FreeFn = int (*)(void*);
    using MemcpyFn = int (*)(void*, const void*, std::size_t, int);
    using MemsetAsyncFn = int (*)(void*, int, std::size_t, void*);
    using StreamCreateFn = int (*)(void**);
    using StreamDestroyFn = int (*)(void*);
    using StreamSyncFn = int (*)(void*);
    using MemGetInfoFn = int (*)(std::size_t*, std::size_t*);
    using ErrorFn = const char* (*)(int);

    void* handle = nullptr;
    SetDeviceFn set_device = nullptr;
    MallocFn malloc = nullptr;
    FreeFn free = nullptr;
    MemcpyFn memcpy = nullptr;
    MemsetAsyncFn memset_async = nullptr;
    StreamCreateFn stream_create = nullptr;
    StreamDestroyFn stream_destroy = nullptr;
    StreamSyncFn stream_sync = nullptr;
    MemGetInfoFn mem_get_info = nullptr;
    ErrorFn error_string = nullptr;
    void* stream = nullptr;

    explicit Runtime(const std::string& path, std::int64_t device = 0) {
        const std::vector<std::string> candidates = path.empty()
            ? std::vector<std::string>{"libcudart.so", "libcudart.so.13", "libcudart.so.12"}
            : std::vector<std::string>{path};
        for (const auto& candidate : candidates) {
            handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (handle != nullptr) break;
        }
        if (handle == nullptr) throw std::runtime_error("failed to load CUDA runtime");
        dlerror();
        set_device = reinterpret_cast<SetDeviceFn>(dlsym(handle, "cudaSetDevice"));
        (void)dlerror();
        if (set_device != nullptr) {
            if (device > std::numeric_limits<int>::max())
                throw std::runtime_error("CUDA device index exceeds int");
            check(set_device(static_cast<int>(device)), "cudaSetDevice");
        } else if (device != 0) {
            throw std::runtime_error(
                "selected CUDA runtime does not expose cudaSetDevice");
        }
        malloc = symbol<MallocFn>(handle, "cudaMalloc");
        free = symbol<FreeFn>(handle, "cudaFree");
        memcpy = symbol<MemcpyFn>(handle, "cudaMemcpy");
        memset_async = symbol<MemsetAsyncFn>(handle, "cudaMemsetAsync");
        stream_create = symbol<StreamCreateFn>(handle, "cudaStreamCreate");
        stream_destroy = symbol<StreamDestroyFn>(handle, "cudaStreamDestroy");
        stream_sync = symbol<StreamSyncFn>(handle, "cudaStreamSynchronize");
        error_string = symbol<ErrorFn>(handle, "cudaGetErrorString");
        dlerror();
        mem_get_info = reinterpret_cast<MemGetInfoFn>(dlsym(handle, "cudaMemGetInfo"));
        (void)dlerror();
        check(stream_create(&stream), "cudaStreamCreate");
    }
    ~Runtime() {
        if (stream && stream_destroy) stream_destroy(stream);
        if (handle) dlclose(handle);
    }
    void check(int status, const char* action) const {
        if (status != 0) {
            const char* detail = error_string ? error_string(status) : nullptr;
            throw std::runtime_error(std::string(action) + " failed" +
                (detail ? std::string(": ") + detail : std::string()));
        }
    }
    void sync() const { check(stream_sync(stream), "cudaStreamSynchronize"); }
    std::pair<std::size_t, std::size_t> memory_info() const {
        if (mem_get_info == nullptr)
            throw std::runtime_error("CUDA runtime does not expose cudaMemGetInfo");
        std::size_t free = 0, total = 0;
        check(mem_get_info(&free, &total), "cudaMemGetInfo");
        if (free == 0 || total == 0 || free > total)
            throw std::runtime_error("CUDA memory information is invalid");
        return {free, total};
    }
};

struct NcclUniqueIdV1 {
    std::array<char, 128> bytes{};
};

class PipelineCollective {
public:
    using GetUniqueIdFn = int (*)(NcclUniqueIdV1*);
    using CommInitRankFn = int (*)(void**, int, NcclUniqueIdV1, int);
    using CommDestroyFn = int (*)(void*);
    using SendFn = int (*)(const void*, std::size_t, int, int, void*, void*);
    using RecvFn = int (*)(void*, std::size_t, int, int, void*, void*);
    using AllReduceFn = int (*)(const void*, void*, std::size_t, int, int, void*, void*);
    using ErrorFn = const char* (*)(int);

    PipelineCollective(Runtime& runtime, const Options& options)
        : runtime_(runtime), world_size_(options.pipeline_parallel_size),
          rank_(options.pipeline_parallel_rank) {
        if (world_size_ <= 1) return;
        const std::vector<std::string> candidates = options.nccl_lib.empty()
            ? std::vector<std::string>{"libnccl.so", "libnccl.so.2"}
            : std::vector<std::string>{options.nccl_lib};
        for (const auto& candidate : candidates) {
            handle_ = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (handle_ != nullptr) break;
        }
        if (handle_ == nullptr) {
            throw std::runtime_error("failed to load NCCL for pipeline training");
        }
        get_unique_id_ = symbol<GetUniqueIdFn>(handle_, "ncclGetUniqueId");
        comm_init_rank_ = symbol<CommInitRankFn>(handle_, "ncclCommInitRank");
        comm_destroy_ = symbol<CommDestroyFn>(handle_, "ncclCommDestroy");
        send_ = symbol<SendFn>(handle_, "ncclSend");
        recv_ = symbol<RecvFn>(handle_, "ncclRecv");
        all_reduce_ = symbol<AllReduceFn>(handle_, "ncclAllReduce");
        error_string_ = symbol<ErrorFn>(handle_, "ncclGetErrorString");
        const NcclUniqueIdV1 id = bootstrap_id(
            fs::path(options.distributed_id_file),
            options.distributed_timeout_seconds);
        check(
            comm_init_rank_(
                &comm_, static_cast<int>(world_size_), id,
                static_cast<int>(rank_)),
            "ncclCommInitRank");
    }

    ~PipelineCollective() {
        if (comm_ != nullptr && comm_destroy_ != nullptr) {
            comm_destroy_(comm_);
        }
        if (handle_ != nullptr) dlclose(handle_);
    }

    PipelineCollective(const PipelineCollective&) = delete;
    PipelineCollective& operator=(const PipelineCollective&) = delete;

    bool enabled() const { return world_size_ > 1; }
    std::int64_t rank() const { return rank_; }
    std::int64_t world_size() const { return world_size_; }

    void send_float(const float* data, std::int64_t count, std::int64_t peer) {
        validate_transfer(data, count, peer);
        check(send_(data, static_cast<std::size_t>(count), kNcclFloat32,
                    static_cast<int>(peer), comm_, runtime_.stream),
              "ncclSend");
    }

    void recv_float(float* data, std::int64_t count, std::int64_t peer) {
        validate_transfer(data, count, peer);
        check(recv_(data, static_cast<std::size_t>(count), kNcclFloat32,
                    static_cast<int>(peer), comm_, runtime_.stream),
              "ncclRecv");
    }

    float all_reduce_sum(float local) {
        float* send = nullptr;
        float* receive = nullptr;
        runtime_.check(
            runtime_.malloc(reinterpret_cast<void**>(&send), sizeof(float)),
            "cudaMalloc NCCL scalar send");
        try {
            runtime_.check(
                runtime_.malloc(reinterpret_cast<void**>(&receive), sizeof(float)),
                "cudaMalloc NCCL scalar receive");
            runtime_.check(
                runtime_.memcpy(send, &local, sizeof(float), kHostToDevice),
                "cudaMemcpy NCCL scalar H2D");
            check(all_reduce_(send, receive, 1, kNcclFloat32,
                              kNcclSum, comm_, runtime_.stream),
                  "ncclAllReduce");
            runtime_.sync();
            float result = 0.0f;
            runtime_.check(
                runtime_.memcpy(&result, receive, sizeof(float), kDeviceToHost),
                "cudaMemcpy NCCL scalar D2H");
            runtime_.free(receive);
            runtime_.free(send);
            return result;
        } catch (...) {
            if (receive != nullptr) runtime_.free(receive);
            if (send != nullptr) runtime_.free(send);
            throw;
        }
    }

    void barrier() { (void)all_reduce_sum(0.0f); }

private:
    static constexpr int kNcclFloat32 = 7;
    static constexpr int kNcclSum = 0;

    template <typename T>
    static T symbol(void* handle, const char* name) {
        dlerror();
        void* value = dlsym(handle, name);
        const char* error = dlerror();
        if (error != nullptr || value == nullptr) {
            throw std::runtime_error(
                std::string("missing NCCL symbol ") + name);
        }
        return reinterpret_cast<T>(value);
    }

    void check(int status, const char* operation) const {
        if (status == 0) return;
        const char* detail = error_string_ ? error_string_(status) : nullptr;
        throw std::runtime_error(
            std::string(operation) + " failed" +
            (detail ? std::string(": ") + detail : std::string()));
    }

    template <typename T>
    void validate_transfer(T* data, std::int64_t count, std::int64_t peer) const {
        if (data == nullptr || count <= 0 || peer < 0 || peer >= world_size_ ||
            peer == rank_ || static_cast<std::uint64_t>(count) >
                std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("invalid NCCL pipeline transfer");
        }
    }

    NcclUniqueIdV1 bootstrap_id(
        const fs::path& path, std::int64_t timeout_seconds) {
        if (path.empty()) {
            throw std::runtime_error("NCCL bootstrap ID path is empty");
        }
        if (rank_ == 0) {
            if (fs::exists(path)) {
                throw std::runtime_error(
                    "refusing to reuse an existing NCCL bootstrap ID file");
            }
            fs::create_directories(path.parent_path());
            NcclUniqueIdV1 id{};
            check(get_unique_id_(&id), "ncclGetUniqueId");
            const fs::path temporary = path.string() + ".tmp";
            std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
            out.write(id.bytes.data(), static_cast<std::streamsize>(id.bytes.size()));
            if (!out) throw std::runtime_error("failed to write NCCL bootstrap ID");
            out.close();
            fs::rename(temporary, path);
            return id;
        }
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::seconds(timeout_seconds);
        while (std::chrono::steady_clock::now() < deadline) {
            std::error_code error;
            const auto extent = fs::file_size(path, error);
            if (!error && extent == sizeof(NcclUniqueIdV1)) {
                NcclUniqueIdV1 id{};
                std::ifstream in(path, std::ios::binary);
                in.read(id.bytes.data(), static_cast<std::streamsize>(id.bytes.size()));
                if (in && in.peek() == std::char_traits<char>::eof()) return id;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        throw std::runtime_error("timed out waiting for NCCL bootstrap ID");
    }

    Runtime& runtime_;
    std::int64_t world_size_ = 1;
    std::int64_t rank_ = 0;
    void* handle_ = nullptr;
    void* comm_ = nullptr;
    GetUniqueIdFn get_unique_id_ = nullptr;
    CommInitRankFn comm_init_rank_ = nullptr;
    CommDestroyFn comm_destroy_ = nullptr;
    SendFn send_ = nullptr;
    RecvFn recv_ = nullptr;
    AllReduceFn all_reduce_ = nullptr;
    ErrorFn error_string_ = nullptr;
};

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(Runtime& runtime, std::int64_t count) { allocate(runtime, count); }
    ~DeviceBuffer() { reset(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept { *this = std::move(other); }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset(); runtime_ = other.runtime_; data_ = other.data_; count_ = other.count_;
            other.runtime_ = nullptr; other.data_ = nullptr; other.count_ = 0;
        }
        return *this;
    }
    void allocate(Runtime& runtime, std::int64_t count) {
        reset();
        if (count <= 0 || static_cast<std::uint64_t>(count) >
                std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::runtime_error("invalid CUDA buffer size");
        runtime_ = &runtime; count_ = count;
        runtime.check(runtime.malloc(reinterpret_cast<void**>(&data_), bytes()), "cudaMalloc");
    }
    void reset() {
        if (data_ && runtime_) runtime_->free(data_);
        runtime_ = nullptr; data_ = nullptr; count_ = 0;
    }
    T* get() { return data_; }
    const T* get() const { return data_; }
    std::int64_t count() const { return count_; }
    std::size_t bytes() const { return static_cast<std::size_t>(count_) * sizeof(T); }
    void zero() { runtime_->check(runtime_->memset_async(data_, 0, bytes(), runtime_->stream), "cudaMemsetAsync"); }
    void upload(const T* source, std::int64_t count) {
        if (count != count_) throw std::runtime_error("CUDA upload extent mismatch");
        runtime_->check(runtime_->memcpy(data_, source, bytes(), kHostToDevice), "cudaMemcpy H2D");
    }
    void upload_range(const T* source, std::int64_t count, std::int64_t offset) {
        if (source == nullptr || count < 0 || offset < 0 ||
            offset > count_ || count > count_ - offset)
            throw std::runtime_error("CUDA ranged upload extent mismatch");
        runtime_->check(runtime_->memcpy(
            data_ + offset, source, static_cast<std::size_t>(count) * sizeof(T),
            kHostToDevice), "cudaMemcpy ranged H2D");
    }
    void download(T* target, std::int64_t count) const {
        if (count != count_) throw std::runtime_error("CUDA download extent mismatch");
        runtime_->check(runtime_->memcpy(target, data_, bytes(), kDeviceToHost), "cudaMemcpy D2H");
    }
    void download_range(T* target, std::int64_t count, std::int64_t offset) const {
        if (target == nullptr || count < 0 || offset < 0 || offset > count_ ||
            count > count_ - offset) {
            throw std::runtime_error("CUDA ranged download extent mismatch");
        }
        runtime_->check(
            runtime_->memcpy(
                target, data_ + offset,
                static_cast<std::size_t>(count) * sizeof(T), kDeviceToHost),
            "cudaMemcpy ranged D2H");
    }
    void copy_from(const DeviceBuffer& source) {
        if (source.count_ != count_) throw std::runtime_error("CUDA D2D extent mismatch");
        runtime_->check(runtime_->memcpy(data_, source.data_, bytes(), kDeviceToDevice), "cudaMemcpy D2D");
    }
private:
    Runtime* runtime_ = nullptr;
    T* data_ = nullptr;
    std::int64_t count_ = 0;
};

struct ParameterSpec {
    std::string name;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    bool centered = false;
    std::int64_t elements() const { return rows * cols; }
};

std::vector<ParameterSpec> parameter_specs(const Geometry& g) {
    std::vector<ParameterSpec> out;
    out.push_back({"token_embedding.weight", g.vocab, g.dim, false});
    for (std::int64_t layer = 0; layer < g.layers; ++layer) {
        const std::string p = "layers." + std::to_string(layer) + ".";
        out.push_back({p + "input_layernorm.weight", 1, g.dim, true});
        out.push_back({p + "post_attention_layernorm.weight", 1, g.dim, true});
        out.push_back({p + "pre_feedforward_layernorm.weight", 1, g.dim, true});
        out.push_back({p + "post_feedforward_layernorm.weight", 1, g.dim, true});
        out.push_back({p + "q_proj.weight", g.query_width(), g.dim, false});
        out.push_back({p + "k_proj.weight", g.kv_width(), g.dim, false});
        out.push_back({p + "v_proj.weight", g.kv_width(), g.dim, false});
        out.push_back({p + "attn_gate_proj.weight", g.query_width(), g.dim, false});
        out.push_back({p + "o_proj.weight", g.dim, g.query_width(), false});
        out.push_back({p + "gate_proj.weight", g.intermediate, g.dim, false});
        out.push_back({p + "up_proj.weight", g.intermediate, g.dim, false});
        out.push_back({p + "down_proj.weight", g.dim, g.intermediate, false});
    }
    out.push_back({"final_norm.weight", 1, g.dim, false});
    out.push_back({"lm_head.weight", g.vocab, g.dim, false});
    return out;
}

std::int64_t parameter_elements(const Geometry& g) {
    std::int64_t total = 0;
    for (const auto& spec : parameter_specs(g)) {
        if (spec.rows > std::numeric_limits<std::int64_t>::max() / spec.cols ||
            total > std::numeric_limits<std::int64_t>::max() - spec.elements())
            throw std::runtime_error("Muse Glimmer parameter extent overflow");
        total += spec.elements();
    }
    return total;
}

struct PipelinePartition {
    std::int64_t world_size = 1;
    std::int64_t rank = 0;
    std::int64_t layer_begin = 0;
    std::int64_t layer_end = 0;

    PipelinePartition() = default;
    PipelinePartition(const Geometry& geometry, std::int64_t world, std::int64_t stage)
        : world_size(world), rank(stage),
          layer_begin(geometry.layers * stage / world),
          layer_end(geometry.layers * (stage + 1) / world) {
        if (world <= 0 || stage < 0 || stage >= world || world > geometry.layers ||
            layer_begin < 0 || layer_end <= layer_begin || layer_end > geometry.layers) {
            throw std::runtime_error("invalid Muse Glimmer pipeline partition");
        }
    }

    bool distributed() const { return world_size > 1; }
    bool first() const { return rank == 0; }
    bool last() const { return rank + 1 == world_size; }
    std::int64_t local_layers() const { return layer_end - layer_begin; }

    bool owns_parameter_index(
        const Geometry& geometry, std::size_t index) const {
        if (index == 0) return first();
        const std::size_t final_norm_index =
            static_cast<std::size_t>(1 + geometry.layers * 12);
        if (index >= final_norm_index) return last();
        const std::int64_t layer =
            static_cast<std::int64_t>((index - 1) / 12);
        return layer >= layer_begin && layer < layer_end;
    }

    std::int64_t parameter_elements(const Geometry& geometry) const {
        const auto specs = parameter_specs(geometry);
        std::int64_t total = 0;
        for (std::size_t index = 0; index < specs.size(); ++index) {
            if (!owns_parameter_index(geometry, index)) continue;
            if (total > std::numeric_limits<std::int64_t>::max() -
                    specs[index].elements()) {
                throw std::runtime_error("pipeline parameter extent overflows");
            }
            total += specs[index].elements();
        }
        return total;
    }
};

std::uint64_t checked_u64_product(
    std::initializer_list<std::uint64_t> factors, const char* label) {
    std::uint64_t value = 1;
    for (const std::uint64_t factor : factors) {
        if (factor != 0 && value > std::numeric_limits<std::uint64_t>::max() / factor) {
            throw std::runtime_error(std::string(label) + " byte extent overflows");
        }
        value *= factor;
    }
    return value;
}

std::uint64_t checked_u64_sum(
    std::initializer_list<std::uint64_t> values, const char* label) {
    std::uint64_t total = 0;
    for (const std::uint64_t value : values) {
        if (total > std::numeric_limits<std::uint64_t>::max() - value) {
            throw std::runtime_error(std::string(label) + " byte extent overflows");
        }
        total += value;
    }
    return total;
}

struct PipelineMemoryPlan {
    PipelinePartition partition;
    std::uint64_t parameter_bytes = 0;
    std::uint64_t gradient_bytes = 0;
    std::uint64_t optimizer_bytes = 0;
    std::uint64_t activation_bytes = 0;
    std::uint64_t checkpoint_bytes = 0;
    std::uint64_t workspace_bytes = 0;
    std::uint64_t required_bytes = 0;
};

PipelineMemoryPlan pipeline_memory_plan(
    const Geometry& geometry, const Options& options, std::int64_t rank) {
    PipelineMemoryPlan plan;
    plan.partition = PipelinePartition(
        geometry, options.pipeline_parallel_size, rank);
    const auto elements = static_cast<std::uint64_t>(
        plan.partition.parameter_elements(geometry));
    const auto rows = checked_u64_product(
        {static_cast<std::uint64_t>(options.batch_size),
         static_cast<std::uint64_t>(options.sequence_length)},
        "pipeline rows");
    const auto d = static_cast<std::uint64_t>(geometry.dim);
    const auto f = static_cast<std::uint64_t>(geometry.intermediate);
    const auto q = static_cast<std::uint64_t>(geometry.query_width());
    const auto kv = static_cast<std::uint64_t>(geometry.kv_width());
    const auto h = static_cast<std::uint64_t>(geometry.query_heads);
    const auto v = static_cast<std::uint64_t>(geometry.vocab);
    plan.parameter_bytes = checked_u64_product({elements, 2}, "pipeline parameters");
    plan.gradient_bytes = checked_u64_product({elements, 4}, "pipeline gradients");
    plan.optimizer_bytes = checked_u64_product({elements, 8}, "pipeline optimizer");

    // Exact persistent trainer buffers for the AR/SFT pipeline path after
    // stage ownership is applied.  The forward/backward layer scratch is
    // shared across local layers; only activation checkpoints scale with the
    // number of local layer segments.
    const std::uint64_t scalar_rows = checked_u64_product({rows, 16}, "pipeline batch metadata");
    const std::uint64_t common_state = checked_u64_product({rows, d, 6, 4}, "pipeline states");
    const std::uint64_t scratch_width = checked_u64_sum(
        {24 * d, 6 * f, 10 * q, 6 * kv, h}, "pipeline scratch width");
    const std::uint64_t scratch = checked_u64_product(
        {rows, scratch_width, 4}, "pipeline layer scratch");
    std::uint64_t owned = checked_u64_sum(
        {scalar_rows, common_state, scratch}, "pipeline activations");
    if (plan.partition.first()) {
        owned = checked_u64_sum(
            {owned, checked_u64_product({rows, d, 2, 4}, "pipeline embedding state")},
            "pipeline first-stage activations");
    }
    if (plan.partition.last()) {
        owned = checked_u64_sum(
            {owned,
             checked_u64_product({rows, d, 2, 4}, "pipeline head hidden state"),
             checked_u64_product({rows, v, 3, 4}, "pipeline logits"),
             checked_u64_product({rows, 4}, "pipeline loss rows")},
            "pipeline last-stage activations");
    }
    plan.activation_bytes = owned;
    const std::uint64_t checkpoints = static_cast<std::uint64_t>(
        (plan.partition.local_layers() + options.activation_checkpoint_interval - 1) /
            options.activation_checkpoint_interval +
        1 + options.activation_checkpoint_interval + 1);
    plan.checkpoint_bytes = checked_u64_product(
        {checkpoints, rows, d, 4}, "pipeline activation checkpoints");
    const std::uint64_t largest_projection = std::max({d, f, q, v});
    plan.workspace_bytes = std::max<std::uint64_t>(
        512ULL * 1024ULL * 1024ULL,
        checked_u64_product(
            {rows, largest_projection, 4, 2}, "pipeline kernel workspace"));
    plan.required_bytes = checked_u64_sum(
        {plan.parameter_bytes, plan.gradient_bytes, plan.optimizer_bytes,
         plan.activation_bytes, plan.checkpoint_bytes, plan.workspace_bytes},
        "pipeline required memory");
    return plan;
}

void print_pipeline_memory_plan(const Options& options) {
    std::uint64_t maximum = 0;
    std::cout
        << "{\"schema\":\"neuralfn.muse_glimmer_pipeline_plan.v1\","
        << "\"world_size\":" << options.pipeline_parallel_size
        << ",\"batch_size\":" << options.batch_size
        << ",\"sequence_length\":" << options.sequence_length
        << ",\"activation_checkpoint_interval\":"
        << options.activation_checkpoint_interval << ",\"stages\":[";
    for (std::int64_t rank = 0; rank < options.pipeline_parallel_size; ++rank) {
        const auto plan = pipeline_memory_plan(options.geometry, options, rank);
        maximum = std::max(maximum, plan.required_bytes);
        if (rank != 0) std::cout << ',';
        std::cout
            << "{\"rank\":" << rank
            << ",\"layer_begin\":" << plan.partition.layer_begin
            << ",\"layer_end\":" << plan.partition.layer_end
            << ",\"owns_embedding\":" << (plan.partition.first() ? "true" : "false")
            << ",\"owns_final_norm_and_head\":"
            << (plan.partition.last() ? "true" : "false")
            << ",\"parameter_elements\":"
            << plan.partition.parameter_elements(options.geometry)
            << ",\"parameter_bytes\":" << plan.parameter_bytes
            << ",\"gradient_bytes\":" << plan.gradient_bytes
            << ",\"optimizer_bytes\":" << plan.optimizer_bytes
            << ",\"activation_bytes\":" << plan.activation_bytes
            << ",\"checkpoint_bytes\":" << plan.checkpoint_bytes
            << ",\"workspace_bytes\":" << plan.workspace_bytes
            << ",\"required_bytes_before_reserve\":" << plan.required_bytes
            << '}';
    }
    std::cout << "],\"maximum_required_bytes_before_reserve\":" << maximum
              << ",\"reserve_bytes_per_rank\":"
              << options.distributed_reserve_bytes
              << ",\"synchronous_pipeline\":true,"
              << "\"activation_checkpointing\":true,"
              << "\"requires_nccl\":true}\n";
}

std::string sha256_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open file for SHA-256: " + path.string());
    neuralfn::resident_support::Sha256 hash;
    std::vector<std::uint8_t> buffer(4 * 1024 * 1024);
    while (in) {
        in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        const auto count = in.gcount();
        if (count > 0) hash.update(buffer.data(), static_cast<std::size_t>(count));
    }
    if (!in.eof()) throw std::runtime_error("failed while hashing file: " + path.string());
    return hash.finish_hex();
}

bool valid_sha256(std::string_view value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char ch) {
        return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
    });
}

// Minimal strict GGUF-v3 reader for frozen K-Quant adapter training.  The
// resident inference loader has a richer metadata allowlist; training can use
// the full-file SHA of either canonical Meta artifact as that allowlist's
// cryptographic identity, then independently re-check every tensor name,
// shape, encoding, offset, stride, and byte extent below.  No tensor is ever
// expanded into a dense model-sized staging allocation.
enum : std::uint32_t {
    kGgufF32 = 0,
    kGgufQ4K = 12,
    kGgufQ5K = 13,
    kGgufQ6K = 14,
    kGgufBf16 = 30,
};

struct GgufTensorInfo {
    std::string gguf_name;
    std::string parameter_name;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::uint32_t encoding = 0;
    std::int64_t absolute_offset = 0;
    std::int64_t nbytes = 0;
    std::int64_t row_stride_bytes = 0;
};

class GgufReader {
public:
    explicit GgufReader(const fs::path& path) : in_(path, std::ios::binary) {
        if (!in_) throw std::runtime_error("failed to open Muse Glimmer K-Quant GGUF");
    }

    template <typename T>
    T read(const char* label) {
        static_assert(std::is_trivially_copyable_v<T>);
        T value{};
        in_.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!in_) throw std::runtime_error(std::string("truncated GGUF ") + label);
        position_ += sizeof(value);
        return value;
    }

    std::string string(const char* label) {
        const std::uint64_t length = read<std::uint64_t>(label);
        if (length == 0 || length > 32ULL * 1024ULL * 1024ULL ||
            length > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error(std::string("invalid GGUF ") + label + " length");
        }
        std::string value(static_cast<std::size_t>(length), '\0');
        in_.read(value.data(), static_cast<std::streamsize>(value.size()));
        if (!in_ || value.find('\0') != std::string::npos) {
            throw std::runtime_error(std::string("invalid GGUF ") + label);
        }
        position_ += length;
        return value;
    }

    void skip(std::uint64_t count, const char* label) {
        if (count > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
            position_ > std::numeric_limits<std::uint64_t>::max() - count) {
            throw std::runtime_error(std::string("GGUF ") + label + " extent overflows");
        }
        in_.seekg(static_cast<std::streamoff>(count), std::ios::cur);
        if (!in_) throw std::runtime_error(std::string("truncated GGUF ") + label);
        position_ += count;
    }

    std::uint8_t byte(const char* label) { return read<std::uint8_t>(label); }
    std::uint64_t position() const { return position_; }

private:
    std::ifstream in_;
    std::uint64_t position_ = 0;
};

void skip_gguf_value(GgufReader& reader, std::uint32_t type, int depth = 0) {
    if (depth > 1) throw std::runtime_error("nested GGUF arrays are unsupported");
    switch (type) {
        case 0: case 1: case 7: reader.skip(1, "metadata scalar"); return;
        case 2: case 3: reader.skip(2, "metadata scalar"); return;
        case 4: case 5: case 6: reader.skip(4, "metadata scalar"); return;
        case 10: case 11: case 12: reader.skip(8, "metadata scalar"); return;
        case 8: {
            const std::uint64_t length = reader.read<std::uint64_t>("metadata string length");
            if (length > 32ULL * 1024ULL * 1024ULL)
                throw std::runtime_error("GGUF metadata string exceeds 32 MiB");
            reader.skip(length, "metadata string");
            return;
        }
        case 9: {
            const std::uint32_t element = reader.read<std::uint32_t>("array type");
            if (element == 9 || element > 12)
                throw std::runtime_error("GGUF metadata array type is unsupported");
            const std::uint64_t count = reader.read<std::uint64_t>("array length");
            if (count > 1'000'000ULL)
                throw std::runtime_error("GGUF metadata array exceeds 1,000,000 values");
            for (std::uint64_t index = 0; index < count; ++index)
                skip_gguf_value(reader, element, depth + 1);
            return;
        }
        default: throw std::runtime_error("GGUF metadata type is unsupported");
    }
}

struct ExpectedGgufTensor {
    std::string parameter_name;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
};

std::unordered_map<std::string, ExpectedGgufTensor> expected_gguf_tensors(
    const Geometry& g) {
    std::unordered_map<std::string, ExpectedGgufTensor> expected;
    expected.emplace("token_embd.weight",
                     ExpectedGgufTensor{"token_embedding.weight", g.vocab, g.dim});
    expected.emplace("output.weight",
                     ExpectedGgufTensor{"lm_head.weight", g.vocab, g.dim});
    expected.emplace("output_norm.weight",
                     ExpectedGgufTensor{"final_norm.weight", 1, g.dim});
    const std::array<std::tuple<std::string_view, std::string_view,
                                std::int64_t, std::int64_t>, 14> layer_specs{{
        {"attn_norm", "input_layernorm", 1, g.dim},
        {"post_attention_norm", "post_attention_layernorm", 1, g.dim},
        {"ffn_norm", "pre_feedforward_layernorm", 1, g.dim},
        {"post_ffw_norm", "post_feedforward_layernorm", 1, g.dim},
        {"attn_q", "q_proj", g.query_width(), g.dim},
        {"attn_k", "k_proj", g.kv_width(), g.dim},
        {"attn_v", "v_proj", g.kv_width(), g.dim},
        {"attn_gate", "attn_gate_proj", g.query_width(), g.dim},
        {"attn_output", "o_proj", g.dim, g.query_width()},
        {"ffn_gate", "gate_proj", g.intermediate, g.dim},
        {"ffn_up", "up_proj", g.intermediate, g.dim},
        {"ffn_down", "down_proj", g.dim, g.intermediate},
        {"attn_q_norm", "q_norm", 1, g.head_dim},
        {"attn_k_norm", "k_norm", 1, g.head_dim},
    }};
    for (std::int64_t layer = 0; layer < g.layers; ++layer) {
        for (const auto& [gguf_suffix, parameter_suffix, rows, cols] : layer_specs) {
            const std::string prefix = "layers." + std::to_string(layer) + ".";
            expected.emplace(
                "blk." + std::to_string(layer) + "." +
                    std::string(gguf_suffix) + ".weight",
                ExpectedGgufTensor{
                    prefix + std::string(parameter_suffix) + ".weight", rows, cols});
        }
    }
    return expected;
}

class KQuantCheckpoint {
public:
    KQuantCheckpoint(
        const fs::path& path, std::string expected_sha, std::string profile,
        const Geometry& geometry, bool tiny)
        : path_(path), sha256_(std::move(expected_sha)), profile_(std::move(profile)) {
        if (!valid_sha256(sha256_))
            throw std::runtime_error("K-Quant checkpoint SHA-256 is malformed");
        if (!fs::is_regular_file(path_))
            throw std::runtime_error("K-Quant checkpoint is not a regular file");
        file_nbytes_ = static_cast<std::int64_t>(fs::file_size(path_));
        last_write_time_ = fs::last_write_time(path_);
        if (file_nbytes_ <= 0 || sha256_file(path_) != sha256_)
            throw std::runtime_error("K-Quant checkpoint authentication failed");
        if (!tiny) validate_official_identity();
        parse(geometry, tiny);
        if (static_cast<std::int64_t>(fs::file_size(path_)) != file_nbytes_ ||
            fs::last_write_time(path_) != last_write_time_) {
            throw std::runtime_error("K-Quant checkpoint changed during inspection");
        }
    }

    const GgufTensorInfo& tensor(std::string_view parameter_name) const {
        const auto found = tensors_.find(std::string(parameter_name));
        if (found == tensors_.end())
            throw std::runtime_error(
                "K-Quant GGUF is missing native tensor " + std::string(parameter_name));
        return found->second;
    }
    const fs::path& path() const { return path_; }
    const std::string& sha256() const { return sha256_; }
    const std::string& profile() const { return profile_; }
    std::int64_t file_nbytes() const { return file_nbytes_; }
    fs::file_time_type last_write_time() const { return last_write_time_; }

private:
    void validate_official_identity() const {
        struct Identity { std::int64_t nbytes; const char* sha; };
        const Identity identity = profile_ == "k-quant-17gb"
            ? Identity{16'756'683'904LL,
                "4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e"}
            : profile_ == "k-quant-dynamic"
                ? Identity{19'653'960'832LL,
                    "ac7023d6a4c704eb9af54ab53e476a66b7f5b6c0ef2fc4a8dde5253c291a6c38"}
                : throw std::runtime_error("unsupported production K-Quant profile");
        if (file_nbytes_ != identity.nbytes || sha256_ != identity.sha) {
            throw std::runtime_error(
                "K-Quant profile does not match Meta's canonical artifact size/SHA-256");
        }
    }

    void parse(const Geometry& geometry, bool tiny) {
        GgufReader reader(path_);
        const std::array<char, 4> magic{
            static_cast<char>(reader.byte("magic")),
            static_cast<char>(reader.byte("magic")),
            static_cast<char>(reader.byte("magic")),
            static_cast<char>(reader.byte("magic"))};
        if (magic != std::array<char, 4>{'G', 'G', 'U', 'F'} ||
            reader.read<std::uint32_t>("version") != 3) {
            throw std::runtime_error("K-Quant checkpoint is not GGUF v3");
        }
        const std::uint64_t tensor_count = reader.read<std::uint64_t>("tensor count");
        const std::uint64_t metadata_count = reader.read<std::uint64_t>("metadata count");
        if (metadata_count > 4096)
            throw std::runtime_error("GGUF metadata count is unreasonable");
        for (std::uint64_t index = 0; index < metadata_count; ++index) {
            (void)reader.string("metadata key");
            skip_gguf_value(reader, reader.read<std::uint32_t>("metadata type"));
            if (reader.position() > 64ULL * 1024ULL * 1024ULL)
                throw std::runtime_error("GGUF metadata exceeds the 64 MiB bound");
        }

        const auto expected = expected_gguf_tensors(geometry);
        if (tensor_count != expected.size())
            throw std::runtime_error("K-Quant GGUF tensor count does not match the architecture");
        struct RawTensor {
            std::string name;
            std::vector<std::uint64_t> dimensions;
            std::uint32_t encoding = 0;
            std::uint64_t relative_offset = 0;
        };
        std::vector<RawTensor> raw;
        raw.reserve(static_cast<std::size_t>(tensor_count));
        std::unordered_set<std::string> names;
        for (std::uint64_t index = 0; index < tensor_count; ++index) {
            RawTensor tensor;
            tensor.name = reader.string("tensor name");
            if (!names.insert(tensor.name).second)
                throw std::runtime_error("K-Quant GGUF contains duplicate tensor names");
            const std::uint32_t rank = reader.read<std::uint32_t>("tensor rank");
            if (rank < 1 || rank > 2)
                throw std::runtime_error("K-Quant GGUF tensor rank is unsupported");
            for (std::uint32_t dimension = 0; dimension < rank; ++dimension) {
                const auto value = reader.read<std::uint64_t>("tensor dimension");
                if (value == 0 || value > static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max()))
                    throw std::runtime_error("K-Quant GGUF tensor dimension is invalid");
                tensor.dimensions.push_back(value);
            }
            tensor.encoding = reader.read<std::uint32_t>("tensor encoding");
            tensor.relative_offset = reader.read<std::uint64_t>("tensor offset");
            raw.push_back(std::move(tensor));
        }
        if (reader.position() > 64ULL * 1024ULL * 1024ULL)
            throw std::runtime_error("GGUF tensor table exceeds the 64 MiB bound");
        const std::uint64_t data_offset = (reader.position() + 31ULL) & ~31ULL;
        while (reader.position() < data_offset) {
            if (reader.byte("header padding") != 0)
                throw std::runtime_error("K-Quant GGUF header padding is nonzero");
        }

        struct Extent { std::uint64_t offset; std::uint64_t nbytes; };
        std::vector<Extent> extents;
        std::unordered_map<std::uint32_t, std::int64_t> inventory;
        for (const auto& tensor : raw) {
            const auto contract = expected.find(tensor.name);
            if (contract == expected.end())
                throw std::runtime_error("K-Quant GGUF tensor allowlist mismatch");
            const auto& shape = contract->second;
            if (tensor.dimensions[0] != static_cast<std::uint64_t>(shape.cols) ||
                (shape.rows == 1
                    ? tensor.dimensions.size() != 1
                    : tensor.dimensions.size() != 2 ||
                      tensor.dimensions[1] != static_cast<std::uint64_t>(shape.rows))) {
                throw std::runtime_error(
                    "K-Quant GGUF tensor shape mismatch at " + tensor.name);
            }
            std::int64_t block_elements = 0;
            std::int64_t block_bytes = 0;
            switch (tensor.encoding) {
                case kGgufF32: block_elements = 1; block_bytes = 4; break;
                case kGgufQ4K: block_elements = 256; block_bytes = 144; break;
                case kGgufQ5K: block_elements = 256; block_bytes = 176; break;
                case kGgufQ6K: block_elements = 256; block_bytes = 210; break;
                case kGgufBf16: block_elements = 1; block_bytes = 2; break;
                default: throw std::runtime_error("K-Quant GGUF tensor encoding is unsupported");
            }
            if (shape.cols % block_elements != 0 || tensor.relative_offset % 32 != 0)
                throw std::runtime_error("K-Quant GGUF block/alignment contract failed");
            const std::int64_t row_stride =
                (shape.cols / block_elements) * block_bytes;
            if (shape.rows > std::numeric_limits<std::int64_t>::max() / row_stride)
                throw std::runtime_error("K-Quant GGUF tensor byte extent overflows");
            const std::int64_t nbytes = shape.rows * row_stride;
            if (tensor.relative_offset > static_cast<std::uint64_t>(file_nbytes_) ||
                data_offset > static_cast<std::uint64_t>(file_nbytes_) ||
                tensor.relative_offset > static_cast<std::uint64_t>(file_nbytes_) - data_offset ||
                static_cast<std::uint64_t>(nbytes) >
                    static_cast<std::uint64_t>(file_nbytes_) - data_offset - tensor.relative_offset) {
                throw std::runtime_error("K-Quant GGUF tensor exceeds the file extent");
            }
            GgufTensorInfo info{
                .gguf_name = tensor.name,
                .parameter_name = shape.parameter_name,
                .rows = shape.rows,
                .cols = shape.cols,
                .encoding = tensor.encoding,
                .absolute_offset = static_cast<std::int64_t>(data_offset + tensor.relative_offset),
                .nbytes = nbytes,
                .row_stride_bytes = row_stride,
            };
            if (!tensors_.emplace(info.parameter_name, std::move(info)).second)
                throw std::runtime_error("K-Quant native tensor mapping is duplicated");
            extents.push_back({tensor.relative_offset, static_cast<std::uint64_t>(nbytes)});
            ++inventory[tensor.encoding];
        }
        if (tensors_.size() != expected.size())
            throw std::runtime_error("K-Quant GGUF tensor mapping is incomplete");
        std::sort(extents.begin(), extents.end(), [](const auto& left, const auto& right) {
            return left.offset < right.offset;
        });
        std::uint64_t expected_offset = 0;
        for (const auto& extent : extents) {
            if (extent.offset != expected_offset)
                throw std::runtime_error("K-Quant GGUF tensors are not canonical/contiguous");
            expected_offset = (extent.offset + extent.nbytes + 31ULL) & ~31ULL;
        }
        const auto& last = extents.back();
        if (data_offset + last.offset + last.nbytes !=
            static_cast<std::uint64_t>(file_nbytes_)) {
            throw std::runtime_error("K-Quant GGUF has trailing or missing tensor bytes");
        }
        if (!tiny) {
            const bool profile17 = inventory[kGgufF32] == 313 &&
                inventory[kGgufQ4K] == 365 && inventory[kGgufQ5K] == 1 &&
                inventory[kGgufQ6K] == 52 && inventory[kGgufBf16] == 0;
            const bool dynamic = inventory[kGgufF32] == 313 &&
                inventory[kGgufQ4K] == 51 && inventory[kGgufQ5K] == 130 &&
                inventory[kGgufQ6K] == 237 && inventory[kGgufBf16] == 0;
            if ((profile_ == "k-quant-17gb" && !profile17) ||
                (profile_ == "k-quant-dynamic" && !dynamic)) {
                throw std::runtime_error("K-Quant GGUF encoding inventory/profile mismatch");
            }
        } else if (inventory[kGgufQ4K] == 0 || inventory[kGgufQ5K] == 0 ||
                   inventory[kGgufQ6K] == 0) {
            throw std::runtime_error(
                "tiny K-Quant fixture must exercise Q4_K, Q5_K, and Q6_K");
        }
    }

    fs::path path_;
    std::string sha256_;
    std::string profile_;
    std::int64_t file_nbytes_ = 0;
    fs::file_time_type last_write_time_{};
    std::unordered_map<std::string, GgufTensorInfo> tensors_;
};

std::shared_ptr<const KQuantCheckpoint> load_kquant_checkpoint(
    const Options& options) {
    if (options.kquant_profile.empty()) return nullptr;
    return std::make_shared<const KQuantCheckpoint>(
        fs::path(options.checkpoint), options.checkpoint_sha256,
        options.kquant_profile, options.geometry, options.tiny_geometry);
}

struct TileOps {
    using AbiFn = int (*)();
    using PackedLinearFn = int (*)(const NfnNativeTilePackedWeightDescriptorV1*, const float*, const float*, float*, std::int64_t, bool);
    using PackedLinearBackwardInputFn = int (*)(const NfnNativeTilePackedWeightDescriptorV1*, const float*, float*, std::int64_t);
    using EmbeddingFn = int (*)(const NfnNativeTilePackedWeightDescriptorV1*, const std::int32_t*, float*, std::int64_t);
    using NormFn = int (*)(const float*, const NfnNativeTilePackedWeightDescriptorV1*, float*, std::int64_t, std::int64_t, float, bool, void*);
    using NormBackwardFn = int (*)(const NfnNativeTileGlimmerRmsNormBackwardDescriptorV1*);
    using RopeBatchFn = int (*)(float*, float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t, float, std::uint32_t, bool, void*);
    using AttentionFn = int (*)(const NfnNativeTileGlimmerAttentionTrainingDescriptorV1*);
    using GateFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
    using GateBackwardFn = int (*)(const float*, const float*, const float*, float*, float*, std::int64_t, void*);
    using SwiGluFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
    using SwiGluBackwardFn = int (*)(const float*, const float*, const float*, float*, float*, std::int64_t, void*);
    using AddFn = int (*)(const float*, const float*, float*, std::int64_t, void*);
    using ScaleFn = int (*)(float*, std::int64_t, float, void*);
    using CopyFn = int (*)(const float*, float*, std::int64_t, void*);
    using DropoutFn = int (*)(const float*, float*, std::int64_t, float, std::int64_t, void*);
    using LinearBackwardWeightFn = int (*)(const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using LogitFn = int (*)(float*, std::int64_t, float, float, void*);
    using LogitBackwardFn = int (*)(const float*, const float*, float*, std::int64_t, float, float, void*);
    using CeFn = int (*)(const NfnNativeTileGlimmerMaskedCeDescriptorV1*);
    using SequenceLogpFn = int (*)(const NfnNativeTileSequenceLogpDescriptorV1*);
    using DpoFn = int (*)(const NfnNativeTileDpoPairwiseDescriptorV1*);
    using RewardHeadFn = int (*)(const NfnNativeTileMaskedRewardHeadDescriptorV1*);
    using PreferenceBceFn = int (*)(const NfnNativeTilePreferenceBceDescriptorV1*);
    using TokenLogpEntropyFn = int (*)(const NfnNativeTileTokenLogpEntropyDescriptorV1*);
    using MaskedPpoFn = int (*)(const NfnNativeTileMaskedPpoLossDescriptorV1*);
    using EmbeddingBackwardFn = int (*)(const std::int32_t*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, void*);
    using AdamFn = int (*)(std::uint16_t*, const float*, float*, float*, std::int64_t, float, float, float, float, float, std::int64_t, float, void*);
    using ErrorFn = const char* (*)(int);

    void* handle = nullptr;
    PackedLinearFn linear = nullptr;
    PackedLinearBackwardInputFn linear_backward_input = nullptr;
    EmbeddingFn embedding = nullptr;
    NormFn norm = nullptr;
    NormBackwardFn norm_backward = nullptr;
    RopeBatchFn rope = nullptr;
    AttentionFn attention_forward = nullptr;
    AttentionFn attention_backward = nullptr;
    GateFn gate = nullptr;
    GateBackwardFn gate_backward = nullptr;
    SwiGluFn swiglu = nullptr;
    SwiGluBackwardFn swiglu_backward = nullptr;
    AddFn add = nullptr;
    ScaleFn scale = nullptr;
    CopyFn copy = nullptr;
    DropoutFn dropout_forward = nullptr;
    DropoutFn dropout_backward = nullptr;
    LinearBackwardWeightFn linear_backward_weight = nullptr;
    LogitFn logit = nullptr;
    LogitBackwardFn logit_backward = nullptr;
    CeFn ce = nullptr;
    SequenceLogpFn sequence_logp_forward = nullptr;
    SequenceLogpFn sequence_logp_backward = nullptr;
    DpoFn dpo_forward = nullptr;
    DpoFn dpo_backward = nullptr;
    RewardHeadFn reward_head_forward = nullptr;
    RewardHeadFn reward_head_backward = nullptr;
    PreferenceBceFn preference_bce_forward = nullptr;
    PreferenceBceFn preference_bce_backward = nullptr;
    TokenLogpEntropyFn token_logp_entropy_forward = nullptr;
    TokenLogpEntropyFn token_logp_entropy_backward = nullptr;
    MaskedPpoFn masked_ppo_forward = nullptr;
    MaskedPpoFn masked_ppo_backward = nullptr;
    EmbeddingBackwardFn embedding_backward = nullptr;
    AdamFn adam = nullptr;
    ErrorFn error_string = nullptr;

    explicit TileOps(const std::string& path) {
        const std::string resolved = !path.empty() ? path :
            (std::getenv("NFN_NATIVE_TRAIN_TILE_OPS_LIB")
                ? std::getenv("NFN_NATIVE_TRAIN_TILE_OPS_LIB")
                : "libnfn_native_train_tile_ops.so");
        handle = dlopen(resolved.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) throw std::runtime_error("failed to load Tile-CUDA ops: " + resolved);
        const auto abi = symbol<AbiFn>(handle, "nfn_native_tile_glimmer_training_abi_version");
        if (abi() != NFN_NATIVE_TILE_GLIMMER_TRAINING_V1)
            throw std::runtime_error("Tile-CUDA Glimmer training ABI mismatch");
        linear = symbol<PackedLinearFn>(handle, "nfn_native_tile_linear_packed_weight_float32_v1");
        linear_backward_input = symbol<PackedLinearBackwardInputFn>(handle, "nfn_native_tile_linear_backward_input_packed_weight_float32_v1");
        embedding = symbol<EmbeddingFn>(handle, "nfn_native_tile_glimmer_embedding_batch_i32_float32_v1");
        norm = symbol<NormFn>(handle, "nfn_native_tile_glimmer_rms_norm_affine_float32_v1");
        norm_backward = symbol<NormBackwardFn>(handle, "nfn_native_tile_glimmer_rms_norm_backward_float32_v1");
        rope = symbol<RopeBatchFn>(handle, "nfn_native_tile_glimmer_positioned_rope_batch_float32_v1");
        attention_forward = symbol<AttentionFn>(handle, "nfn_native_tile_glimmer_attention_forward_float32_v1");
        attention_backward = symbol<AttentionFn>(handle, "nfn_native_tile_glimmer_attention_backward_float32_v1");
        gate = symbol<GateFn>(handle, "nfn_native_tile_glimmer_sigmoid_gate_float32_v1");
        gate_backward = symbol<GateBackwardFn>(handle, "nfn_native_tile_glimmer_sigmoid_gate_backward_float32_v1");
        swiglu = symbol<SwiGluFn>(handle, "nfn_native_tile_swiglu_float32");
        swiglu_backward = symbol<SwiGluBackwardFn>(handle, "nfn_native_tile_swiglu_backward_float32");
        add = symbol<AddFn>(handle, "nfn_native_tile_add_float32");
        scale = symbol<ScaleFn>(handle, "nfn_native_tile_scale_inplace_float32");
        copy = symbol<CopyFn>(handle, "nfn_native_tile_copy_float32");
        dropout_forward = symbol<DropoutFn>(handle, "nfn_native_tile_dropout_forward_float32");
        dropout_backward = symbol<DropoutFn>(handle, "nfn_native_tile_dropout_backward_float32");
        linear_backward_weight = symbol<LinearBackwardWeightFn>(handle, "nfn_native_tile_linear_backward_weight_float32");
        logit = symbol<LogitFn>(handle, "nfn_native_tile_glimmer_logit_transform_float32_v1");
        logit_backward = symbol<LogitBackwardFn>(handle, "nfn_native_tile_glimmer_logit_transform_backward_float32_v1");
        ce = symbol<CeFn>(handle, "nfn_native_tile_glimmer_masked_cross_entropy_i32_float32_v1");
        sequence_logp_forward = symbol<SequenceLogpFn>(
            handle, "nfn_native_tile_sequence_logp_i32_float32_forward_v1");
        sequence_logp_backward = symbol<SequenceLogpFn>(
            handle, "nfn_native_tile_sequence_logp_i32_float32_backward_v1");
        dpo_forward = symbol<DpoFn>(
            handle, "nfn_native_tile_dpo_pairwise_loss_float32_forward_v1");
        dpo_backward = symbol<DpoFn>(
            handle, "nfn_native_tile_dpo_pairwise_loss_float32_backward_v1");
        reward_head_forward = symbol<RewardHeadFn>(
            handle, "nfn_native_tile_masked_reward_head_float32_forward_v1");
        reward_head_backward = symbol<RewardHeadFn>(
            handle, "nfn_native_tile_masked_reward_head_float32_backward_v1");
        preference_bce_forward = symbol<PreferenceBceFn>(
            handle, "nfn_native_tile_preference_bce_loss_float32_forward_v1");
        preference_bce_backward = symbol<PreferenceBceFn>(
            handle, "nfn_native_tile_preference_bce_loss_float32_backward_v1");
        token_logp_entropy_forward = symbol<TokenLogpEntropyFn>(
            handle,
            "nfn_native_tile_token_logp_entropy_i32_float32_forward_v1");
        token_logp_entropy_backward = symbol<TokenLogpEntropyFn>(
            handle,
            "nfn_native_tile_token_logp_entropy_i32_float32_backward_v1");
        masked_ppo_forward = symbol<MaskedPpoFn>(
            handle, "nfn_native_tile_masked_ppo_loss_float32_forward_v1");
        masked_ppo_backward = symbol<MaskedPpoFn>(
            handle, "nfn_native_tile_masked_ppo_loss_float32_backward_v1");
        embedding_backward = symbol<EmbeddingBackwardFn>(handle, "nfn_native_tile_token_embedding_backward_weight_i32_float32");
        adam = symbol<AdamFn>(handle, "nfn_native_tile_glimmer_adamw_bf16_float32_v1");
        error_string = symbol<ErrorFn>(handle, "nfn_native_tile_ops_error_string");
    }
    ~TileOps() { if (handle) dlclose(handle); }
    void check(int status, const char* operation) const {
        if (status != 0) {
            const char* detail = error_string ? error_string(status) : nullptr;
            throw std::runtime_error(std::string(operation) + " failed" +
                (detail ? std::string(": ") + detail : std::string()));
        }
    }
};

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return static_cast<std::uint16_t>(bits >> 16u);
}

float bf16_to_float(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

constexpr std::array<float, 16> kNf4Codebook{
    -1.0f, -0.6961928009986877f, -0.5250730514526367f,
    -0.39491748809814453f, -0.28444138169288635f,
    -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f,
    0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f,
    0.7229568362236023f, 1.0f};

std::int64_t nf4_row_stride(std::int64_t columns) {
    if (columns <= 0 ||
        (columns + 63) / 64 > std::numeric_limits<std::int64_t>::max() / 36)
        throw std::runtime_error("invalid NF4 base width");
    return ((columns + 63) / 64) * 36;
}

void quantize_nf4_rows(
    const std::uint16_t* source, std::int64_t rows, std::int64_t columns,
    std::uint8_t* output) {
    if (source == nullptr || output == nullptr || rows <= 0)
        throw std::runtime_error("invalid NF4 quantization input");
    const std::int64_t row_stride = nf4_row_stride(columns);
    for (std::int64_t row = 0; row < rows; ++row) {
        const std::uint16_t* source_row = source + row * columns;
        std::uint8_t* output_row = output + row * row_stride;
        const std::int64_t groups = (columns + 63) / 64;
        for (std::int64_t group_index = 0; group_index < groups; ++group_index) {
            const std::uint16_t* group = source_row + group_index * 64;
            std::uint8_t* packed = output_row + group_index * 36;
            const std::int64_t group_elements = std::min<std::int64_t>(
                64, columns - group_index * 64);
            float scale = 1.0e-8f;
            for (std::int64_t index = 0; index < group_elements; ++index)
                scale = std::max(scale, std::abs(bf16_to_float(group[index])));
            std::memcpy(packed, &scale, sizeof(scale));
            std::memset(packed + 4, 0x77, 32);
            for (std::int64_t index = 0; index < group_elements; ++index) {
                const float normalized = bf16_to_float(group[index]) / scale;
                std::uint8_t best = 0;
                float best_distance = std::numeric_limits<float>::infinity();
                for (std::uint8_t code = 0; code < kNf4Codebook.size(); ++code) {
                    const float distance = std::abs(normalized - kNf4Codebook[code]);
                    if (distance < best_distance) {
                        best_distance = distance;
                        best = code;
                    }
                }
                if ((index & 1) == 0) packed[4 + index / 2] = best;
                else packed[4 + index / 2] |= static_cast<std::uint8_t>(best << 4U);
            }
        }
    }
}

struct Parameter {
    ParameterSpec spec;
    bool trainable = true;
    bool nf4 = false;
    bool kquant = false;
    std::uint32_t packed_encoding = NFN_NATIVE_TILE_PACKED_WEIGHT_BF16;
    std::int64_t packed_row_stride_bytes = 0;
    std::int64_t packed_file_offset = 0;
    DeviceBuffer<std::uint16_t> value;
    DeviceBuffer<std::uint8_t> packed_value;
    DeviceBuffer<float> gradient;
    DeviceBuffer<float> exp_avg;
    DeviceBuffer<float> exp_avg_sq;

    Parameter(
        Runtime& runtime, ParameterSpec input, bool should_train = true,
        bool quantize_nf4 = false, const GgufTensorInfo* packed = nullptr)
        : spec(std::move(input)), trainable(should_train), nf4(quantize_nf4),
          kquant(packed != nullptr) {
        if (kquant) {
            if (trainable || nf4 || packed->rows != spec.rows ||
                packed->cols != spec.cols || packed->nbytes <= 0) {
                throw std::runtime_error("invalid immutable K-Quant parameter contract");
            }
            packed_encoding = packed->encoding;
            packed_row_stride_bytes = packed->row_stride_bytes;
            packed_file_offset = packed->absolute_offset;
            packed_value.allocate(runtime, packed->nbytes);
        } else if (nf4) {
            if (trainable)
                throw std::runtime_error("NF4 base parameters must be immutable");
            const std::int64_t stride = nf4_row_stride(spec.cols);
            if (spec.rows > std::numeric_limits<std::int64_t>::max() / stride)
                throw std::runtime_error("NF4 parameter extent overflow");
            packed_value.allocate(runtime, spec.rows * stride);
        } else {
            value.allocate(runtime, spec.elements());
        }
        if (trainable) {
            gradient.allocate(runtime, spec.elements());
            exp_avg.allocate(runtime, spec.elements());
            exp_avg_sq.allocate(runtime, spec.elements());
            gradient.zero(); exp_avg.zero(); exp_avg_sq.zero();
        }
    }

    NfnNativeTilePackedWeightDescriptorV1 descriptor(void* stream) const {
        const std::int64_t row_stride = kquant
            ? packed_row_stride_bytes
            : nf4 ? nf4_row_stride(spec.cols) : spec.cols * 2;
        return NfnNativeTilePackedWeightDescriptorV1{
            .struct_size = sizeof(NfnNativeTilePackedWeightDescriptorV1),
            .version = NFN_NATIVE_TILE_PACKED_WEIGHT_V1,
            .encoding = kquant ? packed_encoding
                : nf4 ? NFN_NATIVE_TILE_PACKED_WEIGHT_NF4_GROUP64
                      : NFN_NATIVE_TILE_PACKED_WEIGHT_BF16,
            .flags = 0,
            .data = (nf4 || kquant) ? packed_value.get()
                        : reinterpret_cast<const std::uint8_t*>(value.get()),
            .data_nbytes = (nf4 || kquant)
                               ? static_cast<std::int64_t>(packed_value.bytes())
                               : static_cast<std::int64_t>(value.bytes()),
            .output_dim = spec.rows,
            .input_dim = spec.cols,
            .row_stride_bytes = row_stride,
            .reserved0 = 0,
            .reserved1 = 0,
            .cuda_stream = stream,
        };
    }
};

class Parameters {
public:
    Parameters(
        Runtime& runtime, const Geometry& geometry, bool trainable = true,
        bool quantize_nf4 = false, bool train_lm_head = true,
        std::shared_ptr<const KQuantCheckpoint> kquant = nullptr,
        const PipelinePartition* partition = nullptr)
        : runtime_(runtime), geometry_(geometry), kquant_(std::move(kquant)),
          partition_(partition == nullptr
                         ? PipelinePartition(geometry, 1, 0)
                         : *partition) {
        if (kquant_ && (trainable || quantize_nf4)) {
            throw std::runtime_error(
                "K-Quant parameters are immutable and cannot use the NF4 conversion path");
        }
        const auto specs = parameter_specs(geometry);
        for (std::size_t index = 0; index < specs.size(); ++index) {
            auto spec = specs[index];
            if (!partition_.owns_parameter_index(geometry, index)) {
                values_.push_back(nullptr);
                continue;
            }
            const bool packed = quantize_nf4 && spec.rows > 1;
            const bool should_train =
                trainable && (train_lm_head || spec.name != "lm_head.weight");
            const GgufTensorInfo* packed_tensor =
                kquant_ ? &kquant_->tensor(spec.name) : nullptr;
            values_.push_back(std::make_unique<Parameter>(
                runtime, std::move(spec), should_train, packed, packed_tensor));
        }
        if (kquant_) {
            for (std::int64_t layer = 0; layer < geometry_.layers; ++layer) {
                for (std::string_view suffix : {"q_norm", "k_norm"}) {
                    ParameterSpec spec{
                        "layers." + std::to_string(layer) + "." +
                            std::string(suffix) + ".weight",
                        1, geometry_.head_dim, false};
                    const auto& tensor = kquant_->tensor(spec.name);
                    auxiliary_.push_back(std::make_unique<Parameter>(
                        runtime, std::move(spec), false, false, &tensor));
                }
            }
        }
    }

    Parameter& embedding() { return *values_.at(0); }
    Parameter& layer(std::int64_t index, std::int64_t slot) {
        return *values_.at(static_cast<std::size_t>(1 + index * 12 + slot));
    }
    Parameter& final_norm() { return *values_.at(static_cast<std::size_t>(1 + geometry_.layers * 12)); }
    Parameter& lm_head() { return *values_.at(static_cast<std::size_t>(2 + geometry_.layers * 12)); }
    Parameter* q_norm(std::int64_t layer) {
        return kquant_ ? auxiliary_.at(static_cast<std::size_t>(layer * 2)).get()
                       : nullptr;
    }
    Parameter* k_norm(std::int64_t layer) {
        return kquant_ ? auxiliary_.at(static_cast<std::size_t>(layer * 2 + 1)).get()
                       : nullptr;
    }
    bool is_kquant() const { return static_cast<bool>(kquant_); }
    bool distributed() const { return partition_.distributed(); }
    const PipelinePartition& partition() const { return partition_; }
    bool centered_layer_norms() const { return !is_kquant(); }
    float query_scale(float dense_scale) const { return is_kquant() ? 1.0f : dense_scale; }
    std::uint32_t rope_layout() const {
        return is_kquant() ? NFN_NATIVE_TILE_GLIMMER_ROPE_INTERLEAVED
                           : NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT;
    }
    const std::vector<std::unique_ptr<Parameter>>& all() const { return values_; }

    void load(
        const fs::path& path, std::string_view expected_sha,
        bool verify_full_digest = true) {
        if (!valid_sha256(expected_sha)) throw std::runtime_error("checkpoint SHA-256 must be lowercase hexadecimal");
        if (kquant_) {
            if (path != kquant_->path() || expected_sha != kquant_->sha256() ||
                !fs::is_regular_file(path) ||
                static_cast<std::int64_t>(fs::file_size(path)) !=
                    kquant_->file_nbytes() ||
                fs::last_write_time(path) != kquant_->last_write_time()) {
                throw std::runtime_error(
                    "K-Quant checkpoint changed after its strict GGUF inspection");
            }
            std::ifstream in(path, std::ios::binary);
            constexpr std::int64_t kChunkBytes = 8 * 1024 * 1024;
            std::vector<std::uint8_t> host;
            const auto upload = [&](Parameter& parameter) {
                if (!parameter.kquant)
                    throw std::runtime_error("mixed K-Quant parameter storage is invalid");
                std::int64_t copied = 0;
                while (copied < static_cast<std::int64_t>(parameter.packed_value.bytes())) {
                    const std::int64_t count = std::min<std::int64_t>(
                        kChunkBytes,
                        static_cast<std::int64_t>(parameter.packed_value.bytes()) - copied);
                    host.resize(static_cast<std::size_t>(count));
                    in.clear();
                    in.seekg(parameter.packed_file_offset + copied, std::ios::beg);
                    in.read(reinterpret_cast<char*>(host.data()), count);
                    if (!in) throw std::runtime_error("short K-Quant tensor read");
                    parameter.packed_value.upload_range(host.data(), count, copied);
                    copied += count;
                }
            };
            for (auto& parameter : values_) {
                if (parameter) upload(*parameter);
            }
            for (auto& parameter : auxiliary_) upload(*parameter);
            if (static_cast<std::int64_t>(fs::file_size(path)) !=
                    kquant_->file_nbytes() ||
                fs::last_write_time(path) != kquant_->last_write_time()) {
                throw std::runtime_error("K-Quant checkpoint changed while uploading tensors");
            }
            return;
        }
        const std::uintmax_t expected_bytes = static_cast<std::uintmax_t>(parameter_elements(geometry_)) * 2U;
        if (!fs::is_regular_file(path) || fs::file_size(path) != expected_bytes)
            throw std::runtime_error("Muse Glimmer BF16 checkpoint has the wrong byte extent");
        if (verify_full_digest && sha256_file(path) != expected_sha)
            throw std::runtime_error("Muse Glimmer BF16 checkpoint SHA-256 mismatch");
        const auto source_size = fs::file_size(path);
        const auto source_time = fs::last_write_time(path);
        std::ifstream in(path, std::ios::binary);
        constexpr std::int64_t kHostChunkElements = 2 * 1024 * 1024;
        std::vector<std::uint16_t> host;
        std::vector<std::uint8_t> packed;
        const auto specs = parameter_specs(geometry_);
        for (std::size_t parameter_index = 0;
             parameter_index < values_.size(); ++parameter_index) {
            auto& parameter = values_[parameter_index];
            if (!parameter) {
                const auto bytes = static_cast<std::streamoff>(
                    specs[parameter_index].elements() * 2);
                in.seekg(bytes, std::ios::cur);
                if (!in) {
                    throw std::runtime_error(
                        "short Muse Glimmer BF16 checkpoint skip");
                }
                continue;
            }
            const std::int64_t rows_per_chunk = std::max<std::int64_t>(
                1, kHostChunkElements / parameter->spec.cols);
            for (std::int64_t row = 0; row < parameter->spec.rows; row += rows_per_chunk) {
                const std::int64_t chunk_rows = std::min(
                    rows_per_chunk, parameter->spec.rows - row);
                const std::int64_t chunk_elements = chunk_rows * parameter->spec.cols;
                host.resize(static_cast<std::size_t>(chunk_elements));
                in.read(reinterpret_cast<char*>(host.data()),
                        static_cast<std::streamsize>(host.size() * 2));
                if (!in) throw std::runtime_error("short Muse Glimmer BF16 checkpoint read");
                if (parameter->nf4) {
                    const std::int64_t stride = nf4_row_stride(parameter->spec.cols);
                    packed.resize(static_cast<std::size_t>(chunk_rows * stride));
                    quantize_nf4_rows(
                        host.data(), chunk_rows, parameter->spec.cols, packed.data());
                    parameter->packed_value.upload_range(
                        packed.data(), static_cast<std::int64_t>(packed.size()), row * stride);
                } else {
                    parameter->value.upload_range(
                        host.data(), chunk_elements, row * parameter->spec.cols);
                }
            }
        }
        if (in.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("Muse Glimmer BF16 checkpoint has trailing bytes");
        if (fs::file_size(path) != source_size ||
            fs::last_write_time(path) != source_time) {
            throw std::runtime_error(
                "Muse Glimmer BF16 checkpoint changed while streaming a partition");
        }
    }

    void zero_grad() {
        for (auto& parameter : values_)
            if (parameter && parameter->trainable) parameter->gradient.zero();
    }

    double gradient_norm() const {
        double sum = 0.0;
        std::vector<float> host;
        for (const auto& parameter : values_) {
            if (!parameter || !parameter->trainable) continue;
            host.resize(static_cast<std::size_t>(parameter->spec.elements()));
            parameter->gradient.download(host.data(), parameter->spec.elements());
            for (float value : host) sum += static_cast<double>(value) * value;
        }
        return std::sqrt(sum);
    }

    void step(TileOps& ops, std::int64_t step, const Options& options, float gradient_scale) {
        for (auto& parameter : values_) {
            if (!parameter || !parameter->trainable) continue;
            ops.check(ops.adam(
                parameter->value.get(), parameter->gradient.get(),
                parameter->exp_avg.get(), parameter->exp_avg_sq.get(),
                parameter->spec.elements(), options.learning_rate, options.beta1,
                options.beta2, options.adam_eps, options.weight_decay, step,
                gradient_scale, runtime_.stream), "Glimmer AdamW");
        }
    }

    void save_model(const fs::path& path) const {
        if (distributed()) {
            throw std::runtime_error(
                "distributed parameters require a rank-local checkpoint shard");
        }
        if (std::any_of(values_.begin(), values_.end(), [](const auto& parameter) {
                return parameter && (parameter->nf4 || parameter->kquant);
            }))
            throw std::runtime_error("quantized base weights are immutable and adapter-only");
        fs::create_directories(path.parent_path());
        const fs::path temporary = path.string() + ".tmp";
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("failed to create native Glimmer checkpoint");
        std::vector<std::uint16_t> host;
        for (const auto& parameter : values_) {
            host.resize(static_cast<std::size_t>(parameter->spec.elements()));
            parameter->value.download(host.data(), parameter->spec.elements());
            out.write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(host.size() * 2));
            if (!out) throw std::runtime_error("failed to write native Glimmer checkpoint");
        }
        out.close();
        fs::rename(temporary, path);
    }

    void save_optimizer(const fs::path& path) const {
        if (distributed()) {
            throw std::runtime_error(
                "distributed parameters require a rank-local optimizer shard");
        }
        fs::create_directories(path.parent_path());
        const fs::path temporary = path.string() + ".tmp";
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("failed to create native Glimmer optimizer checkpoint");
        std::vector<float> host;
        for (int moment = 0; moment < 2; ++moment) {
            for (const auto& parameter : values_) {
                if (!parameter->trainable) continue;
                host.resize(static_cast<std::size_t>(parameter->spec.elements()));
                const DeviceBuffer<float>& source = moment == 0
                    ? parameter->exp_avg : parameter->exp_avg_sq;
                source.download(host.data(), parameter->spec.elements());
                out.write(reinterpret_cast<const char*>(host.data()),
                          static_cast<std::streamsize>(host.size() * sizeof(float)));
                if (!out) throw std::runtime_error("failed to write native Glimmer optimizer checkpoint");
            }
        }
        out.close();
        fs::rename(temporary, path);
    }

    void load_optimizer(const fs::path& path, std::string_view expected_sha) {
        std::int64_t trainable_elements = 0;
        for (const auto& parameter : values_)
            if (parameter && parameter->trainable)
                trainable_elements += parameter->spec.elements();
        const std::uintmax_t expected_bytes =
            static_cast<std::uintmax_t>(trainable_elements) * 2U * sizeof(float);
        if (!valid_sha256(expected_sha) || !fs::is_regular_file(path) ||
            fs::file_size(path) != expected_bytes || sha256_file(path) != expected_sha)
            throw std::runtime_error("native Glimmer optimizer checkpoint authentication failed");
        std::ifstream in(path, std::ios::binary);
        std::vector<float> host;
        for (int moment = 0; moment < 2; ++moment) {
            for (auto& parameter : values_) {
                if (!parameter || !parameter->trainable) continue;
                host.resize(static_cast<std::size_t>(parameter->spec.elements()));
                in.read(reinterpret_cast<char*>(host.data()),
                        static_cast<std::streamsize>(host.size() * sizeof(float)));
                if (!in) throw std::runtime_error("short native Glimmer optimizer checkpoint read");
                DeviceBuffer<float>& target = moment == 0
                    ? parameter->exp_avg : parameter->exp_avg_sq;
                target.upload(host.data(), parameter->spec.elements());
            }
        }
        if (in.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("native Glimmer optimizer checkpoint has trailing bytes");
    }

    std::int64_t local_parameter_elements() const {
        std::int64_t total = 0;
        for (const auto& parameter : values_) {
            if (parameter) total += parameter->spec.elements();
        }
        return total;
    }

    std::int64_t local_trainable_elements() const {
        std::int64_t total = 0;
        for (const auto& parameter : values_) {
            if (parameter && parameter->trainable)
                total += parameter->spec.elements();
        }
        return total;
    }

    std::vector<ParameterSpec> local_specs() const {
        std::vector<ParameterSpec> result;
        for (const auto& parameter : values_) {
            if (parameter) result.push_back(parameter->spec);
        }
        return result;
    }

    void save_local_model(const fs::path& path) const {
        if (!distributed())
            throw std::runtime_error("rank-local model save requires a distributed partition");
        fs::create_directories(path.parent_path());
        const fs::path temporary = path.string() + ".tmp";
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("failed to create pipeline model shard");
        std::vector<std::uint16_t> host;
        for (const auto& parameter : values_) {
            if (!parameter) continue;
            if (parameter->nf4 || parameter->kquant)
                throw std::runtime_error("pipeline model shards require BF16 parameters");
            host.resize(static_cast<std::size_t>(parameter->spec.elements()));
            parameter->value.download(host.data(), parameter->spec.elements());
            out.write(
                reinterpret_cast<const char*>(host.data()),
                static_cast<std::streamsize>(host.size() * sizeof(std::uint16_t)));
            if (!out) throw std::runtime_error("failed to write pipeline model shard");
        }
        out.close();
        fs::rename(temporary, path);
    }

    void load_local_model(
        const fs::path& path, std::string_view expected_sha,
        bool verify_digest = true) {
        const std::uintmax_t expected_bytes =
            static_cast<std::uintmax_t>(local_parameter_elements()) * 2U;
        if (!distributed() || !valid_sha256(expected_sha) ||
            !fs::is_regular_file(path) || fs::file_size(path) != expected_bytes ||
            (verify_digest && sha256_file(path) != expected_sha)) {
            throw std::runtime_error("pipeline model shard authentication failed");
        }
        const auto source_time = fs::last_write_time(path);
        std::ifstream in(path, std::ios::binary);
        std::vector<std::uint16_t> host;
        for (auto& parameter : values_) {
            if (!parameter) continue;
            host.resize(static_cast<std::size_t>(parameter->spec.elements()));
            in.read(
                reinterpret_cast<char*>(host.data()),
                static_cast<std::streamsize>(host.size() * sizeof(std::uint16_t)));
            if (!in) throw std::runtime_error("short pipeline model shard");
            parameter->value.upload(host.data(), parameter->spec.elements());
        }
        if (in.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("pipeline model shard has trailing bytes");
        if (fs::file_size(path) != expected_bytes ||
            fs::last_write_time(path) != source_time) {
            throw std::runtime_error("pipeline model shard changed while loading");
        }
    }

    void save_local_optimizer(const fs::path& path) const {
        if (!distributed())
            throw std::runtime_error("rank-local optimizer save requires a distributed partition");
        fs::create_directories(path.parent_path());
        const fs::path temporary = path.string() + ".tmp";
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("failed to create pipeline optimizer shard");
        std::vector<float> host;
        for (int moment = 0; moment < 2; ++moment) {
            for (const auto& parameter : values_) {
                if (!parameter || !parameter->trainable) continue;
                host.resize(static_cast<std::size_t>(parameter->spec.elements()));
                const auto& source = moment == 0
                    ? parameter->exp_avg : parameter->exp_avg_sq;
                source.download(host.data(), parameter->spec.elements());
                out.write(
                    reinterpret_cast<const char*>(host.data()),
                    static_cast<std::streamsize>(host.size() * sizeof(float)));
                if (!out)
                    throw std::runtime_error("failed to write pipeline optimizer shard");
            }
        }
        out.close();
        fs::rename(temporary, path);
    }

    void load_local_optimizer(
        const fs::path& path, std::string_view expected_sha,
        bool verify_digest = true) {
        const std::uintmax_t expected_bytes =
            static_cast<std::uintmax_t>(local_trainable_elements()) * 2U *
            sizeof(float);
        if (!distributed() || !valid_sha256(expected_sha) ||
            !fs::is_regular_file(path) || fs::file_size(path) != expected_bytes ||
            (verify_digest && sha256_file(path) != expected_sha)) {
            throw std::runtime_error("pipeline optimizer shard authentication failed");
        }
        const auto source_time = fs::last_write_time(path);
        std::ifstream in(path, std::ios::binary);
        std::vector<float> host;
        for (int moment = 0; moment < 2; ++moment) {
            for (auto& parameter : values_) {
                if (!parameter || !parameter->trainable) continue;
                host.resize(static_cast<std::size_t>(parameter->spec.elements()));
                in.read(
                    reinterpret_cast<char*>(host.data()),
                    static_cast<std::streamsize>(host.size() * sizeof(float)));
                if (!in) throw std::runtime_error("short pipeline optimizer shard");
                auto& target = moment == 0
                    ? parameter->exp_avg : parameter->exp_avg_sq;
                target.upload(host.data(), parameter->spec.elements());
            }
        }
        if (in.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("pipeline optimizer shard has trailing bytes");
        if (fs::file_size(path) != expected_bytes ||
            fs::last_write_time(path) != source_time) {
            throw std::runtime_error("pipeline optimizer shard changed while loading");
        }
    }

private:
    Runtime& runtime_;
    Geometry geometry_;
    std::shared_ptr<const KQuantCheckpoint> kquant_;
    PipelinePartition partition_;
    std::vector<std::unique_ptr<Parameter>> values_;
    std::vector<std::unique_ptr<Parameter>> auxiliary_;
};

constexpr std::array<std::string_view, 8> kLoraRoleOrder{
    "q_proj", "k_proj", "v_proj", "o_proj", "attn_gate_proj",
    "gate_proj", "up_proj", "down_proj"};

std::string parameter_role(std::string_view name) {
    constexpr std::string_view suffix = ".weight";
    if (!name.ends_with(suffix)) return {};
    name.remove_suffix(suffix.size());
    const std::size_t dot = name.rfind('.');
    return std::string(dot == std::string_view::npos ? name : name.substr(dot + 1));
}

std::uint32_t lora_target_mask(const std::vector<std::string>& targets) {
    std::uint32_t mask = 0;
    for (const auto& target : targets) {
        const auto found = std::find(kLoraRoleOrder.begin(), kLoraRoleOrder.end(), target);
        if (found == kLoraRoleOrder.end())
            throw std::runtime_error("unsupported Muse Glimmer LoRA target: " + target);
        mask |= 1U << static_cast<unsigned>(found - kLoraRoleOrder.begin());
    }
    return mask;
}

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

struct LoraSite {
    std::string base_name;
    std::size_t index = 0;
    Parameter a;
    Parameter b;

    LoraSite(
        Runtime& runtime, const ParameterSpec& base, std::int64_t rank,
        std::size_t site_index, std::uint64_t seed)
        : base_name(base.name), index(site_index),
          a(runtime, ParameterSpec{base.name + ".lora_A", rank, base.cols, false}),
          b(runtime, ParameterSpec{base.name + ".lora_B", base.rows, rank, false}) {
        std::vector<std::uint16_t> values(static_cast<std::size_t>(a.spec.elements()));
        const float bound = 1.0f / std::sqrt(static_cast<float>(base.cols));
        for (std::size_t element = 0; element < values.size(); ++element) {
            const std::uint64_t random = splitmix64(seed ^
                (static_cast<std::uint64_t>(site_index) << 32U) ^ element);
            const double unit = static_cast<double>(random >> 11U) *
                (1.0 / 9007199254740992.0);
            values[element] = float_to_bf16(
                static_cast<float>((2.0 * unit - 1.0) * bound));
        }
        a.value.upload(values.data(), a.spec.elements());
        values.assign(static_cast<std::size_t>(b.spec.elements()), 0U);
        b.value.upload(values.data(), b.spec.elements());
    }
};

class LoraParameters {
public:
    LoraParameters(Runtime& runtime, const Geometry& geometry, const Options& options)
        : runtime_(runtime), rank_(options.lora_rank), alpha_(options.lora_alpha),
          dropout_(options.lora_dropout), seed_(options.lora_seed),
          target_mask_(lora_target_mask(options.lora_targets)) {
        const std::unordered_set<std::string> targets(
            options.lora_targets.begin(), options.lora_targets.end());
        for (const auto& base : parameter_specs(geometry)) {
            if (!targets.contains(parameter_role(base.name))) continue;
            const std::size_t index = sites_.size();
            sites_.push_back(std::make_unique<LoraSite>(
                runtime, base, rank_, index, seed_));
            lookup_.emplace(base.name, sites_.back().get());
        }
        if (sites_.empty()) throw std::runtime_error("Muse Glimmer LoRA resolved no projection sites");
    }

    LoraSite* find(const Parameter& base) const {
        const auto found = lookup_.find(base.spec.name);
        return found == lookup_.end() ? nullptr : found->second;
    }
    float scaling() const { return alpha_ / static_cast<float>(rank_); }
    float dropout() const { return dropout_; }
    std::int64_t rank() const { return rank_; }
    float alpha() const { return alpha_; }
    std::uint64_t seed() const { return seed_; }
    std::uint32_t target_mask() const { return target_mask_; }
    std::size_t site_count() const { return sites_.size(); }
    std::uint64_t dropout_seed(const LoraSite& site, std::int64_t step) const {
        return splitmix64(seed_ ^ (static_cast<std::uint64_t>(step) << 24U) ^
                          static_cast<std::uint64_t>(site.index));
    }

    void zero_grad() {
        for (auto& site : sites_) { site->a.gradient.zero(); site->b.gradient.zero(); }
    }

    double gradient_norm() const {
        double sum = 0.0;
        std::vector<float> host;
        for (const auto& site : sites_) {
            for (const Parameter* parameter : {&site->a, &site->b}) {
                host.resize(static_cast<std::size_t>(parameter->spec.elements()));
                parameter->gradient.download(host.data(), parameter->spec.elements());
                for (float value : host) sum += static_cast<double>(value) * value;
            }
        }
        return std::sqrt(sum);
    }

    void step(TileOps& ops, std::int64_t step, const Options& options, float gradient_scale) {
        for (auto& site : sites_) {
            for (Parameter* parameter : {&site->a, &site->b}) {
                ops.check(ops.adam(
                    parameter->value.get(), parameter->gradient.get(),
                    parameter->exp_avg.get(), parameter->exp_avg_sq.get(),
                    parameter->spec.elements(), options.learning_rate, options.beta1,
                    options.beta2, options.adam_eps, 0.0f, step, gradient_scale,
                    runtime_.stream), "Glimmer LoRA AdamW");
            }
        }
    }

    std::int64_t elements() const {
        std::int64_t result = 0;
        for (const auto& site : sites_)
            result += site->a.spec.elements() + site->b.spec.elements();
        return result;
    }

    void save(const fs::path& path) const {
        fs::create_directories(path.parent_path());
        const fs::path temporary = path.string() + ".tmp";
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("failed to create native Glimmer LoRA artifact");
        std::vector<std::uint16_t> host;
        for (const auto& site : sites_) {
            for (const Parameter* parameter : {&site->a, &site->b}) {
                host.resize(static_cast<std::size_t>(parameter->spec.elements()));
                parameter->value.download(host.data(), parameter->spec.elements());
                out.write(reinterpret_cast<const char*>(host.data()),
                          static_cast<std::streamsize>(host.size() * sizeof(std::uint16_t)));
                if (!out) throw std::runtime_error("failed to write native Glimmer LoRA artifact");
            }
        }
        out.close();
        fs::rename(temporary, path);
    }

    void load(const fs::path& path, std::string_view expected_sha) {
        const std::uintmax_t expected_bytes =
            static_cast<std::uintmax_t>(elements()) * sizeof(std::uint16_t);
        if (!valid_sha256(expected_sha) || !fs::is_regular_file(path) ||
            fs::file_size(path) != expected_bytes || sha256_file(path) != expected_sha)
            throw std::runtime_error("native Glimmer LoRA artifact authentication failed");
        std::ifstream in(path, std::ios::binary);
        std::vector<std::uint16_t> host;
        for (auto& site : sites_) {
            for (Parameter* parameter : {&site->a, &site->b}) {
                host.resize(static_cast<std::size_t>(parameter->spec.elements()));
                in.read(reinterpret_cast<char*>(host.data()),
                        static_cast<std::streamsize>(host.size() * sizeof(std::uint16_t)));
                if (!in) throw std::runtime_error("short native Glimmer LoRA artifact read");
                parameter->value.upload(host.data(), parameter->spec.elements());
            }
        }
        if (in.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("native Glimmer LoRA artifact has trailing bytes");
    }

    void save_optimizer(const fs::path& path) const {
        fs::create_directories(path.parent_path());
        const fs::path temporary = path.string() + ".tmp";
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("failed to create native Glimmer LoRA optimizer");
        std::vector<float> host;
        for (int moment = 0; moment < 2; ++moment) {
            for (const auto& site : sites_) {
                for (const Parameter* parameter : {&site->a, &site->b}) {
                    const DeviceBuffer<float>& source = moment == 0
                        ? parameter->exp_avg : parameter->exp_avg_sq;
                    host.resize(static_cast<std::size_t>(parameter->spec.elements()));
                    source.download(host.data(), parameter->spec.elements());
                    out.write(reinterpret_cast<const char*>(host.data()),
                              static_cast<std::streamsize>(host.size() * sizeof(float)));
                    if (!out) throw std::runtime_error("failed to write native Glimmer LoRA optimizer");
                }
            }
        }
        out.close();
        fs::rename(temporary, path);
    }

    void load_optimizer(const fs::path& path, std::string_view expected_sha) {
        const std::uintmax_t expected_bytes =
            static_cast<std::uintmax_t>(elements()) * 2U * sizeof(float);
        if (!valid_sha256(expected_sha) || !fs::is_regular_file(path) ||
            fs::file_size(path) != expected_bytes || sha256_file(path) != expected_sha)
            throw std::runtime_error("native Glimmer LoRA optimizer authentication failed");
        std::ifstream in(path, std::ios::binary);
        std::vector<float> host;
        for (int moment = 0; moment < 2; ++moment) {
            for (auto& site : sites_) {
                for (Parameter* parameter : {&site->a, &site->b}) {
                    DeviceBuffer<float>& target = moment == 0
                        ? parameter->exp_avg : parameter->exp_avg_sq;
                    host.resize(static_cast<std::size_t>(parameter->spec.elements()));
                    in.read(reinterpret_cast<char*>(host.data()),
                            static_cast<std::streamsize>(host.size() * sizeof(float)));
                    if (!in) throw std::runtime_error("short native Glimmer LoRA optimizer read");
                    target.upload(host.data(), parameter->spec.elements());
                }
            }
        }
        if (in.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("native Glimmer LoRA optimizer has trailing bytes");
    }

    const std::vector<std::unique_ptr<LoraSite>>& sites() const { return sites_; }

private:
    Runtime& runtime_;
    std::int64_t rank_;
    float alpha_;
    float dropout_;
    std::uint64_t seed_;
    std::uint32_t target_mask_;
    std::vector<std::unique_ptr<LoraSite>> sites_;
    std::unordered_map<std::string, LoraSite*> lookup_;
};

class RewardHead {
public:
    RewardHead(Runtime& runtime, std::int64_t hidden_size, std::uint64_t seed)
        : runtime_(runtime),
          parameter_(
              runtime,
              ParameterSpec{"reward_head.weight", 1, hidden_size, false},
              true,
              false) {
        std::vector<std::uint16_t> host(static_cast<std::size_t>(hidden_size));
        const float bound = 1.0f / std::sqrt(static_cast<float>(hidden_size));
        for (std::size_t index = 0; index < host.size(); ++index) {
            const std::uint64_t random = splitmix64(seed ^ index);
            const double unit = static_cast<double>(random >> 11U) *
                (1.0 / 9007199254740992.0);
            host[index] = float_to_bf16(
                static_cast<float>((2.0 * unit - 1.0) * bound));
        }
        parameter_.value.upload(host.data(), hidden_size);
    }

    Parameter& parameter() { return parameter_; }
    void zero_grad() { parameter_.gradient.zero(); }

    double gradient_norm() const {
        std::vector<float> host(
            static_cast<std::size_t>(parameter_.spec.elements()));
        parameter_.gradient.download(host.data(), parameter_.spec.elements());
        double sum = 0.0;
        for (float value : host) sum += static_cast<double>(value) * value;
        return std::sqrt(sum);
    }

    void step(
        TileOps& ops,
        std::int64_t step,
        const Options& options,
        float gradient_scale) {
        ops.check(
            ops.adam(
                parameter_.value.get(), parameter_.gradient.get(),
                parameter_.exp_avg.get(), parameter_.exp_avg_sq.get(),
                parameter_.spec.elements(), options.learning_rate, options.beta1,
                options.beta2, options.adam_eps, 0.0f, step, gradient_scale,
                runtime_.stream),
            "Glimmer reward-head AdamW");
    }

    void save(const fs::path& path) const {
        fs::create_directories(path.parent_path());
        const fs::path temporary = path.string() + ".tmp";
        std::vector<std::uint16_t> host(
            static_cast<std::size_t>(parameter_.spec.elements()));
        parameter_.value.download(host.data(), parameter_.spec.elements());
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        out.write(
            reinterpret_cast<const char*>(host.data()),
            static_cast<std::streamsize>(host.size() * sizeof(std::uint16_t)));
        if (!out) throw std::runtime_error("failed to write native Glimmer reward head");
        out.close();
        fs::rename(temporary, path);
    }

    void load(const fs::path& path, std::string_view expected_sha) {
        const std::uintmax_t expected_bytes =
            static_cast<std::uintmax_t>(parameter_.spec.elements()) * 2U;
        if (!valid_sha256(expected_sha) || !fs::is_regular_file(path) ||
            fs::file_size(path) != expected_bytes || sha256_file(path) != expected_sha) {
            throw std::runtime_error(
                "native Glimmer reward-head artifact authentication failed");
        }
        std::vector<std::uint16_t> host(
            static_cast<std::size_t>(parameter_.spec.elements()));
        std::ifstream in(path, std::ios::binary);
        in.read(
            reinterpret_cast<char*>(host.data()),
            static_cast<std::streamsize>(host.size() * sizeof(std::uint16_t)));
        if (!in || in.peek() != std::char_traits<char>::eof()) {
            throw std::runtime_error("invalid native Glimmer reward-head artifact");
        }
        parameter_.value.upload(host.data(), parameter_.spec.elements());
    }

    void save_optimizer(const fs::path& path) const {
        fs::create_directories(path.parent_path());
        const fs::path temporary = path.string() + ".tmp";
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        std::vector<float> host(
            static_cast<std::size_t>(parameter_.spec.elements()));
        for (const DeviceBuffer<float>* source :
             {&parameter_.exp_avg, &parameter_.exp_avg_sq}) {
            source->download(host.data(), parameter_.spec.elements());
            out.write(
                reinterpret_cast<const char*>(host.data()),
                static_cast<std::streamsize>(host.size() * sizeof(float)));
        }
        if (!out) {
            throw std::runtime_error(
                "failed to write native Glimmer reward-head optimizer");
        }
        out.close();
        fs::rename(temporary, path);
    }

    void load_optimizer(const fs::path& path, std::string_view expected_sha) {
        const std::uintmax_t expected_bytes =
            static_cast<std::uintmax_t>(parameter_.spec.elements()) * 2U *
            sizeof(float);
        if (!valid_sha256(expected_sha) || !fs::is_regular_file(path) ||
            fs::file_size(path) != expected_bytes || sha256_file(path) != expected_sha) {
            throw std::runtime_error(
                "native Glimmer reward-head optimizer authentication failed");
        }
        std::ifstream in(path, std::ios::binary);
        std::vector<float> host(
            static_cast<std::size_t>(parameter_.spec.elements()));
        for (DeviceBuffer<float>* target :
             {&parameter_.exp_avg, &parameter_.exp_avg_sq}) {
            in.read(
                reinterpret_cast<char*>(host.data()),
                static_cast<std::streamsize>(host.size() * sizeof(float)));
            if (!in) {
                throw std::runtime_error(
                    "short native Glimmer reward-head optimizer");
            }
            target->upload(host.data(), parameter_.spec.elements());
        }
        if (in.peek() != std::char_traits<char>::eof()) {
            throw std::runtime_error(
                "native Glimmer reward-head optimizer has trailing bytes");
        }
    }

private:
    Runtime& runtime_;
    Parameter parameter_;
};

struct LayerActivations {
    DeviceBuffer<float> input_norm;
    DeviceBuffer<float> q_raw;
    DeviceBuffer<float> q;
    DeviceBuffer<float> k_raw;
    DeviceBuffer<float> k;
    DeviceBuffer<float> v;
    DeviceBuffer<float> gate;
    DeviceBuffer<float> attention;
    DeviceBuffer<float> logsumexp;
    DeviceBuffer<float> gated_attention;
    DeviceBuffer<float> attention_output;
    DeviceBuffer<float> post_attention;
    DeviceBuffer<float> attention_residual;
    DeviceBuffer<float> pre_feedforward;
    DeviceBuffer<float> mlp_gate;
    DeviceBuffer<float> mlp_up;
    DeviceBuffer<float> swiglu;
    DeviceBuffer<float> mlp_down;
    DeviceBuffer<float> post_feedforward;
    DeviceBuffer<float> output;

    LayerActivations(Runtime& runtime, const Geometry& g, std::int64_t rows)
        : input_norm(runtime, rows * g.dim),
          q_raw(runtime, rows * g.query_width()), q(runtime, rows * g.query_width()),
          k_raw(runtime, rows * g.kv_width()), k(runtime, rows * g.kv_width()),
          v(runtime, rows * g.kv_width()), gate(runtime, rows * g.query_width()),
          attention(runtime, rows * g.query_width()),
          logsumexp(runtime, rows * g.query_heads),
          gated_attention(runtime, rows * g.query_width()),
          attention_output(runtime, rows * g.dim), post_attention(runtime, rows * g.dim),
          attention_residual(runtime, rows * g.dim),
          pre_feedforward(runtime, rows * g.dim),
          mlp_gate(runtime, rows * g.intermediate), mlp_up(runtime, rows * g.intermediate),
          swiglu(runtime, rows * g.intermediate), mlp_down(runtime, rows * g.dim),
          post_feedforward(runtime, rows * g.dim), output(runtime, rows * g.dim) {}
};

struct LayerGradients {
    DeviceBuffer<float> post_feedforward_input;
    DeviceBuffer<float> swiglu;
    DeviceBuffer<float> mlp_gate;
    DeviceBuffer<float> mlp_up;
    DeviceBuffer<float> pre_feedforward_a;
    DeviceBuffer<float> pre_feedforward_b;
    DeviceBuffer<float> pre_feedforward;
    DeviceBuffer<float> attention_residual_from_ffn;
    DeviceBuffer<float> attention_residual;
    DeviceBuffer<float> attention_output;
    DeviceBuffer<float> gated_attention;
    DeviceBuffer<float> attention;
    DeviceBuffer<float> gate;
    DeviceBuffer<float> q;
    DeviceBuffer<float> k;
    DeviceBuffer<float> v;
    DeviceBuffer<float> q_raw;
    DeviceBuffer<float> k_raw;
    DeviceBuffer<float> norm_q;
    DeviceBuffer<float> norm_k;
    DeviceBuffer<float> norm_v;
    DeviceBuffer<float> norm_gate;
    DeviceBuffer<float> norm_qk;
    DeviceBuffer<float> norm_vg;
    DeviceBuffer<float> norm_all;
    DeviceBuffer<float> input_from_norm;
    DeviceBuffer<float> input;

    LayerGradients(Runtime& runtime, const Geometry& g, std::int64_t rows)
        : post_feedforward_input(runtime, rows * g.dim),
          swiglu(runtime, rows * g.intermediate), mlp_gate(runtime, rows * g.intermediate),
          mlp_up(runtime, rows * g.intermediate),
          pre_feedforward_a(runtime, rows * g.dim), pre_feedforward_b(runtime, rows * g.dim),
          pre_feedforward(runtime, rows * g.dim),
          attention_residual_from_ffn(runtime, rows * g.dim),
          attention_residual(runtime, rows * g.dim), attention_output(runtime, rows * g.dim),
          gated_attention(runtime, rows * g.query_width()),
          attention(runtime, rows * g.query_width()), gate(runtime, rows * g.query_width()),
          q(runtime, rows * g.query_width()), k(runtime, rows * g.kv_width()),
          v(runtime, rows * g.kv_width()), q_raw(runtime, rows * g.query_width()),
          k_raw(runtime, rows * g.kv_width()), norm_q(runtime, rows * g.dim),
          norm_k(runtime, rows * g.dim), norm_v(runtime, rows * g.dim),
          norm_gate(runtime, rows * g.dim), norm_qk(runtime, rows * g.dim),
          norm_vg(runtime, rows * g.dim), norm_all(runtime, rows * g.dim),
          input_from_norm(runtime, rows * g.dim),
          input(runtime, rows * g.dim) {}
};

class GlimmerTrainer {
public:
    GlimmerTrainer(
        Runtime& runtime, TileOps& ops, Parameters& parameters,
        LoraParameters* adapters, const Options& options,
        PipelineCollective* collective = nullptr)
        : runtime_(runtime), ops_(ops), parameters_(parameters), adapters_(adapters), options_(options),
          g_(options.geometry), partition_(parameters.partition()),
          collective_(collective), rows_(options.batch_size * options.sequence_length),
          token_ids_(runtime, rows_), targets_(runtime, rows_), mask_(runtime, rows_),
          sequence_ids_(runtime, rows_),
          initial_state_(runtime, rows_ * g_.dim),
          grad_final_state_(runtime, rows_ * g_.dim),
          forward_scratch_(runtime, g_, rows_), backward_scratch_(runtime, g_, rows_) {
        if (partition_.distributed() &&
            (collective_ == nullptr || !collective_->enabled() || adapters_ != nullptr)) {
            throw std::runtime_error(
                "distributed Glimmer trainer requires NCCL and full BF16 parameters");
        }
        if (partition_.first()) {
            raw_embedding_.allocate(runtime_, rows_ * g_.dim);
            grad_raw_embedding_.allocate(runtime_, rows_ * g_.dim);
        }
        if (partition_.last()) {
            final_norm_.allocate(runtime_, rows_ * g_.dim);
            logits_.allocate(runtime_, rows_ * g_.vocab);
            loss_rows_.allocate(runtime_, rows_);
            grad_logits_.allocate(runtime_, rows_ * g_.vocab);
            grad_raw_logits_.allocate(runtime_, rows_ * g_.vocab);
            grad_final_norm_.allocate(runtime_, rows_ * g_.dim);
        }
        if (!partition_.distributed()) {
            token_logp_.allocate(runtime_, rows_);
            token_entropy_.allocate(runtime_, rows_);
            value_.allocate(runtime_, rows_);
            rollout_logp_old_.allocate(runtime_, rows_);
            rollout_value_old_.allocate(runtime_, rows_);
            rollout_advantages_.allocate(runtime_, rows_);
            rollout_returns_.allocate(runtime_, rows_);
            grad_token_logp_.allocate(runtime_, rows_);
            grad_token_entropy_.allocate(runtime_, rows_);
            grad_value_.allocate(runtime_, rows_);
            ppo_policy_loss_.allocate(runtime_, 1);
            ppo_value_loss_.allocate(runtime_, 1);
            ppo_entropy_bonus_.allocate(runtime_, 1);
            ppo_total_loss_.allocate(runtime_, 1);
            sequence_logp_.allocate(runtime_, options.batch_size);
            grad_sequence_logp_.allocate(runtime_, options.batch_size);
            dpo_row_loss_.allocate(runtime_, options.batch_size);
            dpo_chosen_reward_.allocate(runtime_, options.batch_size);
            dpo_rejected_reward_.allocate(runtime_, options.batch_size);
            reward_.allocate(runtime_, options.batch_size);
            reward_selected_positions_.allocate(runtime_, options.batch_size);
            grad_reward_.allocate(runtime_, options.batch_size);
            grad_value_final_norm_.allocate(runtime_, rows_ * g_.dim);
        }
        const std::int64_t checkpoint_count =
            (partition_.local_layers() + options_.activation_checkpoint_interval - 1) /
                options_.activation_checkpoint_interval + 1;
        for (std::int64_t i = 0; i < checkpoint_count; ++i)
            checkpoints_.push_back(std::make_unique<DeviceBuffer<float>>(runtime_, rows_ * g_.dim));
        if (adapters_ != nullptr) {
            const std::int64_t width = std::max({g_.dim, g_.intermediate, g_.query_width()});
            lora_input_.allocate(runtime_, rows_ * width);
            lora_rank_.allocate(runtime_, rows_ * adapters_->rank());
            lora_delta_.allocate(runtime_, rows_ * width);
            lora_combined_.allocate(runtime_, rows_ * width);
            lora_grad_rank_.allocate(runtime_, rows_ * adapters_->rank());
            lora_input_grad_.allocate(runtime_, rows_ * width);
            lora_dropout_grad_.allocate(runtime_, rows_ * width);
        }
    }

    double train_step(
        const std::vector<std::uint32_t>& tokens,
        const std::vector<std::int32_t>& targets,
        const std::vector<float>* loss_mask,
        const std::vector<std::int32_t>* sequence_ids,
        std::int64_t optimizer_step) {
        const double mask_count = upload_batch(
            tokens, targets, loss_mask, sequence_ids, false);
        optimizer_step_ = optimizer_step;
        parameters_.zero_grad();
        if (adapters_ != nullptr) adapters_->zero_grad();

        embed_forward();
        forward_decoder();
        const double loss = loss_forward_backward(static_cast<float>(1.0 / mask_count));
        backward_decoder();
        embed_backward();
        runtime_.sync();

        apply_optimizer_step(optimizer_step);
        return loss;
    }

    double pipeline_train_step(
        const std::vector<std::uint32_t>& tokens,
        const std::vector<std::int32_t>& targets,
        const std::vector<float>* loss_mask,
        const std::vector<std::int32_t>* sequence_ids,
        std::int64_t optimizer_step) {
        if (!partition_.distributed() || collective_ == nullptr) {
            throw std::runtime_error(
                "pipeline_train_step requires a distributed trainer");
        }
        const double mask_count = upload_batch(
            tokens, targets, loss_mask, sequence_ids, false);
        optimizer_step_ = optimizer_step;
        parameters_.zero_grad();

        if (partition_.first()) {
            embed_forward();
        } else {
            collective_->recv_float(
                initial_state_.get(), rows_ * g_.dim, partition_.rank - 1);
            checkpoints_.front()->copy_from(initial_state_);
        }
        forward_decoder();

        double local_loss = 0.0;
        if (partition_.last()) {
            local_loss = loss_forward_backward(
                static_cast<float>(1.0 / mask_count));
        } else {
            collective_->send_float(
                final_decoder_state_.get(), rows_ * g_.dim,
                partition_.rank + 1);
            collective_->recv_float(
                grad_final_state_.get(), rows_ * g_.dim,
                partition_.rank + 1);
        }
        backward_decoder();
        if (partition_.first()) {
            embed_backward();
        } else {
            collective_->send_float(
                grad_initial_state_.get(), rows_ * g_.dim,
                partition_.rank - 1);
        }
        runtime_.sync();

        const double local_norm = parameters_.gradient_norm();
        if (!std::isfinite(local_norm)) {
            throw std::runtime_error(
                "pipeline-local Glimmer gradient norm is not finite");
        }
        const float global_squared = collective_->all_reduce_sum(
            static_cast<float>(local_norm * local_norm));
        if (!std::isfinite(global_squared) || global_squared < 0.0f) {
            throw std::runtime_error(
                "pipeline-global Glimmer gradient norm is not finite");
        }
        const double global_norm = std::sqrt(static_cast<double>(global_squared));
        const float clip = global_norm > options_.max_grad_norm
            ? static_cast<float>(options_.max_grad_norm / global_norm)
            : 1.0f;
        parameters_.step(ops_, optimizer_step, options_, clip);
        runtime_.sync();
        const float global_loss = collective_->all_reduce_sum(
            partition_.last() ? static_cast<float>(local_loss) : 0.0f);
        if (!std::isfinite(global_loss)) {
            throw std::runtime_error("pipeline Glimmer loss is not finite");
        }
        return global_loss;
    }

    double dpo_step(
        const std::vector<std::uint32_t>& paired_tokens,
        const std::vector<std::int32_t>& paired_targets,
        const std::vector<float>& paired_loss_mask,
        const std::vector<std::int32_t>& paired_sequence_ids,
        GlimmerTrainer& reference,
        std::int64_t examples,
        std::int64_t optimizer_step) {
        if (examples <= 0 || options_.batch_size != examples * 2 ||
            reference.options_.batch_size != options_.batch_size ||
            reference.rows_ != rows_ || reference.adapters_ != nullptr) {
            throw std::runtime_error(
                "native Glimmer DPO requires equal paired policy/reference trainers");
        }
        upload_batch(
            paired_tokens, paired_targets, &paired_loss_mask,
            &paired_sequence_ids, true);
        reference.upload_batch(
            paired_tokens, paired_targets, &paired_loss_mask,
            &paired_sequence_ids, true);
        optimizer_step_ = optimizer_step;
        reference.optimizer_step_ = optimizer_step;
        parameters_.zero_grad();
        if (adapters_ != nullptr) adapters_->zero_grad();

        embed_forward();
        forward_decoder();
        sequence_logp_forward();
        reference.embed_forward();
        reference.forward_decoder();
        reference.sequence_logp_forward();

        const std::uint32_t loss_type = options_.dpo_loss_type == "hinge"
            ? NFN_NATIVE_TILE_DPO_LOSS_HINGE
            : options_.dpo_loss_type == "ipo"
                ? NFN_NATIVE_TILE_DPO_LOSS_IPO
                : NFN_NATIVE_TILE_DPO_LOSS_SIGMOID;
        NfnNativeTileDpoPairwiseDescriptorV1 dpo{
            .struct_size = sizeof(NfnNativeTileDpoPairwiseDescriptorV1),
            .version = NFN_NATIVE_TILE_GLIMMER_TRAINING_V1,
            .loss_type = loss_type,
            .flags = 0,
            .policy_logp_chosen = sequence_logp_.get(),
            .policy_logp_rejected = sequence_logp_.get() + examples,
            .reference_logp_chosen = reference.sequence_logp_.get(),
            .reference_logp_rejected = reference.sequence_logp_.get() + examples,
            .row_loss = dpo_row_loss_.get(),
            .chosen_reward = dpo_chosen_reward_.get(),
            .rejected_reward = dpo_rejected_reward_.get(),
            .grad_policy_logp_chosen = grad_sequence_logp_.get(),
            .grad_policy_logp_rejected = grad_sequence_logp_.get() + examples,
            .examples = examples,
            .beta = options_.dpo_beta,
            .label_smoothing = options_.dpo_label_smoothing,
            .grad_scale = 1.0f / static_cast<float>(examples),
            .reserved0 = 0,
            .cuda_stream = runtime_.stream,
        };
        check(ops_.dpo_forward(&dpo), "Glimmer DPO pairwise forward");
        check(ops_.dpo_backward(&dpo), "Glimmer DPO pairwise backward");
        sequence_logp_backward(grad_sequence_logp_.get());
        backward_decoder();
        embed_backward();
        runtime_.sync();

        std::vector<float> host_loss(static_cast<std::size_t>(options_.batch_size));
        dpo_row_loss_.download(host_loss.data(), options_.batch_size);
        double sum = 0.0;
        for (std::int64_t index = 0; index < examples; ++index)
            sum += host_loss[static_cast<std::size_t>(index)];
        if (!std::isfinite(sum))
            throw std::runtime_error("native Glimmer DPO loss is not finite");
        apply_optimizer_step(optimizer_step);
        return sum / static_cast<double>(examples);
    }

    double reward_step(
        const std::vector<std::uint32_t>& paired_tokens,
        const std::vector<std::int32_t>& paired_targets,
        const std::vector<float>& paired_loss_mask,
        const std::vector<std::int32_t>& paired_sequence_ids,
        RewardHead& reward_head,
        std::int64_t examples,
        std::int64_t optimizer_step) {
        if (examples <= 0 || options_.batch_size != examples * 2 ||
            adapters_ != nullptr) {
            throw std::runtime_error(
                "native Glimmer reward training requires one full paired batch");
        }
        upload_batch(
            paired_tokens, paired_targets, &paired_loss_mask,
            &paired_sequence_ids, true);
        optimizer_step_ = optimizer_step;
        parameters_.zero_grad();
        reward_head.zero_grad();

        embed_forward();
        forward_decoder();
        Parameter& final_norm_parameter = parameters_.final_norm();
        norm_forward(
            final_decoder_state_.get(), &final_norm_parameter, final_norm_.get(),
            rows_, g_.dim, g_.norm_eps, false);
        auto reward_weight = reward_head.parameter().descriptor(runtime_.stream);
        NfnNativeTileMaskedRewardHeadDescriptorV1 head{
            .struct_size = sizeof(NfnNativeTileMaskedRewardHeadDescriptorV1),
            .version = NFN_NATIVE_TILE_GLIMMER_TRAINING_V1,
            .flags = 0,
            .reserved0 = 0,
            .hidden = final_norm_.get(),
            .sequence_mask = mask_.get(),
            .weight = &reward_weight,
            .reward = reward_.get(),
            .selected_positions = reward_selected_positions_.get(),
            .grad_reward = grad_reward_.get(),
            .grad_hidden = grad_final_norm_.get(),
            .grad_weight = reward_head.parameter().gradient.get(),
            .batch_size = options_.batch_size,
            .sequence_length = options_.sequence_length,
            .hidden_size = g_.dim,
            .cuda_stream = runtime_.stream,
        };
        check(
            ops_.reward_head_forward(&head),
            "Glimmer masked reward-head forward");
        NfnNativeTilePreferenceBceDescriptorV1 preference{
            .struct_size = sizeof(NfnNativeTilePreferenceBceDescriptorV1),
            .version = NFN_NATIVE_TILE_GLIMMER_TRAINING_V1,
            .flags = 0,
            .reserved0 = 0,
            .reward_chosen = reward_.get(),
            .reward_rejected = reward_.get() + examples,
            .row_loss = dpo_row_loss_.get(),
            .grad_reward_chosen = grad_reward_.get(),
            .grad_reward_rejected = grad_reward_.get() + examples,
            .examples = examples,
            .grad_scale = 1.0f / static_cast<float>(examples),
            .reserved1 = 0,
            .cuda_stream = runtime_.stream,
        };
        check(
            ops_.preference_bce_forward(&preference),
            "Glimmer preference BCE forward");
        check(
            ops_.preference_bce_backward(&preference),
            "Glimmer preference BCE backward");
        check(
            ops_.reward_head_backward(&head),
            "Glimmer masked reward-head backward");
        norm_backward(
            final_decoder_state_.get(), &final_norm_parameter,
            grad_final_norm_.get(), grad_final_state_.get(), rows_, g_.dim,
            g_.norm_eps, false);
        backward_decoder();
        embed_backward();
        runtime_.sync();

        std::vector<float> host_loss(static_cast<std::size_t>(options_.batch_size));
        dpo_row_loss_.download(host_loss.data(), options_.batch_size);
        double sum = 0.0;
        for (std::int64_t index = 0; index < examples; ++index)
            sum += host_loss[static_cast<std::size_t>(index)];
        if (!std::isfinite(sum)) {
            throw std::runtime_error("native Glimmer reward loss is not finite");
        }
        const double base_norm = parameters_.gradient_norm();
        const double head_norm = reward_head.gradient_norm();
        const double norm = std::hypot(base_norm, head_norm);
        if (!std::isfinite(norm)) {
            throw std::runtime_error(
                "native Glimmer reward gradient norm is not finite");
        }
        const float clip = norm > options_.max_grad_norm
            ? static_cast<float>(options_.max_grad_norm / norm)
            : 1.0f;
        parameters_.step(ops_, optimizer_step, options_, clip);
        reward_head.step(ops_, optimizer_step, options_, clip);
        runtime_.sync();
        return sum / static_cast<double>(examples);
    }

    void rollout_forward(
        const std::vector<std::uint32_t>& tokens,
        const std::vector<std::int32_t>& sequence_ids,
        RewardHead* value_head) {
        if (tokens.size() != static_cast<std::size_t>(rows_) ||
            sequence_ids.size() != tokens.size()) {
            throw std::runtime_error("native Glimmer rollout extent mismatch");
        }
        std::vector<std::int32_t> targets(tokens.size());
        std::vector<float> mask(tokens.size(), 1.0f);
        for (std::size_t index = 0; index < tokens.size(); ++index) {
            if (tokens[index] >= static_cast<std::uint32_t>(g_.vocab)) {
                throw std::runtime_error(
                    "native Glimmer rollout token is outside vocabulary");
            }
            targets[index] = static_cast<std::int32_t>(tokens[index]);
        }
        upload_batch(tokens, targets, &mask, &sequence_ids, false);
        embed_forward();
        forward_decoder();
        policy_heads_forward(value_head);
        runtime_.sync();
    }

    void download_logits_at(
        const std::vector<std::int64_t>& positions,
        std::vector<float>& output) const {
        if (positions.size() != static_cast<std::size_t>(options_.batch_size)) {
            throw std::runtime_error("native Glimmer rollout position extent mismatch");
        }
        output.resize(static_cast<std::size_t>(options_.batch_size * g_.vocab));
        for (std::int64_t batch = 0; batch < options_.batch_size; ++batch) {
            const std::int64_t position = positions[static_cast<std::size_t>(batch)];
            if (position < 0 || position >= options_.sequence_length) {
                throw std::runtime_error("native Glimmer rollout position is invalid");
            }
            logits_.download_range(
                output.data() + batch * g_.vocab, g_.vocab,
                (batch * options_.sequence_length + position) * g_.vocab);
        }
    }

    void evaluate_policy(
        const std::vector<std::uint32_t>& tokens,
        const std::vector<std::int32_t>& targets,
        const std::vector<float>& loss_mask,
        const std::vector<std::int32_t>& sequence_ids,
        RewardHead* value_head,
        std::vector<float>& token_logp,
        std::vector<float>* values) {
        upload_batch(tokens, targets, &loss_mask, &sequence_ids, true);
        embed_forward();
        forward_decoder();
        policy_heads_forward(value_head);
        token_logp_entropy_forward();
        runtime_.sync();
        token_logp.resize(static_cast<std::size_t>(rows_));
        token_logp_.download(token_logp.data(), rows_);
        if (values != nullptr) {
            if (value_head == nullptr) {
                throw std::runtime_error(
                    "native Glimmer policy values require a value head");
            }
            values->resize(static_cast<std::size_t>(rows_));
            value_.download(values->data(), rows_);
        }
    }

    std::vector<float> evaluate_reward(
        const std::vector<std::uint32_t>& tokens,
        const std::vector<std::int32_t>& targets,
        const std::vector<float>& loss_mask,
        const std::vector<std::int32_t>& sequence_ids,
        RewardHead& reward_head) {
        upload_batch(tokens, targets, &loss_mask, &sequence_ids, true);
        embed_forward();
        forward_decoder();
        Parameter& final_norm_parameter = parameters_.final_norm();
        norm_forward(
            final_decoder_state_.get(), &final_norm_parameter, final_norm_.get(),
            rows_, g_.dim, g_.norm_eps, false);
        auto reward_weight = reward_head.parameter().descriptor(runtime_.stream);
        NfnNativeTileMaskedRewardHeadDescriptorV1 head{
            .struct_size = sizeof(NfnNativeTileMaskedRewardHeadDescriptorV1),
            .version = NFN_NATIVE_TILE_GLIMMER_TRAINING_V1,
            .flags = 0,
            .reserved0 = 0,
            .hidden = final_norm_.get(),
            .sequence_mask = mask_.get(),
            .weight = &reward_weight,
            .reward = reward_.get(),
            .selected_positions = reward_selected_positions_.get(),
            .grad_reward = nullptr,
            .grad_hidden = nullptr,
            .grad_weight = nullptr,
            .batch_size = options_.batch_size,
            .sequence_length = options_.sequence_length,
            .hidden_size = g_.dim,
            .cuda_stream = runtime_.stream,
        };
        check(
            ops_.reward_head_forward(&head),
            "Glimmer PPO frozen reward forward");
        runtime_.sync();
        std::vector<float> scores(static_cast<std::size_t>(options_.batch_size));
        reward_.download(scores.data(), options_.batch_size);
        return scores;
    }

    double ppo_step(
        const std::vector<std::uint32_t>& tokens,
        const std::vector<std::int32_t>& targets,
        const std::vector<float>& loss_mask,
        const std::vector<std::int32_t>& sequence_ids,
        const std::vector<float>& logp_old,
        const std::vector<float>& value_old,
        const std::vector<float>& advantages,
        const std::vector<float>& returns,
        RewardHead& value_head,
        std::int64_t optimizer_step) {
        upload_batch(tokens, targets, &loss_mask, &sequence_ids, true);
        const auto require_rows = [&](const std::vector<float>& values,
                                      const char* label) {
            if (values.size() != static_cast<std::size_t>(rows_)) {
                throw std::runtime_error(
                    std::string("native Glimmer PPO ") + label +
                    " extent mismatch");
            }
        };
        require_rows(logp_old, "old log-probability");
        require_rows(value_old, "old value");
        require_rows(advantages, "advantage");
        require_rows(returns, "return");
        rollout_logp_old_.upload(logp_old.data(), rows_);
        rollout_value_old_.upload(value_old.data(), rows_);
        rollout_advantages_.upload(advantages.data(), rows_);
        rollout_returns_.upload(returns.data(), rows_);
        optimizer_step_ = optimizer_step;
        parameters_.zero_grad();
        if (adapters_ != nullptr) adapters_->zero_grad();
        value_head.zero_grad();

        embed_forward();
        forward_decoder();
        policy_heads_forward(&value_head);
        token_logp_entropy_forward();
        NfnNativeTileMaskedPpoLossDescriptorV1 ppo{
            .struct_size = sizeof(NfnNativeTileMaskedPpoLossDescriptorV1),
            .version = NFN_NATIVE_TILE_GLIMMER_TRAINING_V1,
            .flags = NFN_NATIVE_TILE_PPO_NORMALIZE_ADVANTAGES,
            .reserved0 = 0,
            .logp_new = token_logp_.get(),
            .logp_old = rollout_logp_old_.get(),
            .advantages = rollout_advantages_.get(),
            .value_new = value_.get(),
            .value_old = rollout_value_old_.get(),
            .returns = rollout_returns_.get(),
            .loss_mask = mask_.get(),
            .entropy = token_entropy_.get(),
            .policy_loss = ppo_policy_loss_.get(),
            .value_loss = ppo_value_loss_.get(),
            .entropy_bonus = ppo_entropy_bonus_.get(),
            .total_loss = ppo_total_loss_.get(),
            .grad_logp_new = grad_token_logp_.get(),
            .grad_value_new = grad_value_.get(),
            .grad_entropy = grad_token_entropy_.get(),
            .rows = rows_,
            .clip_range = options_.ppo_clip,
            .value_coefficient = options_.ppo_value_coefficient,
            .entropy_coefficient = options_.ppo_entropy_coefficient,
            .epsilon = 1.0e-8f,
            .cuda_stream = runtime_.stream,
        };
        check(ops_.masked_ppo_forward(&ppo), "Glimmer masked PPO forward");
        check(ops_.masked_ppo_backward(&ppo), "Glimmer masked PPO backward");
        token_logp_entropy_backward();
        check(
            ops_.logit_backward(
                logits_.get(), grad_logits_.get(), grad_raw_logits_.get(),
                logits_.count(), g_.output_multiplier, g_.softcap,
                runtime_.stream),
            "Glimmer PPO logit transform backward");
        linear_backward(
            parameters_.lm_head(), final_norm_.get(), grad_raw_logits_.get(),
            grad_final_norm_.get(), rows_);
        linear_backward(
            value_head.parameter(), final_norm_.get(), grad_value_.get(),
            grad_value_final_norm_.get(), rows_);
        check(
            ops_.add(
                grad_final_norm_.get(), grad_value_final_norm_.get(),
                grad_final_norm_.get(), grad_final_norm_.count(),
                runtime_.stream),
            "Glimmer PPO policy/value hidden gradient sum");
        Parameter& final_norm_parameter = parameters_.final_norm();
        norm_backward(
            final_decoder_state_.get(), &final_norm_parameter,
            grad_final_norm_.get(), grad_final_state_.get(), rows_, g_.dim,
            g_.norm_eps, false);
        backward_decoder();
        embed_backward();
        runtime_.sync();

        float loss = 0.0f;
        ppo_total_loss_.download(&loss, 1);
        if (!std::isfinite(loss)) {
            throw std::runtime_error("native Glimmer PPO loss is not finite");
        }
        const double base_norm = parameters_.gradient_norm();
        const double adapter_norm =
            adapters_ == nullptr ? 0.0 : adapters_->gradient_norm();
        const double value_norm = value_head.gradient_norm();
        const double norm = std::hypot(std::hypot(base_norm, adapter_norm), value_norm);
        if (!std::isfinite(norm)) {
            throw std::runtime_error(
                "native Glimmer PPO gradient norm is not finite");
        }
        const float clip = norm > options_.max_grad_norm
            ? static_cast<float>(options_.max_grad_norm / norm)
            : 1.0f;
        parameters_.step(ops_, optimizer_step, options_, clip);
        if (adapters_ != nullptr)
            adapters_->step(ops_, optimizer_step, options_, clip);
        value_head.step(ops_, optimizer_step, options_, clip);
        runtime_.sync();
        return loss;
    }

private:
    void check(int status, const char* operation) { ops_.check(status, operation); }

    double upload_batch(
        const std::vector<std::uint32_t>& tokens,
        const std::vector<std::int32_t>& targets,
        const std::vector<float>* loss_mask,
        const std::vector<std::int32_t>* sequence_ids,
        bool require_each_example) {
        if (tokens.size() != static_cast<std::size_t>(rows_) ||
            targets.size() != static_cast<std::size_t>(rows_)) {
            throw std::runtime_error("native Glimmer batch extent mismatch");
        }
        std::vector<std::int32_t> host_tokens(tokens.size());
        std::vector<std::int32_t> host_targets(targets.size());
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            if (tokens[i] >= static_cast<std::uint32_t>(g_.vocab) ||
                (targets[i] != -100 &&
                 (targets[i] < 0 || targets[i] >= g_.vocab))) {
                throw std::runtime_error(
                    "native Glimmer batch token is outside vocabulary");
            }
            host_tokens[i] = static_cast<std::int32_t>(tokens[i]);
            host_targets[i] = targets[i];
        }
        std::vector<float> host_mask(tokens.size(), 1.0f);
        if (loss_mask != nullptr) {
            if (loss_mask->size() != tokens.size()) {
                throw std::runtime_error("native Glimmer loss mask extent mismatch");
            }
            host_mask = *loss_mask;
        }
        double mask_count = 0.0;
        std::vector<double> example_mask(
            static_cast<std::size_t>(options_.batch_size), 0.0);
        for (std::size_t index = 0; index < host_mask.size(); ++index) {
            const float value = host_mask[index];
            if (!std::isfinite(value) || value < 0.0f ||
                (host_targets[index] == -100 && value != 0.0f)) {
                throw std::runtime_error(
                    "native Glimmer loss mask must be finite, non-negative, "
                    "and zero for ignored targets");
            }
            mask_count += value;
            example_mask[index / static_cast<std::size_t>(options_.sequence_length)] +=
                value;
        }
        if (!(mask_count > 0.0) ||
            (require_each_example &&
             std::any_of(example_mask.begin(), example_mask.end(), [](double value) {
                 return !(value > 0.0);
             }))) {
            throw std::runtime_error("native Glimmer loss mask is empty");
        }
        std::vector<std::int32_t> host_sequence_ids(tokens.size(), 0);
        if (sequence_ids != nullptr) {
            if (sequence_ids->size() != tokens.size()) {
                throw std::runtime_error("native Glimmer sequence-id extent mismatch");
            }
            host_sequence_ids = *sequence_ids;
        }
        for (std::int64_t batch = 0; batch < options_.batch_size; ++batch) {
            std::int32_t previous = 0;
            for (std::int64_t position = 0;
                 position < options_.sequence_length;
                 ++position) {
                const std::size_t index = static_cast<std::size_t>(
                    batch * options_.sequence_length + position);
                const std::int32_t value = host_sequence_ids[index];
                if (value < 0 || (position == 0 && value != 0) ||
                    (position > 0 && value != previous &&
                     value != previous + 1)) {
                    throw std::runtime_error(
                        "native Glimmer packed sequence IDs are invalid");
                }
                previous = value;
            }
        }
        token_ids_.upload(host_tokens.data(), rows_);
        targets_.upload(host_targets.data(), rows_);
        mask_.upload(host_mask.data(), rows_);
        sequence_ids_.upload(host_sequence_ids.data(), rows_);
        return mask_count;
    }

    void apply_optimizer_step(std::int64_t optimizer_step) {
        const double base_norm = parameters_.gradient_norm();
        const double adapter_norm =
            adapters_ == nullptr ? 0.0 : adapters_->gradient_norm();
        const double norm = std::hypot(base_norm, adapter_norm);
        if (!std::isfinite(norm)) {
            throw std::runtime_error("native Glimmer gradient norm is not finite");
        }
        const float clip = norm > options_.max_grad_norm
            ? static_cast<float>(options_.max_grad_norm / norm)
            : 1.0f;
        parameters_.step(ops_, optimizer_step, options_, clip);
        if (adapters_ != nullptr)
            adapters_->step(ops_, optimizer_step, options_, clip);
        runtime_.sync();
    }

    NfnNativeTilePackedWeightDescriptorV1 descriptor(Parameter& parameter) {
        return parameter.descriptor(runtime_.stream);
    }

    void linear(Parameter& parameter, const float* input, float* output, std::int64_t rows) {
        auto weight = descriptor(parameter);
        check(ops_.linear(&weight, input, nullptr, output, rows, false), "Glimmer linear forward");
        LoraSite* site = adapters_ == nullptr ? nullptr : adapters_->find(parameter);
        if (site == nullptr) return;
        const std::int64_t input_elements = rows * parameter.spec.cols;
        const std::uint64_t seed = adapters_->dropout_seed(*site, optimizer_step_);
        if (adapters_->dropout() == 0.0f) {
            check(ops_.copy(input, lora_input_.get(), input_elements, runtime_.stream),
                  "Glimmer LoRA input copy");
        } else {
            check(ops_.dropout_forward(
                input, lora_input_.get(), input_elements, adapters_->dropout(),
                static_cast<std::int64_t>(seed),
                runtime_.stream), "Glimmer LoRA dropout forward");
        }
        auto a = descriptor(site->a);
        auto b = descriptor(site->b);
        check(ops_.linear(&a, lora_input_.get(), nullptr, lora_rank_.get(), rows, false),
              "Glimmer LoRA A forward");
        check(ops_.linear(&b, lora_rank_.get(), nullptr, lora_delta_.get(), rows, false),
              "Glimmer LoRA B forward");
        const std::int64_t output_elements = rows * parameter.spec.rows;
        check(ops_.scale(lora_delta_.get(), output_elements, adapters_->scaling(), runtime_.stream),
              "Glimmer LoRA scaling");
        check(ops_.add(output, lora_delta_.get(), lora_combined_.get(), output_elements,
                       runtime_.stream), "Glimmer LoRA residual");
        check(ops_.copy(lora_combined_.get(), output, output_elements, runtime_.stream),
              "Glimmer LoRA output copy");
    }

    void linear_backward(
        Parameter& parameter, const float* input, const float* grad_output,
        float* grad_input, std::int64_t rows) {
        auto weight = descriptor(parameter);
        check(ops_.linear_backward_input(&weight, grad_output, grad_input, rows),
              "Glimmer linear backward input");
        if (parameter.trainable) {
            check(ops_.linear_backward_weight(
                input, grad_output, parameter.gradient.get(), rows,
                parameter.spec.cols, parameter.spec.rows, runtime_.stream),
                "Glimmer linear backward weight");
        }
        LoraSite* site = adapters_ == nullptr ? nullptr : adapters_->find(parameter);
        if (site == nullptr) return;
        const std::int64_t input_elements = rows * parameter.spec.cols;
        const std::int64_t output_elements = rows * parameter.spec.rows;
        const std::uint64_t seed = adapters_->dropout_seed(*site, optimizer_step_);
        if (adapters_->dropout() == 0.0f) {
            check(ops_.copy(input, lora_input_.get(), input_elements, runtime_.stream),
                  "Glimmer LoRA backward input copy");
        } else {
            check(ops_.dropout_forward(
                input, lora_input_.get(), input_elements, adapters_->dropout(),
                static_cast<std::int64_t>(seed),
                runtime_.stream), "Glimmer LoRA backward dropout replay");
        }
        auto a = descriptor(site->a);
        auto b = descriptor(site->b);
        check(ops_.linear(&a, lora_input_.get(), nullptr, lora_rank_.get(), rows, false),
              "Glimmer LoRA A replay");
        check(ops_.linear_backward_input(&b, grad_output, lora_grad_rank_.get(), rows),
              "Glimmer LoRA B backward input");
        check(ops_.linear_backward_weight(
            lora_rank_.get(), grad_output, site->b.gradient.get(), rows,
            site->b.spec.cols, site->b.spec.rows, runtime_.stream),
            "Glimmer LoRA B backward weight");
        check(ops_.scale(site->b.gradient.get(), site->b.spec.elements(),
                         adapters_->scaling(), runtime_.stream),
              "Glimmer LoRA B gradient scale");
        check(ops_.linear_backward_weight(
            lora_input_.get(), lora_grad_rank_.get(), site->a.gradient.get(), rows,
            site->a.spec.cols, site->a.spec.rows, runtime_.stream),
            "Glimmer LoRA A backward weight");
        check(ops_.scale(site->a.gradient.get(), site->a.spec.elements(),
                         adapters_->scaling(), runtime_.stream),
              "Glimmer LoRA A gradient scale");
        check(ops_.linear_backward_input(&a, lora_grad_rank_.get(), lora_input_grad_.get(), rows),
              "Glimmer LoRA A backward input");
        check(ops_.scale(lora_input_grad_.get(), input_elements,
                         adapters_->scaling(), runtime_.stream),
              "Glimmer LoRA input gradient scale");
        const float* lora_input_gradient = lora_input_grad_.get();
        if (adapters_->dropout() != 0.0f) {
            check(ops_.dropout_backward(
                lora_input_grad_.get(), lora_dropout_grad_.get(), input_elements,
                adapters_->dropout(), static_cast<std::int64_t>(seed), runtime_.stream),
                "Glimmer LoRA dropout backward");
            lora_input_gradient = lora_dropout_grad_.get();
        }
        check(ops_.add(grad_input, lora_input_gradient, lora_combined_.get(),
                       input_elements, runtime_.stream), "Glimmer LoRA input gradient residual");
        check(ops_.copy(lora_combined_.get(), grad_input, input_elements, runtime_.stream),
              "Glimmer LoRA input gradient copy");
        (void)output_elements;
    }

    void norm_forward(
        const float* input, Parameter* parameter, float* output,
        std::int64_t rows, std::int64_t width, float eps, bool centered) {
        NfnNativeTilePackedWeightDescriptorV1 weight{};
        const NfnNativeTilePackedWeightDescriptorV1* pointer = nullptr;
        if (parameter != nullptr) { weight = descriptor(*parameter); pointer = &weight; }
        check(ops_.norm(input, pointer, output, rows, width, eps, centered, runtime_.stream),
              "Glimmer RMSNorm forward");
    }

    void norm_backward(
        const float* input, Parameter* parameter, const float* grad_output,
        float* grad_input, std::int64_t rows, std::int64_t width,
        float eps, bool centered) {
        NfnNativeTilePackedWeightDescriptorV1 weight{};
        const NfnNativeTilePackedWeightDescriptorV1* pointer = nullptr;
        if (parameter != nullptr) { weight = descriptor(*parameter); pointer = &weight; }
        NfnNativeTileGlimmerRmsNormBackwardDescriptorV1 operation{
            .struct_size = sizeof(NfnNativeTileGlimmerRmsNormBackwardDescriptorV1),
            .version = NFN_NATIVE_TILE_GLIMMER_TRAINING_V1,
            .flags = 0, .reserved0 = 0, .input = input, .weight = pointer,
            .grad_output = grad_output, .grad_input = grad_input,
            .grad_weight = parameter && parameter->trainable ? parameter->gradient.get() : nullptr,
            .rows = rows, .width = width, .eps = eps,
            .centered = centered ? 1U : 0U, .cuda_stream = runtime_.stream,
        };
        check(ops_.norm_backward(&operation), "Glimmer RMSNorm backward");
    }

    void embed_forward() {
        auto embedding = descriptor(parameters_.embedding());
        check(ops_.embedding(&embedding, token_ids_.get(), raw_embedding_.get(), rows_),
              "Glimmer embedding forward");
        norm_forward(raw_embedding_.get(), nullptr, initial_state_.get(), rows_, g_.dim,
                     g_.norm_eps, false);
        checkpoints_.front()->copy_from(initial_state_);
    }

    void forward_layer(std::int64_t layer_index, const float* input, LayerActivations& a) {
        Parameter& input_norm = parameters_.layer(layer_index, 0);
        Parameter& post_attention_norm = parameters_.layer(layer_index, 1);
        Parameter& pre_ffn_norm = parameters_.layer(layer_index, 2);
        Parameter& post_ffn_norm = parameters_.layer(layer_index, 3);
        const bool centered = parameters_.centered_layer_norms();
        norm_forward(input, &input_norm, a.input_norm.get(), rows_, g_.dim,
                     g_.norm_eps, centered);
        linear(parameters_.layer(layer_index, 4), a.input_norm.get(), a.q_raw.get(), rows_);
        linear(parameters_.layer(layer_index, 5), a.input_norm.get(), a.k_raw.get(), rows_);
        linear(parameters_.layer(layer_index, 6), a.input_norm.get(), a.v.get(), rows_);
        linear(parameters_.layer(layer_index, 7), a.input_norm.get(), a.gate.get(), rows_);
        norm_forward(a.q_raw.get(), parameters_.q_norm(layer_index), a.q.get(),
                     rows_ * g_.query_heads,
                     g_.head_dim, g_.norm_eps, false);
        norm_forward(a.k_raw.get(), parameters_.k_norm(layer_index), a.k.get(),
                     rows_ * g_.kv_heads,
                     g_.head_dim, g_.norm_eps, false);
        check(ops_.scale(a.q.get(), a.q.count(),
                         parameters_.query_scale(g_.q_scale), runtime_.stream),
              "Glimmer query scale");
        if (g_.local(layer_index)) {
            for (std::int64_t batch = 0; batch < options_.batch_size; ++batch) {
                check(ops_.rope(
                    a.q.get() + batch * options_.sequence_length * g_.query_width(),
                    a.k.get() + batch * options_.sequence_length * g_.kv_width(),
                    options_.sequence_length, g_.query_heads, g_.kv_heads, g_.head_dim,
                    0, g_.rope_theta, parameters_.rope_layout(), false,
                    runtime_.stream), "Glimmer positioned RoPE forward");
            }
        }
        NfnNativeTileGlimmerAttentionTrainingDescriptorV1 attention{
            .struct_size = sizeof(NfnNativeTileGlimmerAttentionTrainingDescriptorV1),
            .version = NFN_NATIVE_TILE_GLIMMER_TRAINING_V1,
            .flags = NFN_NATIVE_TILE_GLIMMER_TRAIN_CAUSAL, .reserved0 = 0,
            .query = a.q.get(), .key = a.k.get(), .value = a.v.get(),
            .output = a.attention.get(), .logsumexp = a.logsumexp.get(),
            .grad_output = nullptr, .grad_query = nullptr, .grad_key = nullptr,
            .grad_value = nullptr, .batch_size = options_.batch_size,
            .sequence_length = options_.sequence_length,
            .query_heads = g_.query_heads, .kv_heads = g_.kv_heads,
            .head_dim = g_.head_dim,
            .window = g_.local(layer_index) ? g_.window : 0,
            .scale = 1.0f / std::sqrt(static_cast<float>(g_.head_dim)),
            .reserved1 = 0, .cuda_stream = runtime_.stream,
            .sequence_ids = sequence_ids_.get(), .reserved2 = 0, .reserved3 = 0,
        };
        check(ops_.attention_forward(&attention), "Glimmer GQA forward");
        check(ops_.gate(a.attention.get(), a.gate.get(), a.gated_attention.get(),
                        a.gated_attention.count(), runtime_.stream),
              "Glimmer attention gate forward");
        linear(parameters_.layer(layer_index, 8), a.gated_attention.get(),
               a.attention_output.get(), rows_);
        norm_forward(a.attention_output.get(), &post_attention_norm,
                     a.post_attention.get(), rows_, g_.dim, g_.post_norm_eps,
                     centered);
        check(ops_.add(input, a.post_attention.get(), a.attention_residual.get(),
                       rows_ * g_.dim, runtime_.stream), "Glimmer attention residual");
        norm_forward(a.attention_residual.get(), &pre_ffn_norm,
                     a.pre_feedforward.get(), rows_, g_.dim, g_.norm_eps,
                     centered);
        linear(parameters_.layer(layer_index, 9), a.pre_feedforward.get(), a.mlp_gate.get(), rows_);
        linear(parameters_.layer(layer_index, 10), a.pre_feedforward.get(), a.mlp_up.get(), rows_);
        check(ops_.swiglu(a.mlp_gate.get(), a.mlp_up.get(), a.swiglu.get(),
                          a.swiglu.count(), runtime_.stream), "Glimmer SwiGLU forward");
        linear(parameters_.layer(layer_index, 11), a.swiglu.get(), a.mlp_down.get(), rows_);
        norm_forward(a.mlp_down.get(), &post_ffn_norm, a.post_feedforward.get(),
                     rows_, g_.dim, g_.post_norm_eps, centered);
        check(ops_.add(a.attention_residual.get(), a.post_feedforward.get(), a.output.get(),
                       rows_ * g_.dim, runtime_.stream), "Glimmer feed-forward residual");
    }

    void forward_decoder() {
        DeviceBuffer<float> current(runtime_, rows_ * g_.dim);
        DeviceBuffer<float> next(runtime_, rows_ * g_.dim);
        current.copy_from(initial_state_);
        std::int64_t checkpoint_index = 1;
        for (std::int64_t layer = partition_.layer_begin;
             layer < partition_.layer_end; ++layer) {
            forward_layer(layer, current.get(), forward_scratch_);
            runtime_.check(runtime_.memcpy(next.get(), forward_scratch_.output.get(), next.bytes(),
                                           kDeviceToDevice), "cudaMemcpy decoder state");
            std::swap(current, next);
            const std::int64_t local_layer = layer - partition_.layer_begin + 1;
            if (local_layer % options_.activation_checkpoint_interval == 0 ||
                layer + 1 == partition_.layer_end)
                checkpoints_.at(static_cast<std::size_t>(checkpoint_index++))->copy_from(current);
        }
        final_decoder_state_.allocate(runtime_, rows_ * g_.dim);
        final_decoder_state_.copy_from(current);
    }

    void policy_heads_forward(RewardHead* value_head) {
        Parameter& final_norm_parameter = parameters_.final_norm();
        norm_forward(
            final_decoder_state_.get(), &final_norm_parameter,
            final_norm_.get(), rows_, g_.dim, g_.norm_eps, false);
        linear(parameters_.lm_head(), final_norm_.get(), logits_.get(), rows_);
        check(
            ops_.logit(
                logits_.get(), logits_.count(), g_.output_multiplier,
                g_.softcap, runtime_.stream),
            "Glimmer policy logit transform");
        if (value_head != nullptr) {
            linear(
                value_head->parameter(), final_norm_.get(), value_.get(),
                rows_);
        }
    }

    void token_logp_entropy_forward() {
        NfnNativeTileTokenLogpEntropyDescriptorV1 descriptor{
            .struct_size = sizeof(NfnNativeTileTokenLogpEntropyDescriptorV1),
            .version = NFN_NATIVE_TILE_GLIMMER_TRAINING_V1,
            .flags = 0,
            .reserved0 = 0,
            .transformed_logits = logits_.get(),
            .targets = targets_.get(),
            .loss_mask = mask_.get(),
            .token_logp = token_logp_.get(),
            .token_entropy = token_entropy_.get(),
            .grad_token_logp = nullptr,
            .grad_token_entropy = nullptr,
            .grad_transformed_logits = nullptr,
            .rows = rows_,
            .vocab_size = g_.vocab,
            .ignore_index = -100,
            .reserved1 = 0,
            .cuda_stream = runtime_.stream,
        };
        check(
            ops_.token_logp_entropy_forward(&descriptor),
            "Glimmer token log-probability/entropy forward");
    }

    void token_logp_entropy_backward() {
        NfnNativeTileTokenLogpEntropyDescriptorV1 descriptor{
            .struct_size = sizeof(NfnNativeTileTokenLogpEntropyDescriptorV1),
            .version = NFN_NATIVE_TILE_GLIMMER_TRAINING_V1,
            .flags = 0,
            .reserved0 = 0,
            .transformed_logits = logits_.get(),
            .targets = targets_.get(),
            .loss_mask = mask_.get(),
            .token_logp = token_logp_.get(),
            .token_entropy = token_entropy_.get(),
            .grad_token_logp = grad_token_logp_.get(),
            .grad_token_entropy = grad_token_entropy_.get(),
            .grad_transformed_logits = grad_logits_.get(),
            .rows = rows_,
            .vocab_size = g_.vocab,
            .ignore_index = -100,
            .reserved1 = 0,
            .cuda_stream = runtime_.stream,
        };
        check(
            ops_.token_logp_entropy_backward(&descriptor),
            "Glimmer token log-probability/entropy backward");
    }

    void sequence_logp_forward() {
        Parameter& final_norm_parameter = parameters_.final_norm();
        norm_forward(
            final_decoder_state_.get(), &final_norm_parameter, final_norm_.get(),
            rows_, g_.dim, g_.norm_eps, false);
        linear(parameters_.lm_head(), final_norm_.get(), logits_.get(), rows_);
        check(
            ops_.logit(
                logits_.get(), logits_.count(), g_.output_multiplier,
                g_.softcap, runtime_.stream),
            "Glimmer logit transform");
        NfnNativeTileSequenceLogpDescriptorV1 descriptor{
            .struct_size = sizeof(NfnNativeTileSequenceLogpDescriptorV1),
            .version = NFN_NATIVE_TILE_GLIMMER_TRAINING_V1,
            .flags = 0,
            .reserved0 = 0,
            .transformed_logits = logits_.get(),
            .targets = targets_.get(),
            .loss_mask = mask_.get(),
            .sequence_logp = sequence_logp_.get(),
            .grad_sequence_logp = nullptr,
            .grad_transformed_logits = nullptr,
            .batch_size = options_.batch_size,
            .sequence_length = options_.sequence_length,
            .vocab_size = g_.vocab,
            .ignore_index = -100,
            .reserved1 = 0,
            .cuda_stream = runtime_.stream,
        };
        check(
            ops_.sequence_logp_forward(&descriptor),
            "Glimmer sequence log-probability forward");
    }

    void sequence_logp_backward(const float* grad_sequence_logp) {
        if (grad_sequence_logp == nullptr) {
            throw std::runtime_error(
                "native Glimmer sequence log-probability gradient is null");
        }
        NfnNativeTileSequenceLogpDescriptorV1 descriptor{
            .struct_size = sizeof(NfnNativeTileSequenceLogpDescriptorV1),
            .version = NFN_NATIVE_TILE_GLIMMER_TRAINING_V1,
            .flags = 0,
            .reserved0 = 0,
            .transformed_logits = logits_.get(),
            .targets = targets_.get(),
            .loss_mask = mask_.get(),
            .sequence_logp = sequence_logp_.get(),
            .grad_sequence_logp = grad_sequence_logp,
            .grad_transformed_logits = grad_logits_.get(),
            .batch_size = options_.batch_size,
            .sequence_length = options_.sequence_length,
            .vocab_size = g_.vocab,
            .ignore_index = -100,
            .reserved1 = 0,
            .cuda_stream = runtime_.stream,
        };
        check(
            ops_.sequence_logp_backward(&descriptor),
            "Glimmer sequence log-probability backward");
        check(
            ops_.logit_backward(
                logits_.get(), grad_logits_.get(), grad_raw_logits_.get(),
                logits_.count(), g_.output_multiplier, g_.softcap,
                runtime_.stream),
            "Glimmer logit transform backward");
        linear_backward(
            parameters_.lm_head(), final_norm_.get(), grad_raw_logits_.get(),
            grad_final_norm_.get(), rows_);
        norm_backward(
            final_decoder_state_.get(), &parameters_.final_norm(),
            grad_final_norm_.get(), grad_final_state_.get(), rows_, g_.dim,
            g_.norm_eps, false);
    }

    double loss_forward_backward(float loss_gradient_scale) {
        Parameter& final_norm_parameter = parameters_.final_norm();
        norm_forward(final_decoder_state_.get(), &final_norm_parameter, final_norm_.get(),
                     rows_, g_.dim, g_.norm_eps, false);
        linear(parameters_.lm_head(), final_norm_.get(), logits_.get(), rows_);
        check(ops_.logit(logits_.get(), logits_.count(), g_.output_multiplier,
                         g_.softcap, runtime_.stream), "Glimmer logit transform");
        NfnNativeTileGlimmerMaskedCeDescriptorV1 ce{
            .struct_size = sizeof(NfnNativeTileGlimmerMaskedCeDescriptorV1),
            .version = NFN_NATIVE_TILE_GLIMMER_TRAINING_V1,
            .flags = 0, .reserved0 = 0, .transformed_logits = logits_.get(),
            .targets = targets_.get(), .loss_mask = mask_.get(),
            .row_loss = loss_rows_.get(), .grad_transformed_logits = grad_logits_.get(),
            .rows = rows_, .vocab_size = g_.vocab, .ignore_index = -100,
            .reserved1 = 0, .grad_scale = loss_gradient_scale, .reserved2 = 0,
            .cuda_stream = runtime_.stream,
        };
        check(ops_.ce(&ce), "Glimmer masked cross entropy");
        check(ops_.logit_backward(logits_.get(), grad_logits_.get(), grad_raw_logits_.get(),
                                  logits_.count(), g_.output_multiplier, g_.softcap,
                                  runtime_.stream), "Glimmer logit transform backward");
        linear_backward(parameters_.lm_head(), final_norm_.get(), grad_raw_logits_.get(),
                        grad_final_norm_.get(), rows_);
        norm_backward(final_decoder_state_.get(), &final_norm_parameter, grad_final_norm_.get(),
                      grad_final_state_.get(), rows_, g_.dim, g_.norm_eps, false);
        runtime_.sync();
        std::vector<float> host_loss(static_cast<std::size_t>(rows_));
        std::vector<float> host_mask(static_cast<std::size_t>(rows_));
        loss_rows_.download(host_loss.data(), rows_);
        mask_.download(host_mask.data(), rows_);
        double sum = 0.0, count = 0.0;
        for (std::int64_t row = 0; row < rows_; ++row) {
            sum += host_loss[static_cast<std::size_t>(row)];
            count += host_mask[static_cast<std::size_t>(row)];
        }
        if (!(count > 0.0) || !std::isfinite(sum)) throw std::runtime_error("invalid native Glimmer loss");
        return sum / count;
    }

    void backward_layer(
        std::int64_t layer_index, const float* input, const float* grad_output,
        LayerActivations& a, LayerGradients& b) {
        // x_out = attention_residual + post_ffn(down(swiglu(...)))
        const bool centered = parameters_.centered_layer_norms();
        norm_backward(a.mlp_down.get(), &parameters_.layer(layer_index, 3), grad_output,
                      b.post_feedforward_input.get(), rows_, g_.dim,
                      g_.post_norm_eps, centered);
        linear_backward(parameters_.layer(layer_index, 11), a.swiglu.get(),
                        b.post_feedforward_input.get(), b.swiglu.get(), rows_);
        check(ops_.swiglu_backward(a.mlp_gate.get(), a.mlp_up.get(), b.swiglu.get(),
                                   b.mlp_gate.get(), b.mlp_up.get(), b.swiglu.count(),
                                   runtime_.stream), "Glimmer SwiGLU backward");
        linear_backward(parameters_.layer(layer_index, 9), a.pre_feedforward.get(),
                        b.mlp_gate.get(), b.pre_feedforward_a.get(), rows_);
        linear_backward(parameters_.layer(layer_index, 10), a.pre_feedforward.get(),
                        b.mlp_up.get(), b.pre_feedforward_b.get(), rows_);
        check(ops_.add(b.pre_feedforward_a.get(), b.pre_feedforward_b.get(),
                       b.pre_feedforward.get(), rows_ * g_.dim, runtime_.stream),
              "Glimmer MLP input gradient sum");
        norm_backward(a.attention_residual.get(), &parameters_.layer(layer_index, 2),
                      b.pre_feedforward.get(), b.attention_residual_from_ffn.get(),
                      rows_, g_.dim, g_.norm_eps, centered);
        check(ops_.add(grad_output, b.attention_residual_from_ffn.get(),
                       b.attention_residual.get(), rows_ * g_.dim, runtime_.stream),
              "Glimmer feed-forward residual backward");
        norm_backward(a.attention_output.get(), &parameters_.layer(layer_index, 1),
                      b.attention_residual.get(), b.attention_output.get(),
                      rows_, g_.dim, g_.post_norm_eps, centered);
        linear_backward(parameters_.layer(layer_index, 8), a.gated_attention.get(),
                        b.attention_output.get(), b.gated_attention.get(), rows_);
        check(ops_.gate_backward(a.attention.get(), a.gate.get(), b.gated_attention.get(),
                                 b.attention.get(), b.gate.get(), b.gated_attention.count(),
                                 runtime_.stream), "Glimmer attention gate backward");
        NfnNativeTileGlimmerAttentionTrainingDescriptorV1 attention{
            .struct_size = sizeof(NfnNativeTileGlimmerAttentionTrainingDescriptorV1),
            .version = NFN_NATIVE_TILE_GLIMMER_TRAINING_V1,
            .flags = NFN_NATIVE_TILE_GLIMMER_TRAIN_CAUSAL, .reserved0 = 0,
            .query = a.q.get(), .key = a.k.get(), .value = a.v.get(),
            .output = a.attention.get(), .logsumexp = a.logsumexp.get(),
            .grad_output = b.attention.get(), .grad_query = b.q.get(),
            .grad_key = b.k.get(), .grad_value = b.v.get(),
            .batch_size = options_.batch_size, .sequence_length = options_.sequence_length,
            .query_heads = g_.query_heads, .kv_heads = g_.kv_heads,
            .head_dim = g_.head_dim, .window = g_.local(layer_index) ? g_.window : 0,
            .scale = 1.0f / std::sqrt(static_cast<float>(g_.head_dim)),
            .reserved1 = 0, .cuda_stream = runtime_.stream,
            .sequence_ids = sequence_ids_.get(), .reserved2 = 0, .reserved3 = 0,
        };
        check(ops_.attention_backward(&attention), "Glimmer GQA backward");
        if (g_.local(layer_index)) {
            for (std::int64_t batch = 0; batch < options_.batch_size; ++batch) {
                check(ops_.rope(
                    b.q.get() + batch * options_.sequence_length * g_.query_width(),
                    b.k.get() + batch * options_.sequence_length * g_.kv_width(),
                    options_.sequence_length, g_.query_heads, g_.kv_heads, g_.head_dim,
                    0, g_.rope_theta, parameters_.rope_layout(), true,
                    runtime_.stream), "Glimmer positioned RoPE backward");
            }
        }
        check(ops_.scale(b.q.get(), b.q.count(),
                         parameters_.query_scale(g_.q_scale), runtime_.stream),
              "Glimmer query scale backward");
        norm_backward(a.q_raw.get(), parameters_.q_norm(layer_index), b.q.get(),
                      b.q_raw.get(),
                      rows_ * g_.query_heads, g_.head_dim, g_.norm_eps, false);
        norm_backward(a.k_raw.get(), parameters_.k_norm(layer_index), b.k.get(),
                      b.k_raw.get(),
                      rows_ * g_.kv_heads, g_.head_dim, g_.norm_eps, false);
        linear_backward(parameters_.layer(layer_index, 4), a.input_norm.get(),
                        b.q_raw.get(), b.norm_q.get(), rows_);
        linear_backward(parameters_.layer(layer_index, 5), a.input_norm.get(),
                        b.k_raw.get(), b.norm_k.get(), rows_);
        linear_backward(parameters_.layer(layer_index, 6), a.input_norm.get(),
                        b.v.get(), b.norm_v.get(), rows_);
        linear_backward(parameters_.layer(layer_index, 7), a.input_norm.get(),
                        b.gate.get(), b.norm_gate.get(), rows_);
        check(ops_.add(b.norm_q.get(), b.norm_k.get(), b.norm_qk.get(),
                       rows_ * g_.dim, runtime_.stream), "Glimmer QK input gradient sum");
        check(ops_.add(b.norm_v.get(), b.norm_gate.get(), b.norm_vg.get(),
                       rows_ * g_.dim, runtime_.stream), "Glimmer VG input gradient sum");
        check(ops_.add(b.norm_qk.get(), b.norm_vg.get(), b.norm_all.get(),
                       rows_ * g_.dim, runtime_.stream), "Glimmer attention input gradient sum");
        norm_backward(input, &parameters_.layer(layer_index, 0), b.norm_all.get(),
                      b.input_from_norm.get(), rows_, g_.dim, g_.norm_eps,
                      centered);
        check(ops_.add(b.attention_residual.get(), b.input_from_norm.get(), b.input.get(),
                       rows_ * g_.dim, runtime_.stream), "Glimmer block residual backward");
    }

    void backward_decoder() {
        DeviceBuffer<float> grad_current(runtime_, rows_ * g_.dim);
        grad_current.copy_from(grad_final_state_);
        const std::int64_t interval = options_.activation_checkpoint_interval;
        const std::int64_t local_layers = partition_.local_layers();
        const std::int64_t segments = (local_layers + interval - 1) / interval;
        for (std::int64_t segment = segments - 1; segment >= 0; --segment) {
            const std::int64_t start =
                partition_.layer_begin + segment * interval;
            const std::int64_t end = std::min(
                partition_.layer_end, start + interval);
            std::vector<std::unique_ptr<DeviceBuffer<float>>> boundaries;
            for (std::int64_t i = start; i <= end; ++i)
                boundaries.push_back(std::make_unique<DeviceBuffer<float>>(runtime_, rows_ * g_.dim));
            boundaries.front()->copy_from(*checkpoints_.at(static_cast<std::size_t>(segment)));
            for (std::int64_t layer = start; layer < end; ++layer) {
                forward_layer(layer, boundaries[static_cast<std::size_t>(layer-start)]->get(),
                              forward_scratch_);
                runtime_.check(runtime_.memcpy(
                    boundaries[static_cast<std::size_t>(layer-start+1)]->get(),
                    forward_scratch_.output.get(), boundaries.front()->bytes(), kDeviceToDevice),
                    "cudaMemcpy recomputed activation");
            }
            for (std::int64_t layer = end - 1; layer >= start; --layer) {
                const float* input = boundaries[static_cast<std::size_t>(layer-start)]->get();
                forward_layer(layer, input, forward_scratch_);
                backward_layer(layer, input, grad_current.get(), forward_scratch_, backward_scratch_);
                runtime_.check(runtime_.memcpy(
                    grad_current.get(), backward_scratch_.input.get(), grad_current.bytes(),
                    kDeviceToDevice), "cudaMemcpy decoder gradient");
            }
        }
        grad_initial_state_.allocate(runtime_, rows_ * g_.dim);
        grad_initial_state_.copy_from(grad_current);
    }

    void embed_backward() {
        if (!parameters_.embedding().trainable) return;
        norm_backward(raw_embedding_.get(), nullptr, grad_initial_state_.get(),
                      grad_raw_embedding_.get(), rows_, g_.dim, g_.norm_eps, false);
        check(ops_.embedding_backward(
            token_ids_.get(), grad_raw_embedding_.get(), parameters_.embedding().gradient.get(),
            rows_, g_.vocab, g_.dim, runtime_.stream), "Glimmer embedding backward");
    }

    Runtime& runtime_;
    TileOps& ops_;
    Parameters& parameters_;
    LoraParameters* adapters_ = nullptr;
    Options options_;
    Geometry g_;
    PipelinePartition partition_;
    PipelineCollective* collective_ = nullptr;
    std::int64_t rows_;
    DeviceBuffer<std::int32_t> token_ids_;
    DeviceBuffer<std::int32_t> targets_;
    DeviceBuffer<float> mask_;
    DeviceBuffer<std::int32_t> sequence_ids_;
    DeviceBuffer<float> raw_embedding_;
    DeviceBuffer<float> initial_state_;
    DeviceBuffer<float> final_decoder_state_;
    DeviceBuffer<float> final_norm_;
    DeviceBuffer<float> logits_;
    DeviceBuffer<float> loss_rows_;
    DeviceBuffer<float> token_logp_;
    DeviceBuffer<float> token_entropy_;
    DeviceBuffer<float> value_;
    DeviceBuffer<float> rollout_logp_old_;
    DeviceBuffer<float> rollout_value_old_;
    DeviceBuffer<float> rollout_advantages_;
    DeviceBuffer<float> rollout_returns_;
    DeviceBuffer<float> grad_token_logp_;
    DeviceBuffer<float> grad_token_entropy_;
    DeviceBuffer<float> grad_value_;
    DeviceBuffer<float> ppo_policy_loss_;
    DeviceBuffer<float> ppo_value_loss_;
    DeviceBuffer<float> ppo_entropy_bonus_;
    DeviceBuffer<float> ppo_total_loss_;
    DeviceBuffer<float> sequence_logp_;
    DeviceBuffer<float> grad_sequence_logp_;
    DeviceBuffer<float> dpo_row_loss_;
    DeviceBuffer<float> dpo_chosen_reward_;
    DeviceBuffer<float> dpo_rejected_reward_;
    DeviceBuffer<float> reward_;
    DeviceBuffer<std::int32_t> reward_selected_positions_;
    DeviceBuffer<float> grad_reward_;
    DeviceBuffer<float> grad_logits_;
    DeviceBuffer<float> grad_raw_logits_;
    DeviceBuffer<float> grad_final_norm_;
    DeviceBuffer<float> grad_value_final_norm_;
    DeviceBuffer<float> grad_final_state_;
    DeviceBuffer<float> grad_initial_state_;
    DeviceBuffer<float> grad_raw_embedding_;
    LayerActivations forward_scratch_;
    LayerGradients backward_scratch_;
    std::vector<std::unique_ptr<DeviceBuffer<float>>> checkpoints_;
    std::int64_t optimizer_step_ = 0;
    DeviceBuffer<float> lora_input_;
    DeviceBuffer<float> lora_rank_;
    DeviceBuffer<float> lora_delta_;
    DeviceBuffer<float> lora_combined_;
    DeviceBuffer<float> lora_grad_rank_;
    DeviceBuffer<float> lora_input_grad_;
    DeviceBuffer<float> lora_dropout_grad_;
};

std::string topology_sha256(const Geometry& g) {
    std::ostringstream canonical;
    canonical << "neuralfn.muse_glimmer_native_train.topology.v1\n"
              << g.max_sequence << ',' << g.vocab << ',' << g.layers << ',' << g.dim << ','
              << g.intermediate << ',' << g.query_heads << ',' << g.kv_heads << ','
              << g.head_dim << ',' << g.window << '\n'
              << std::setprecision(17) << g.rope_theta << ',' << g.norm_eps << ','
              << g.post_norm_eps << ',' << g.q_scale << ',' << g.output_multiplier << ','
              << g.softcap << '\n'
              << "local,local,local,global;centered-sandwich-rms;weightless-qk-rms;"
                 "sigmoid-attention-gate;swiglu;untied-head;multiplier-before-softcap\n";
    const std::string value = canonical.str();
    return neuralfn::resident_support::sha256_hex(
        reinterpret_cast<const std::uint8_t*>(value.data()), value.size());
}

#pragma pack(push, 1)
struct TrainerStateV2 {
    char magic[16];
    std::uint32_t version;
    std::uint32_t header_bytes;
    std::int64_t max_sequence;
    std::int64_t vocab;
    std::int64_t layers;
    std::int64_t dim;
    std::int64_t intermediate;
    std::int64_t query_heads;
    std::int64_t kv_heads;
    std::int64_t head_dim;
    std::int64_t window;
    std::int64_t completed_step;
    std::int64_t sampler_batch;
    char model_sha256[65];
    char optimizer_sha256[65];
    char source_sha256[65];
    char tokenizer_sha256[65];
    char topology_sha256[65];
    char graph_fingerprint[65];
    char chat_template_sha256[65];
    char objective[16];
};

struct DistributedTrainerHeaderV1 {
    char magic[16];
    std::uint32_t version;
    std::uint32_t header_bytes;
    std::uint32_t entry_bytes;
    std::uint32_t reserved;
    std::int64_t world_size;
    std::int64_t max_sequence;
    std::int64_t vocab;
    std::int64_t layers;
    std::int64_t dim;
    std::int64_t intermediate;
    std::int64_t query_heads;
    std::int64_t kv_heads;
    std::int64_t head_dim;
    std::int64_t window;
    std::int64_t completed_step;
    std::int64_t sampler_batch;
    char source_sha256[65];
    char tokenizer_sha256[65];
    char topology_sha256[65];
    char graph_fingerprint[65];
    char chat_template_sha256[65];
    char objective[16];
};

struct DistributedTrainerEntryV1 {
    std::int64_t rank;
    std::int64_t layer_begin;
    std::int64_t layer_end;
    std::int64_t parameter_elements;
    std::int64_t model_bytes;
    std::int64_t optimizer_bytes;
    char model_sha256[65];
    char optimizer_sha256[65];
};

struct LoraTrainerStateV1 {
    char magic[16];
    std::uint32_t version;
    std::uint32_t header_bytes;
    std::int64_t max_sequence;
    std::int64_t vocab;
    std::int64_t layers;
    std::int64_t dim;
    std::int64_t intermediate;
    std::int64_t query_heads;
    std::int64_t kv_heads;
    std::int64_t head_dim;
    std::int64_t window;
    std::int64_t completed_step;
    std::int64_t sampler_batch;
    std::int64_t rank;
    float alpha;
    float dropout;
    std::uint64_t seed;
    std::uint32_t target_mask;
    std::uint32_t reserved;
    char adapter_sha256[65];
    char optimizer_sha256[65];
    char source_sha256[65];
    char tokenizer_sha256[65];
    char topology_sha256[65];
    char graph_fingerprint[65];
    char chat_template_sha256[65];
};

struct DpoTrainerStateV1 {
    char magic[16];
    std::uint32_t version;
    std::uint32_t header_bytes;
    std::int64_t max_sequence;
    std::int64_t vocab;
    std::int64_t layers;
    std::int64_t dim;
    std::int64_t intermediate;
    std::int64_t query_heads;
    std::int64_t kv_heads;
    std::int64_t head_dim;
    std::int64_t window;
    std::int64_t completed_step;
    std::int64_t sampler_batch;
    std::int64_t rank;
    float alpha;
    float dropout;
    std::uint64_t seed;
    std::uint32_t target_mask;
    std::uint32_t adapter_kind;
    float beta;
    float label_smoothing;
    std::uint32_t loss_type;
    std::uint32_t reserved;
    char artifact_sha256[65];
    char optimizer_sha256[65];
    char source_sha256[65];
    char reference_sha256[65];
    char tokenizer_sha256[65];
    char topology_sha256[65];
    char graph_fingerprint[65];
    char chat_template_sha256[65];
};

struct RewardTrainerStateV1 {
    char magic[16];
    std::uint32_t version;
    std::uint32_t header_bytes;
    std::int64_t max_sequence;
    std::int64_t vocab;
    std::int64_t layers;
    std::int64_t dim;
    std::int64_t intermediate;
    std::int64_t query_heads;
    std::int64_t kv_heads;
    std::int64_t head_dim;
    std::int64_t window;
    std::int64_t completed_step;
    std::int64_t sampler_batch;
    std::uint64_t reward_head_seed;
    std::uint64_t reserved;
    char model_sha256[65];
    char optimizer_sha256[65];
    char reward_head_sha256[65];
    char reward_optimizer_sha256[65];
    char source_sha256[65];
    char tokenizer_sha256[65];
    char topology_sha256[65];
    char graph_fingerprint[65];
    char chat_template_sha256[65];
};

struct PpoTrainerStateV1 {
    char magic[16];
    std::uint32_t version;
    std::uint32_t header_bytes;
    std::int64_t max_sequence;
    std::int64_t vocab;
    std::int64_t layers;
    std::int64_t dim;
    std::int64_t intermediate;
    std::int64_t query_heads;
    std::int64_t kv_heads;
    std::int64_t head_dim;
    std::int64_t window;
    std::int64_t completed_rollout;
    std::int64_t optimizer_step;
    std::int64_t sampler_batch;
    std::int64_t rollout_length;
    std::int64_t ppo_epochs_per_rollout;
    std::int64_t ppo_minibatch_size;
    std::int64_t rollout_top_k;
    std::int64_t rank;
    float alpha;
    float dropout;
    std::uint64_t lora_seed;
    std::uint64_t rollout_seed;
    std::uint64_t value_head_seed;
    std::uint32_t target_mask;
    std::uint32_t adapter_kind;
    float kl_coef;
    float ppo_clip;
    float value_coefficient;
    float entropy_coefficient;
    float gae_gamma;
    float gae_lambda;
    float rollout_temperature;
    std::uint32_t reserved;
    char policy_sha256[65];
    char policy_optimizer_sha256[65];
    char value_head_sha256[65];
    char value_optimizer_sha256[65];
    char source_sha256[65];
    char reference_sha256[65];
    char reward_manifest_sha256[65];
    char tokenizer_sha256[65];
    char topology_sha256[65];
    char graph_fingerprint[65];
    char chat_template_sha256[65];
    char rollout_contract_sha256[65];
};
#pragma pack(pop)

void copy_sha(char (&target)[65], std::string_view source, const char* label) {
    if (!valid_sha256(source)) throw std::runtime_error(std::string(label) + " is not a SHA-256 digest");
    std::memcpy(target, source.data(), 64);
    target[64] = '\0';
}

std::string state_sha(const char (&value)[65], const char* label) {
    if (value[64] != '\0') throw std::runtime_error(std::string("corrupt ") + label + " field");
    const std::string out(value, 64);
    if (!valid_sha256(out)) throw std::runtime_error(std::string("corrupt ") + label + " digest");
    return out;
}

std::string state_objective(const TrainerStateV2& state) {
    const auto end = std::find(std::begin(state.objective), std::end(state.objective), '\0');
    if (end == std::end(state.objective)) throw std::runtime_error("corrupt objective field");
    return std::string(std::begin(state.objective), end);
}

void validate_state_geometry(
    const TrainerStateV2& state,
    const Geometry& g,
    const Options& options) {
    if (std::memcmp(state.magic, "NFNGLIMMERTRAIN2", 16) != 0 || state.version != 2 ||
        state.header_bytes != sizeof(TrainerStateV2) ||
        state.max_sequence != g.max_sequence || state.vocab != g.vocab ||
        state.layers != g.layers || state.dim != g.dim ||
        state.intermediate != g.intermediate || state.query_heads != g.query_heads ||
        state.kv_heads != g.kv_heads || state.head_dim != g.head_dim ||
        state.window != g.window || state.completed_step < 0 || state.sampler_batch < 0 ||
        state_sha(state.topology_sha256, "topology") != topology_sha256(g) ||
        state_sha(state.graph_fingerprint, "graph fingerprint") !=
            (options.graph_fingerprint.empty() ? std::string(64, '0') : options.graph_fingerprint) ||
        state_objective(state) != options.objective ||
        (options.objective == "sft" &&
         state_sha(state.chat_template_sha256, "chat template") !=
             options.chat_template_sha256)) {
        throw std::runtime_error("native Glimmer resume state topology/geometry mismatch");
    }
}

TrainerStateV2 read_trainer_state(
    const fs::path& directory,
    const Geometry& g,
    const Options& options) {
    const fs::path path = directory / "trainer_state.v2";
    if (!fs::is_regular_file(path) || fs::file_size(path) != sizeof(TrainerStateV2))
        throw std::runtime_error("native Glimmer resume state is absent or has the wrong extent");
    TrainerStateV2 state{};
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char*>(&state), sizeof(state));
    if (!in) throw std::runtime_error("failed to read native Glimmer resume state");
    validate_state_geometry(state, g, options);
    return state;
}

void write_trainer_state(
    const fs::path& directory, const Geometry& g, std::int64_t step,
    std::int64_t sampler_batch, std::string_view model_sha,
    std::string_view optimizer_sha, std::string_view source_sha,
    std::string_view tokenizer_sha, const Options& options) {
    TrainerStateV2 state{};
    std::memcpy(state.magic, "NFNGLIMMERTRAIN2", 16);
    state.version = 2; state.header_bytes = sizeof(state);
    state.max_sequence = g.max_sequence; state.vocab = g.vocab; state.layers = g.layers;
    state.dim = g.dim; state.intermediate = g.intermediate;
    state.query_heads = g.query_heads; state.kv_heads = g.kv_heads;
    state.head_dim = g.head_dim; state.window = g.window;
    state.completed_step = step; state.sampler_batch = sampler_batch;
    copy_sha(state.model_sha256, model_sha, "model digest");
    copy_sha(state.optimizer_sha256, optimizer_sha, "optimizer digest");
    copy_sha(state.source_sha256, source_sha, "source digest");
    copy_sha(state.tokenizer_sha256, tokenizer_sha, "tokenizer digest");
    const std::string topology = topology_sha256(g);
    copy_sha(state.topology_sha256, topology, "topology digest");
    copy_sha(
        state.graph_fingerprint,
        options.graph_fingerprint.empty() ? std::string(64, '0') : options.graph_fingerprint,
        "graph fingerprint");
    copy_sha(
        state.chat_template_sha256,
        options.objective == "sft" ? options.chat_template_sha256 : std::string(64, '0'),
        "chat template digest");
    if (options.objective.size() >= sizeof(state.objective))
        throw std::runtime_error("native Glimmer objective is too long");
    std::memcpy(state.objective, options.objective.data(), options.objective.size());
    fs::create_directories(directory);
    const fs::path binary_tmp = directory / "trainer_state.v2.tmp";
    const fs::path binary = directory / "trainer_state.v2";
    {
        std::ofstream out(binary_tmp, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(&state), sizeof(state));
        if (!out) throw std::runtime_error("failed to write native Glimmer trainer state");
    }
    fs::rename(binary_tmp, binary);
    const fs::path json_tmp = directory / "trainer_state.json.tmp";
    const fs::path json = directory / "trainer_state.json";
    {
        std::ofstream out(json_tmp, std::ios::trunc);
        out << "{\n  \"schema\": \"neuralfn.muse_glimmer_native_training.v2\",\n"
            << "  \"completed_step\": " << step << ",\n"
            << "  \"sampler_batch\": " << sampler_batch << ",\n"
            << "  \"model_sha256\": \"" << model_sha << "\",\n"
            << "  \"optimizer_sha256\": \"" << optimizer_sha << "\",\n"
            << "  \"source_sha256\": \"" << source_sha << "\",\n"
            << "  \"tokenizer_sha256\": \"" << tokenizer_sha << "\",\n"
            << "  \"chat_template_sha256\": \""
            << (options.objective == "sft" ? options.chat_template_sha256 : "") << "\",\n"
            << "  \"topology_sha256\": \"" << topology << "\",\n"
            << "  \"graph_fingerprint\": \"" << options.graph_fingerprint << "\",\n"
            << "  \"activation_checkpointing\": true,\n"
            << "  \"objective\": \"" << options.objective << "\",\n"
            << "  \"loss\": \"masked_cross_entropy\",\n"
            << "  \"packed_sequence_boundaries\": "
            << (options.objective == "sft" ? "true" : "false") << "\n}\n";
        if (!out) throw std::runtime_error("failed to write native Glimmer JSON trainer state");
    }
    fs::rename(json_tmp, json);
}

void validate_lora_state(
    const LoraTrainerStateV1& state, const Geometry& g, const Options& options) {
    if (std::memcmp(state.magic, "NFNGLIMMERLORA1", 16) != 0 || state.version != 1 ||
        state.header_bytes != sizeof(LoraTrainerStateV1) ||
        state.max_sequence != g.max_sequence || state.vocab != g.vocab ||
        state.layers != g.layers || state.dim != g.dim ||
        state.intermediate != g.intermediate || state.query_heads != g.query_heads ||
        state.kv_heads != g.kv_heads || state.head_dim != g.head_dim ||
        state.window != g.window || state.completed_step < 0 || state.sampler_batch < 0 ||
        state.rank != options.lora_rank ||
        std::bit_cast<std::uint32_t>(state.alpha) != std::bit_cast<std::uint32_t>(options.lora_alpha) ||
        std::bit_cast<std::uint32_t>(state.dropout) !=
            std::bit_cast<std::uint32_t>(options.lora_dropout) ||
        state.seed != options.lora_seed || state.target_mask != lora_target_mask(options.lora_targets) ||
        state.reserved != (options.adapter == "qlora" ? 1U
            : options.kquant_profile == "k-quant-17gb" ? 2U
            : options.kquant_profile == "k-quant-dynamic" ? 3U
            : options.kquant_profile == "test" ? 4U : 0U) ||
        state_sha(state.topology_sha256, "topology") != topology_sha256(g) ||
        state_sha(state.graph_fingerprint, "graph fingerprint") !=
            (options.graph_fingerprint.empty() ? std::string(64, '0') : options.graph_fingerprint) ||
        state_sha(state.chat_template_sha256, "chat template") !=
            options.chat_template_sha256) {
        throw std::runtime_error("native Glimmer LoRA resume state topology/configuration mismatch");
    }
}

LoraTrainerStateV1 read_lora_trainer_state(
    const fs::path& directory, const Geometry& g, const Options& options) {
    const fs::path path = directory / "trainer_state.lora.v1";
    if (!fs::is_regular_file(path) || fs::file_size(path) != sizeof(LoraTrainerStateV1))
        throw std::runtime_error("native Glimmer LoRA resume state is absent or has the wrong extent");
    LoraTrainerStateV1 state{};
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char*>(&state), sizeof(state));
    if (!in) throw std::runtime_error("failed to read native Glimmer LoRA resume state");
    validate_lora_state(state, g, options);
    return state;
}

std::string tensor_sha256(const std::vector<std::uint16_t>& values) {
    return neuralfn::resident_support::sha256_hex(
        reinterpret_cast<const std::uint8_t*>(values.data()),
        values.size() * sizeof(std::uint16_t));
}

void write_lora_artifact_manifest(
    const fs::path& directory, const LoraParameters& adapters,
    std::string_view adapter_sha, std::string_view source_sha,
    std::string_view tokenizer_sha, const Options& options) {
    const fs::path temporary = directory / "adapter_manifest.json.tmp";
    const fs::path path = directory / "adapter_manifest.json";
    std::ofstream out(temporary, std::ios::trunc);
    if (!out) throw std::runtime_error("failed to create native Glimmer LoRA manifest");
    const std::string topology = topology_sha256(options.geometry);
    out << "{\n  \"format\": \"neuralfn.native_muse_glimmer_lora.bf16.v1\",\n"
        << "  \"architecture\": \"muse_glimmer\",\n"
        << "  \"base_weight_precision\": \""
        << (options.kquant_profile.empty() ? "bf16" : options.kquant_profile)
        << "\",\n"
        << "  \"training_base_precision\": \""
        << (!options.kquant_profile.empty()
                ? options.kquant_profile
                : options.adapter == "qlora" ? "nf4-group64-fp32-scale" : "bf16")
        << "\",\n"
        << "  \"training_adapter\": \"" << options.adapter << "\",\n"
        << "  \"layers\": " << options.geometry.layers << ",\n"
        << "  \"hidden_size\": " << options.geometry.dim << ",\n"
        << "  \"attention_size\": " << options.geometry.query_width() << ",\n"
        << "  \"kv_size\": " << options.geometry.kv_width() << ",\n"
        << "  \"intermediate_size\": " << options.geometry.intermediate << ",\n"
        << "  \"adapter_path\": \"adapter.bf16\",\n"
        << "  \"adapter_sha256\": \"" << adapter_sha << "\",\n"
        << "  \"base_sha256\": \"" << source_sha << "\",\n"
        << "  \"graph_topology_sha256\": \"" << topology << "\",\n"
        << "  \"graph_fingerprint\": \"" << options.graph_fingerprint << "\",\n"
        << "  \"tokenizer_sha256\": \"" << tokenizer_sha << "\",\n"
        << "  \"chat_template_sha256\": \"" << options.chat_template_sha256 << "\",\n"
        << "  \"rank\": " << adapters.rank() << ",\n"
        << "  \"alpha\": " << std::setprecision(9) << adapters.alpha() << ",\n"
        << "  \"scaling\": " << adapters.scaling() << ",\n"
        << "  \"dropout\": " << adapters.dropout() << ",\n"
        << "  \"seed\": " << adapters.seed() << ",\n"
        << "  \"dtype\": \"bfloat16\",\n"
        << "  \"targets\": [";
    for (std::size_t i = 0; i < options.lora_targets.size(); ++i) {
        if (i) out << ',';
        out << "\"" << options.lora_targets[i] << "\"";
    }
    out << "],\n  \"tensors\": [\n";
    std::int64_t offset = 0;
    bool first = true;
    std::vector<std::uint16_t> host;
    for (const auto& site : adapters.sites()) {
        for (const Parameter* parameter : {&site->a, &site->b}) {
            host.resize(static_cast<std::size_t>(parameter->spec.elements()));
            parameter->value.download(host.data(), parameter->spec.elements());
            if (!first) out << ",\n";
            first = false;
            out << "    {\"name\":\"" << parameter->spec.name << "\",\"rows\":"
                << parameter->spec.rows << ",\"cols\":" << parameter->spec.cols
                << ",\"byte_offset\":" << offset << ",\"nbytes\":"
                << parameter->spec.elements() * 2 << ",\"sha256\":\""
                << tensor_sha256(host) << "\"}";
            offset += parameter->spec.elements() * 2;
        }
    }
    out << "\n  ]\n}\n";
    if (!out) throw std::runtime_error("failed to write native Glimmer LoRA manifest");
    out.close();
    fs::rename(temporary, path);
}

void write_lora_trainer_state(
    const fs::path& directory, const Geometry& g, std::int64_t step,
    std::int64_t sampler_batch, std::string_view adapter_sha,
    std::string_view optimizer_sha, std::string_view source_sha,
    std::string_view tokenizer_sha, const LoraParameters& adapters,
    const Options& options) {
    LoraTrainerStateV1 state{};
    std::memcpy(state.magic, "NFNGLIMMERLORA1", 16);
    state.version = 1; state.header_bytes = sizeof(state);
    state.max_sequence = g.max_sequence; state.vocab = g.vocab; state.layers = g.layers;
    state.dim = g.dim; state.intermediate = g.intermediate;
    state.query_heads = g.query_heads; state.kv_heads = g.kv_heads;
    state.head_dim = g.head_dim; state.window = g.window;
    state.completed_step = step; state.sampler_batch = sampler_batch;
    state.rank = adapters.rank(); state.alpha = adapters.alpha();
    state.dropout = adapters.dropout(); state.seed = adapters.seed();
    state.target_mask = adapters.target_mask();
    state.reserved = options.adapter == "qlora" ? 1U
        : options.kquant_profile == "k-quant-17gb" ? 2U
        : options.kquant_profile == "k-quant-dynamic" ? 3U
        : options.kquant_profile == "test" ? 4U : 0U;
    copy_sha(state.adapter_sha256, adapter_sha, "adapter digest");
    copy_sha(state.optimizer_sha256, optimizer_sha, "adapter optimizer digest");
    copy_sha(state.source_sha256, source_sha, "source digest");
    copy_sha(state.tokenizer_sha256, tokenizer_sha, "tokenizer digest");
    const std::string topology = topology_sha256(g);
    copy_sha(state.topology_sha256, topology, "topology digest");
    copy_sha(state.graph_fingerprint,
             options.graph_fingerprint.empty() ? std::string(64, '0') : options.graph_fingerprint,
             "graph fingerprint");
    copy_sha(state.chat_template_sha256, options.chat_template_sha256, "chat template digest");
    fs::create_directories(directory);
    const fs::path binary_tmp = directory / "trainer_state.lora.v1.tmp";
    const fs::path binary = directory / "trainer_state.lora.v1";
    {
        std::ofstream out(binary_tmp, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(&state), sizeof(state));
        if (!out) throw std::runtime_error("failed to write native Glimmer LoRA trainer state");
    }
    fs::rename(binary_tmp, binary);
    const fs::path json_tmp = directory / "trainer_state.json.tmp";
    const fs::path json = directory / "trainer_state.json";
    {
        std::ofstream out(json_tmp, std::ios::trunc);
        out << "{\n  \"schema\": \"neuralfn.muse_glimmer_native_lora_training.v1\",\n"
            << "  \"completed_step\": " << step << ",\n"
            << "  \"sampler_batch\": " << sampler_batch << ",\n"
            << "  \"adapter_sha256\": \"" << adapter_sha << "\",\n"
            << "  \"optimizer_sha256\": \"" << optimizer_sha << "\",\n"
            << "  \"source_sha256\": \"" << source_sha << "\",\n"
            << "  \"tokenizer_sha256\": \"" << tokenizer_sha << "\",\n"
            << "  \"chat_template_sha256\": \"" << options.chat_template_sha256 << "\",\n"
            << "  \"topology_sha256\": \"" << topology << "\",\n"
            << "  \"graph_fingerprint\": \"" << options.graph_fingerprint << "\",\n"
            << "  \"objective\": \"sft\",\n  \"adapter\": \""
            << options.adapter << "\",\n"
            << "  \"base_weight_precision\": \""
            << (options.kquant_profile.empty() ? "bf16" : options.kquant_profile)
            << "\",\n"
            << "  \"rank\": " << adapters.rank() << ",\n"
            << "  \"alpha\": " << adapters.alpha() << ",\n"
            << "  \"scaling\": " << adapters.scaling() << ",\n"
            << "  \"dropout\": " << adapters.dropout() << ",\n"
            << "  \"seed\": " << adapters.seed() << ",\n"
            << "  \"base_frozen\": true,\n  \"adapter_only\": true\n}\n";
        if (!out) throw std::runtime_error("failed to write native Glimmer LoRA JSON state");
    }
    fs::rename(json_tmp, json);
    write_lora_artifact_manifest(
        directory, adapters, adapter_sha, source_sha, tokenizer_sha, options);
}

std::uint32_t dpo_loss_type_code(const Options& options) {
    return options.dpo_loss_type == "hinge"
        ? NFN_NATIVE_TILE_DPO_LOSS_HINGE
        : options.dpo_loss_type == "ipo"
            ? NFN_NATIVE_TILE_DPO_LOSS_IPO
            : NFN_NATIVE_TILE_DPO_LOSS_SIGMOID;
}

std::uint32_t adapter_kind_code(const Options& options) {
    if (options.kquant_profile == "k-quant-17gb") return 3U;
    if (options.kquant_profile == "k-quant-dynamic") return 4U;
    if (options.kquant_profile == "test") return 5U;
    return options.adapter == "lora" ? 1U : options.adapter == "qlora" ? 2U : 0U;
}

void validate_dpo_state(
    const DpoTrainerStateV1& state,
    const Geometry& g,
    const Options& options) {
    constexpr char kMagic[] = "NFNGLIMMERDPO1";
    const bool adapter = options.adapter != "none";
    if (std::memcmp(state.magic, kMagic, sizeof(kMagic)) != 0 ||
        state.magic[15] != '\0' || state.version != 1 ||
        state.header_bytes != sizeof(DpoTrainerStateV1) ||
        state.max_sequence != g.max_sequence || state.vocab != g.vocab ||
        state.layers != g.layers || state.dim != g.dim ||
        state.intermediate != g.intermediate ||
        state.query_heads != g.query_heads || state.kv_heads != g.kv_heads ||
        state.head_dim != g.head_dim || state.window != g.window ||
        state.completed_step < 0 || state.sampler_batch < 0 ||
        state.adapter_kind != adapter_kind_code(options) ||
        state.loss_type != dpo_loss_type_code(options) || state.reserved != 0 ||
        std::bit_cast<std::uint32_t>(state.beta) !=
            std::bit_cast<std::uint32_t>(options.dpo_beta) ||
        std::bit_cast<std::uint32_t>(state.label_smoothing) !=
            std::bit_cast<std::uint32_t>(options.dpo_label_smoothing) ||
        (adapter &&
         (state.rank != options.lora_rank ||
          std::bit_cast<std::uint32_t>(state.alpha) !=
              std::bit_cast<std::uint32_t>(options.lora_alpha) ||
          std::bit_cast<std::uint32_t>(state.dropout) !=
              std::bit_cast<std::uint32_t>(options.lora_dropout) ||
          state.seed != options.lora_seed ||
          state.target_mask != lora_target_mask(options.lora_targets))) ||
        (!adapter &&
         (state.rank != 0 || state.alpha != 0.0f || state.dropout != 0.0f ||
          state.seed != 0 || state.target_mask != 0)) ||
        state_sha(state.source_sha256, "DPO source") !=
            options.checkpoint_sha256 ||
        state_sha(state.reference_sha256, "DPO reference") !=
            options.reference_checkpoint_sha256 ||
        state_sha(state.topology_sha256, "DPO topology") != topology_sha256(g) ||
        state_sha(state.graph_fingerprint, "DPO graph fingerprint") !=
            (options.graph_fingerprint.empty() ? std::string(64, '0')
                                               : options.graph_fingerprint) ||
        state_sha(state.chat_template_sha256, "DPO chat template") !=
            options.chat_template_sha256) {
        throw std::runtime_error(
            "native Glimmer DPO resume state topology/configuration mismatch");
    }
}

DpoTrainerStateV1 read_dpo_trainer_state(
    const fs::path& directory,
    const Geometry& g,
    const Options& options) {
    const fs::path path = directory / "trainer_state.dpo.v1";
    if (!fs::is_regular_file(path) ||
        fs::file_size(path) != sizeof(DpoTrainerStateV1)) {
        throw std::runtime_error(
            "native Glimmer DPO resume state is absent or has the wrong extent");
    }
    DpoTrainerStateV1 state{};
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char*>(&state), sizeof(state));
    if (!in) throw std::runtime_error("failed to read native Glimmer DPO resume state");
    validate_dpo_state(state, g, options);
    return state;
}

void write_dpo_trainer_state(
    const fs::path& directory,
    std::int64_t step,
    std::int64_t sampler_batch,
    std::string_view artifact_sha,
    std::string_view optimizer_sha,
    std::string_view tokenizer_sha,
    const Options& options,
    const LoraParameters* adapters) {
    DpoTrainerStateV1 state{};
    constexpr char kMagic[] = "NFNGLIMMERDPO1";
    std::memcpy(state.magic, kMagic, sizeof(kMagic));
    state.version = 1;
    state.header_bytes = sizeof(state);
    const Geometry& g = options.geometry;
    state.max_sequence = g.max_sequence;
    state.vocab = g.vocab;
    state.layers = g.layers;
    state.dim = g.dim;
    state.intermediate = g.intermediate;
    state.query_heads = g.query_heads;
    state.kv_heads = g.kv_heads;
    state.head_dim = g.head_dim;
    state.window = g.window;
    state.completed_step = step;
    state.sampler_batch = sampler_batch;
    if (adapters != nullptr) {
        state.rank = adapters->rank();
        state.alpha = adapters->alpha();
        state.dropout = adapters->dropout();
        state.seed = adapters->seed();
        state.target_mask = adapters->target_mask();
    }
    state.adapter_kind = adapter_kind_code(options);
    state.beta = options.dpo_beta;
    state.label_smoothing = options.dpo_label_smoothing;
    state.loss_type = dpo_loss_type_code(options);
    copy_sha(state.artifact_sha256, artifact_sha, "DPO artifact digest");
    copy_sha(state.optimizer_sha256, optimizer_sha, "DPO optimizer digest");
    copy_sha(state.source_sha256, options.checkpoint_sha256, "DPO source digest");
    copy_sha(
        state.reference_sha256,
        options.reference_checkpoint_sha256,
        "DPO reference digest");
    copy_sha(state.tokenizer_sha256, tokenizer_sha, "DPO tokenizer digest");
    const std::string topology = topology_sha256(g);
    copy_sha(state.topology_sha256, topology, "DPO topology digest");
    copy_sha(
        state.graph_fingerprint,
        options.graph_fingerprint.empty() ? std::string(64, '0')
                                          : options.graph_fingerprint,
        "DPO graph fingerprint");
    copy_sha(
        state.chat_template_sha256,
        options.chat_template_sha256,
        "DPO chat template digest");
    fs::create_directories(directory);
    const fs::path binary_tmp = directory / "trainer_state.dpo.v1.tmp";
    const fs::path binary = directory / "trainer_state.dpo.v1";
    {
        std::ofstream out(binary_tmp, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(&state), sizeof(state));
        if (!out) throw std::runtime_error("failed to write native Glimmer DPO state");
    }
    fs::rename(binary_tmp, binary);
    const fs::path json_tmp = directory / "trainer_state.json.tmp";
    const fs::path json = directory / "trainer_state.json";
    {
        std::ofstream out(json_tmp, std::ios::trunc);
        out << "{\n  \"schema\": \"neuralfn.muse_glimmer_native_dpo_training.v1\",\n"
            << "  \"completed_step\": " << step << ",\n"
            << "  \"sampler_batch\": " << sampler_batch << ",\n"
            << "  \"artifact_sha256\": \"" << artifact_sha << "\",\n"
            << "  \"optimizer_sha256\": \"" << optimizer_sha << "\",\n"
            << "  \"source_sha256\": \"" << options.checkpoint_sha256 << "\",\n"
            << "  \"reference_sha256\": \""
            << options.reference_checkpoint_sha256 << "\",\n"
            << "  \"tokenizer_sha256\": \"" << tokenizer_sha << "\",\n"
            << "  \"chat_template_sha256\": \""
            << options.chat_template_sha256 << "\",\n"
            << "  \"topology_sha256\": \"" << topology << "\",\n"
            << "  \"graph_fingerprint\": \"" << options.graph_fingerprint << "\",\n"
            << "  \"objective\": \"dpo\",\n"
            << "  \"loss_type\": \"" << options.dpo_loss_type << "\",\n"
            << "  \"beta\": " << std::setprecision(9) << options.dpo_beta << ",\n"
            << "  \"label_smoothing\": " << options.dpo_label_smoothing << ",\n"
            << "  \"adapter\": \"" << options.adapter << "\",\n"
            << "  \"base_weight_precision\": \""
            << (options.kquant_profile.empty() ? "bf16" : options.kquant_profile)
            << "\",\n"
            << "  \"reference_frozen\": true,\n"
            << "  \"paired_sequence_masks\": true\n}\n";
        if (!out) throw std::runtime_error("failed to write native Glimmer DPO JSON state");
    }
    fs::rename(json_tmp, json);
}

RewardTrainerStateV1 read_reward_trainer_state(
    const fs::path& directory,
    const Geometry& g,
    const Options& options) {
    const fs::path path = directory / "trainer_state.reward.v1";
    if (!fs::is_regular_file(path) ||
        fs::file_size(path) != sizeof(RewardTrainerStateV1)) {
        throw std::runtime_error(
            "native Glimmer reward resume state is absent or has the wrong extent");
    }
    RewardTrainerStateV1 state{};
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char*>(&state), sizeof(state));
    constexpr char kMagic[] = "NFNGLIMMERREW1";
    if (!in || std::memcmp(state.magic, kMagic, sizeof(kMagic)) != 0 ||
        state.magic[15] != '\0' || state.version != 1 ||
        state.header_bytes != sizeof(state) || state.max_sequence != g.max_sequence ||
        state.vocab != g.vocab || state.layers != g.layers || state.dim != g.dim ||
        state.intermediate != g.intermediate ||
        state.query_heads != g.query_heads || state.kv_heads != g.kv_heads ||
        state.head_dim != g.head_dim || state.window != g.window ||
        state.completed_step < 0 || state.sampler_batch < 0 ||
        state.reward_head_seed != options.reward_head_seed || state.reserved != 0 ||
        state_sha(state.source_sha256, "reward source") !=
            options.checkpoint_sha256 ||
        state_sha(state.topology_sha256, "reward topology") != topology_sha256(g) ||
        state_sha(state.graph_fingerprint, "reward graph fingerprint") !=
            (options.graph_fingerprint.empty() ? std::string(64, '0')
                                               : options.graph_fingerprint) ||
        state_sha(state.chat_template_sha256, "reward chat template") !=
            options.chat_template_sha256) {
        throw std::runtime_error(
            "native Glimmer reward resume state topology/configuration mismatch");
    }
    return state;
}

void write_reward_trainer_state(
    const fs::path& directory,
    std::int64_t step,
    std::int64_t sampler_batch,
    std::string_view model_sha,
    std::string_view optimizer_sha,
    std::string_view reward_head_sha,
    std::string_view reward_optimizer_sha,
    std::string_view tokenizer_sha,
    const Options& options) {
    RewardTrainerStateV1 state{};
    constexpr char kMagic[] = "NFNGLIMMERREW1";
    std::memcpy(state.magic, kMagic, sizeof(kMagic));
    state.version = 1;
    state.header_bytes = sizeof(state);
    const Geometry& g = options.geometry;
    state.max_sequence = g.max_sequence;
    state.vocab = g.vocab;
    state.layers = g.layers;
    state.dim = g.dim;
    state.intermediate = g.intermediate;
    state.query_heads = g.query_heads;
    state.kv_heads = g.kv_heads;
    state.head_dim = g.head_dim;
    state.window = g.window;
    state.completed_step = step;
    state.sampler_batch = sampler_batch;
    state.reward_head_seed = options.reward_head_seed;
    copy_sha(state.model_sha256, model_sha, "reward model digest");
    copy_sha(state.optimizer_sha256, optimizer_sha, "reward optimizer digest");
    copy_sha(
        state.reward_head_sha256, reward_head_sha, "reward-head digest");
    copy_sha(
        state.reward_optimizer_sha256,
        reward_optimizer_sha,
        "reward-head optimizer digest");
    copy_sha(state.source_sha256, options.checkpoint_sha256, "reward source digest");
    copy_sha(state.tokenizer_sha256, tokenizer_sha, "reward tokenizer digest");
    const std::string topology = topology_sha256(g);
    copy_sha(state.topology_sha256, topology, "reward topology digest");
    copy_sha(
        state.graph_fingerprint,
        options.graph_fingerprint.empty() ? std::string(64, '0')
                                          : options.graph_fingerprint,
        "reward graph fingerprint");
    copy_sha(
        state.chat_template_sha256,
        options.chat_template_sha256,
        "reward chat template digest");
    fs::create_directories(directory);
    const fs::path binary_tmp = directory / "trainer_state.reward.v1.tmp";
    const fs::path binary = directory / "trainer_state.reward.v1";
    {
        std::ofstream out(binary_tmp, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(&state), sizeof(state));
        if (!out) throw std::runtime_error("failed to write native Glimmer reward state");
    }
    fs::rename(binary_tmp, binary);
    const fs::path json_tmp = directory / "trainer_state.json.tmp";
    const fs::path json = directory / "trainer_state.json";
    {
        std::ofstream out(json_tmp, std::ios::trunc);
        out << "{\n  \"schema\": \"neuralfn.muse_glimmer_native_reward_training.v1\",\n"
            << "  \"completed_step\": " << step << ",\n"
            << "  \"sampler_batch\": " << sampler_batch << ",\n"
            << "  \"model_sha256\": \"" << model_sha << "\",\n"
            << "  \"optimizer_sha256\": \"" << optimizer_sha << "\",\n"
            << "  \"reward_head_sha256\": \"" << reward_head_sha << "\",\n"
            << "  \"reward_optimizer_sha256\": \""
            << reward_optimizer_sha << "\",\n"
            << "  \"source_sha256\": \"" << options.checkpoint_sha256 << "\",\n"
            << "  \"tokenizer_sha256\": \"" << tokenizer_sha << "\",\n"
            << "  \"chat_template_sha256\": \""
            << options.chat_template_sha256 << "\",\n"
            << "  \"topology_sha256\": \"" << topology << "\",\n"
            << "  \"graph_fingerprint\": \"" << options.graph_fingerprint << "\",\n"
            << "  \"objective\": \"reward_model\",\n"
            << "  \"loss\": \"preference_bce\",\n"
            << "  \"pool\": \"last_selected_token\",\n"
            << "  \"lm_head_frozen\": true,\n"
            << "  \"reward_head_seed\": " << options.reward_head_seed << "\n}\n";
        if (!out) {
            throw std::runtime_error(
                "failed to write native Glimmer reward JSON state");
        }
    }
    fs::rename(json_tmp, json);
}

std::string ppo_rollout_contract_sha256(const Options& options) {
    std::ostringstream canonical;
    canonical << "neuralfn.muse_glimmer_native_ppo.rollout.v1\n"
              << options.rollout_length << ','
              << options.ppo_epochs_per_rollout << ','
              << options.ppo_minibatch_size << ','
              << options.rollout_top_k << ','
              << options.rollout_seed << ','
              << options.ppo_value_head_seed << '\n'
              << std::setprecision(17)
              << options.kl_coef << ',' << options.ppo_clip << ','
              << options.ppo_value_coefficient << ','
              << options.ppo_entropy_coefficient << ','
              << options.gae_gamma << ',' << options.gae_lambda << ','
              << options.rollout_temperature << '\n';
    for (std::size_t index = 0; index < options.eos_token_ids.size(); ++index) {
        if (index) canonical << ',';
        canonical << options.eos_token_ids[index];
    }
    canonical << "\nzero-bootstrap-gae;terminal-reward;sampled-policy;frozen-ref;frozen-reward\n";
    const std::string value = canonical.str();
    return neuralfn::resident_support::sha256_hex(
        reinterpret_cast<const std::uint8_t*>(value.data()), value.size());
}

PpoTrainerStateV1 read_ppo_trainer_state(
    const fs::path& directory,
    const Geometry& g,
    const Options& options) {
    const fs::path path = directory / "trainer_state.ppo.v1";
    if (!fs::is_regular_file(path) ||
        fs::file_size(path) != sizeof(PpoTrainerStateV1)) {
        throw std::runtime_error(
            "native Glimmer PPO resume state is absent or has the wrong extent");
    }
    PpoTrainerStateV1 state{};
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char*>(&state), sizeof(state));
    constexpr char kMagic[] = "NFNGLIMMERPPO1";
    const std::uint32_t expected_adapter = adapter_kind_code(options);
    if (!in || std::memcmp(state.magic, kMagic, sizeof(kMagic)) != 0 ||
        state.magic[15] != '\0' || state.version != 1 ||
        state.header_bytes != sizeof(state) || state.max_sequence != g.max_sequence ||
        state.vocab != g.vocab || state.layers != g.layers || state.dim != g.dim ||
        state.intermediate != g.intermediate ||
        state.query_heads != g.query_heads || state.kv_heads != g.kv_heads ||
        state.head_dim != g.head_dim || state.window != g.window ||
        state.completed_rollout < 0 || state.optimizer_step < 0 ||
        state.sampler_batch < 0 || state.rollout_length != options.rollout_length ||
        state.ppo_epochs_per_rollout != options.ppo_epochs_per_rollout ||
        state.ppo_minibatch_size != options.ppo_minibatch_size ||
        state.rollout_top_k != options.rollout_top_k ||
        state.rollout_seed != options.rollout_seed ||
        state.value_head_seed != options.ppo_value_head_seed ||
        state.adapter_kind != expected_adapter || state.reserved != 0 ||
        state.kl_coef != options.kl_coef || state.ppo_clip != options.ppo_clip ||
        state.value_coefficient != options.ppo_value_coefficient ||
        state.entropy_coefficient != options.ppo_entropy_coefficient ||
        state.gae_gamma != options.gae_gamma ||
        state.gae_lambda != options.gae_lambda ||
        state.rollout_temperature != options.rollout_temperature ||
        state_sha(state.source_sha256, "PPO source") != options.checkpoint_sha256 ||
        state_sha(state.reference_sha256, "PPO reference") !=
            options.reference_checkpoint_sha256 ||
        state_sha(state.reward_manifest_sha256, "PPO reward manifest") !=
            options.reward_checkpoint_sha256 ||
        state_sha(state.topology_sha256, "PPO topology") != topology_sha256(g) ||
        state_sha(state.graph_fingerprint, "PPO graph fingerprint") !=
            (options.graph_fingerprint.empty() ? std::string(64, '0')
                                               : options.graph_fingerprint) ||
        state_sha(state.chat_template_sha256, "PPO chat template") !=
            options.chat_template_sha256 ||
        state_sha(state.rollout_contract_sha256, "PPO rollout contract") !=
            ppo_rollout_contract_sha256(options)) {
        throw std::runtime_error(
            "native Glimmer PPO resume state topology/configuration mismatch");
    }
    if (expected_adapter == 0) {
        if (state.rank != 0 || state.alpha != 0.0f || state.dropout != 0.0f ||
            state.lora_seed != 0 || state.target_mask != 0) {
            throw std::runtime_error(
                "native Glimmer PPO full-model state contains adapter metadata");
        }
    } else if (state.rank != options.lora_rank || state.alpha != options.lora_alpha ||
               state.dropout != options.lora_dropout ||
               state.lora_seed != options.lora_seed ||
               state.target_mask != lora_target_mask(options.lora_targets)) {
        throw std::runtime_error(
            "native Glimmer PPO adapter state does not match the requested adapter");
    }
    return state;
}

void write_ppo_trainer_state(
    const fs::path& directory,
    std::int64_t completed_rollout,
    std::int64_t optimizer_step,
    std::int64_t sampler_batch,
    std::string_view policy_sha,
    std::string_view policy_optimizer_sha,
    std::string_view value_head_sha,
    std::string_view value_optimizer_sha,
    std::string_view tokenizer_sha,
    const Options& options,
    const LoraParameters* adapters) {
    PpoTrainerStateV1 state{};
    constexpr char kMagic[] = "NFNGLIMMERPPO1";
    std::memcpy(state.magic, kMagic, sizeof(kMagic));
    state.version = 1;
    state.header_bytes = sizeof(state);
    const Geometry& g = options.geometry;
    state.max_sequence = g.max_sequence;
    state.vocab = g.vocab;
    state.layers = g.layers;
    state.dim = g.dim;
    state.intermediate = g.intermediate;
    state.query_heads = g.query_heads;
    state.kv_heads = g.kv_heads;
    state.head_dim = g.head_dim;
    state.window = g.window;
    state.completed_rollout = completed_rollout;
    state.optimizer_step = optimizer_step;
    state.sampler_batch = sampler_batch;
    state.rollout_length = options.rollout_length;
    state.ppo_epochs_per_rollout = options.ppo_epochs_per_rollout;
    state.ppo_minibatch_size = options.ppo_minibatch_size;
    state.rollout_top_k = options.rollout_top_k;
    if (adapters != nullptr) {
        state.rank = adapters->rank();
        state.alpha = adapters->alpha();
        state.dropout = adapters->dropout();
        state.lora_seed = adapters->seed();
        state.target_mask = adapters->target_mask();
    }
    state.rollout_seed = options.rollout_seed;
    state.value_head_seed = options.ppo_value_head_seed;
    state.adapter_kind = adapter_kind_code(options);
    state.kl_coef = options.kl_coef;
    state.ppo_clip = options.ppo_clip;
    state.value_coefficient = options.ppo_value_coefficient;
    state.entropy_coefficient = options.ppo_entropy_coefficient;
    state.gae_gamma = options.gae_gamma;
    state.gae_lambda = options.gae_lambda;
    state.rollout_temperature = options.rollout_temperature;
    copy_sha(state.policy_sha256, policy_sha, "PPO policy digest");
    copy_sha(
        state.policy_optimizer_sha256,
        policy_optimizer_sha,
        "PPO policy optimizer digest");
    copy_sha(state.value_head_sha256, value_head_sha, "PPO value-head digest");
    copy_sha(
        state.value_optimizer_sha256,
        value_optimizer_sha,
        "PPO value optimizer digest");
    copy_sha(state.source_sha256, options.checkpoint_sha256, "PPO source digest");
    copy_sha(
        state.reference_sha256,
        options.reference_checkpoint_sha256,
        "PPO reference digest");
    copy_sha(
        state.reward_manifest_sha256,
        options.reward_checkpoint_sha256,
        "PPO reward manifest digest");
    copy_sha(state.tokenizer_sha256, tokenizer_sha, "PPO tokenizer digest");
    const std::string topology = topology_sha256(g);
    copy_sha(state.topology_sha256, topology, "PPO topology digest");
    copy_sha(
        state.graph_fingerprint,
        options.graph_fingerprint.empty() ? std::string(64, '0')
                                          : options.graph_fingerprint,
        "PPO graph fingerprint");
    copy_sha(
        state.chat_template_sha256,
        options.chat_template_sha256,
        "PPO chat template digest");
    copy_sha(
        state.rollout_contract_sha256,
        ppo_rollout_contract_sha256(options),
        "PPO rollout contract digest");
    fs::create_directories(directory);
    const fs::path binary_tmp = directory / "trainer_state.ppo.v1.tmp";
    const fs::path binary = directory / "trainer_state.ppo.v1";
    {
        std::ofstream out(binary_tmp, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(&state), sizeof(state));
        if (!out) throw std::runtime_error("failed to write native Glimmer PPO state");
    }
    fs::rename(binary_tmp, binary);
    const fs::path json_tmp = directory / "trainer_state.json.tmp";
    const fs::path json = directory / "trainer_state.json";
    {
        std::ofstream out(json_tmp, std::ios::trunc);
        out << "{\n  \"schema\": \"neuralfn.muse_glimmer_native_ppo_training.v1\",\n"
            << "  \"completed_rollout\": " << completed_rollout << ",\n"
            << "  \"optimizer_step\": " << optimizer_step << ",\n"
            << "  \"sampler_batch\": " << sampler_batch << ",\n"
            << "  \"policy_sha256\": \"" << policy_sha << "\",\n"
            << "  \"policy_optimizer_sha256\": \""
            << policy_optimizer_sha << "\",\n"
            << "  \"value_head_sha256\": \"" << value_head_sha << "\",\n"
            << "  \"value_optimizer_sha256\": \""
            << value_optimizer_sha << "\",\n"
            << "  \"source_sha256\": \"" << options.checkpoint_sha256 << "\",\n"
            << "  \"reference_sha256\": \""
            << options.reference_checkpoint_sha256 << "\",\n"
            << "  \"reward_manifest_sha256\": \""
            << options.reward_checkpoint_sha256 << "\",\n"
            << "  \"tokenizer_sha256\": \"" << tokenizer_sha << "\",\n"
            << "  \"chat_template_sha256\": \""
            << options.chat_template_sha256 << "\",\n"
            << "  \"rollout_contract_sha256\": \""
            << ppo_rollout_contract_sha256(options) << "\",\n"
            << "  \"objective\": \"ppo\",\n"
            << "  \"online_rollout\": true,\n"
            << "  \"reference_frozen\": true,\n"
            << "  \"reward_frozen\": true,\n"
            << "  \"adapter\": \"" << options.adapter << "\",\n"
            << "  \"base_weight_precision\": \""
            << (options.kquant_profile.empty() ? "bf16" : options.kquant_profile)
            << "\"\n}\n";
        if (!out) throw std::runtime_error("failed to write native Glimmer PPO JSON state");
    }
    fs::rename(json_tmp, json);
}

std::string dataset_tokenizer_sha(
    const neuralfn::native_train::TokenShardDataset& dataset,
    const Geometry& geometry) {
    std::string digest;
    auto inspect = [&](const neuralfn::native_train::TokenShardFile& shard) {
        if (geometry.vocab > 65536 &&
            shard.dtype != neuralfn::native_train::TokenShardDType::uint32_le)
            throw std::runtime_error("Muse Glimmer production training requires uint32 token shards");
        if (shard.tokenizer_vocab_size != 0 &&
            shard.tokenizer_vocab_size != static_cast<std::uint32_t>(geometry.vocab))
            throw std::runtime_error("Muse Glimmer shard tokenizer vocabulary mismatch");
        if (shard.tokenizer_sha256.empty())
            throw std::runtime_error("Muse Glimmer shard is missing tokenizer SHA-256 metadata");
        if (!valid_sha256(shard.tokenizer_sha256))
            throw std::runtime_error("Muse Glimmer shard tokenizer SHA-256 is malformed");
        if (digest.empty()) digest = shard.tokenizer_sha256;
        else if (digest != shard.tokenizer_sha256)
            throw std::runtime_error("Muse Glimmer shards use different tokenizers");
    };
    for (const auto& shard : dataset.train_shards) inspect(shard);
    for (const auto& shard : dataset.val_shards) inspect(shard);
    return digest;
}

std::string structured_sft_tokenizer_sha(
    const neuralfn::native_train::StructuredSftDataset& dataset,
    const Options& options) {
    std::string digest;
    auto inspect = [&](const neuralfn::native_train::StructuredSftFile& file) {
        if (file.tokenizer_vocab_size != static_cast<std::uint32_t>(options.geometry.vocab) ||
            file.sequence_length != static_cast<std::uint32_t>(options.sequence_length) ||
            file.chat_template_sha256 != options.chat_template_sha256 ||
            !valid_sha256(file.tokenizer_sha256)) {
            throw std::runtime_error(
                "structured SFT tokenizer/template/sequence geometry mismatch");
        }
        if (digest.empty()) digest = file.tokenizer_sha256;
        else if (digest != file.tokenizer_sha256)
            throw std::runtime_error("structured SFT files use different tokenizers");
    };
    for (const auto& file : dataset.train_files) inspect(file);
    for (const auto& file : dataset.val_files) inspect(file);
    return digest;
}

std::string structured_preference_tokenizer_sha(
    const neuralfn::native_train::StructuredPreferenceDataset& dataset,
    const Options& options) {
    std::string digest;
    auto inspect = [&](const neuralfn::native_train::StructuredPreferenceFile& file) {
        if (file.tokenizer_vocab_size !=
                static_cast<std::uint32_t>(options.geometry.vocab) ||
            file.sequence_length !=
                static_cast<std::uint32_t>(options.sequence_length) ||
            file.chat_template_sha256 != options.chat_template_sha256 ||
            !valid_sha256(file.tokenizer_sha256)) {
            throw std::runtime_error(
                "structured preference tokenizer/template/sequence geometry mismatch");
        }
        if (digest.empty()) digest = file.tokenizer_sha256;
        else if (digest != file.tokenizer_sha256) {
            throw std::runtime_error(
                "structured preference files use different tokenizers");
        }
    };
    for (const auto& file : dataset.train_files) inspect(file);
    for (const auto& file : dataset.val_files) inspect(file);
    return digest;
}

std::string structured_ppo_prompt_tokenizer_sha(
    const neuralfn::native_train::StructuredPpoPromptDataset& dataset,
    const Options& options) {
    std::string digest;
    auto inspect = [&](const neuralfn::native_train::StructuredPpoPromptFile& file) {
        if (file.tokenizer_vocab_size !=
                static_cast<std::uint32_t>(options.geometry.vocab) ||
            file.sequence_length !=
                static_cast<std::uint32_t>(options.sequence_length) ||
            file.chat_template_sha256 != options.chat_template_sha256 ||
            !valid_sha256(file.tokenizer_sha256)) {
            throw std::runtime_error(
                "structured PPO prompt tokenizer/template/sequence geometry mismatch");
        }
        if (digest.empty()) digest = file.tokenizer_sha256;
        else if (digest != file.tokenizer_sha256) {
            throw std::runtime_error(
                "structured PPO prompt files use different tokenizers");
        }
    };
    for (const auto& file : dataset.train_files) inspect(file);
    for (const auto& file : dataset.val_files) inspect(file);
    return digest;
}

std::string read_text_file_strict(const fs::path& path) {
    if (!fs::is_regular_file(path)) {
        throw std::runtime_error("required manifest is not a regular file: " + path.string());
    }
    std::ifstream input(path, std::ios::binary);
    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.good() && !input.eof()) {
        throw std::runtime_error("failed to read manifest: " + path.string());
    }
    return contents.str();
}

std::string exact_json_string_field(
    std::string_view payload,
    std::string_view key) {
    const std::string prefix = "\"" + std::string(key) + "\"";
    const std::size_t key_offset = payload.find(prefix);
    if (key_offset == std::string_view::npos ||
        payload.find(prefix, key_offset + prefix.size()) != std::string_view::npos) {
        throw std::runtime_error(
            "manifest must contain exactly one string field: " + std::string(key));
    }
    std::size_t cursor = key_offset + prefix.size();
    while (cursor < payload.size() &&
           (payload[cursor] == ' ' || payload[cursor] == '\t' ||
            payload[cursor] == '\r' || payload[cursor] == '\n')) ++cursor;
    if (cursor >= payload.size() || payload[cursor++] != ':') {
        throw std::runtime_error("malformed manifest field: " + std::string(key));
    }
    while (cursor < payload.size() &&
           (payload[cursor] == ' ' || payload[cursor] == '\t' ||
            payload[cursor] == '\r' || payload[cursor] == '\n')) ++cursor;
    if (cursor >= payload.size() || payload[cursor++] != '"') {
        throw std::runtime_error("manifest field is not a string: " + std::string(key));
    }
    const std::size_t end = payload.find('"', cursor);
    if (end == std::string_view::npos) {
        throw std::runtime_error("unterminated manifest string: " + std::string(key));
    }
    const std::string value(payload.substr(cursor, end - cursor));
    if (value.empty() || value.find('\\') != std::string::npos ||
        value.find('/') != std::string::npos || value.find("..") != std::string::npos) {
        throw std::runtime_error(
            "manifest field is empty, escaped, or path-unsafe: " +
            std::string(key));
    }
    return value;
}

struct FrozenRewardArtifact {
    fs::path model_path;
    fs::path head_path;
    std::string model_sha256;
    std::string head_sha256;
};

FrozenRewardArtifact resolve_frozen_reward_artifact(
    const Options& options,
    std::string_view tokenizer_sha) {
    const fs::path root = fs::weakly_canonical(fs::path(options.reward_checkpoint));
    if (!fs::is_directory(root)) {
        throw std::runtime_error(
            "native PPO reward checkpoint must be a checkpoint directory");
    }
    const fs::path manifest_path = root / "reward_model_manifest.json";
    if (sha256_file(manifest_path) != options.reward_checkpoint_sha256) {
        throw std::runtime_error(
            "native PPO reward manifest SHA-256 does not match the pinned digest");
    }
    const std::string manifest = read_text_file_strict(manifest_path);
    if (exact_json_string_field(manifest, "format") !=
            "neuralfn.native_muse_glimmer_reward.bf16.v1" ||
        exact_json_string_field(manifest, "architecture") !=
            "muse_glimmer_reward" ||
        exact_json_string_field(manifest, "tokenizer_sha256") != tokenizer_sha ||
        exact_json_string_field(manifest, "chat_template_sha256") !=
            options.chat_template_sha256 ||
        exact_json_string_field(manifest, "graph_topology_sha256") !=
            topology_sha256(options.geometry)) {
        throw std::runtime_error(
            "native PPO reward manifest lineage/topology is incompatible");
    }
    const std::string model_name =
        exact_json_string_field(manifest, "base_model_path");
    const std::string head_name =
        exact_json_string_field(manifest, "reward_head_path");
    if (model_name != "model.bf16" || head_name != "reward_head.bf16") {
        throw std::runtime_error(
            "native PPO reward manifest uses an unsupported contained layout");
    }
    FrozenRewardArtifact artifact{
        .model_path = root / model_name,
        .head_path = root / head_name,
        .model_sha256 = exact_json_string_field(manifest, "base_model_sha256"),
        .head_sha256 = exact_json_string_field(manifest, "reward_head_sha256"),
    };
    if (!valid_sha256(artifact.model_sha256) ||
        !valid_sha256(artifact.head_sha256) ||
        sha256_file(artifact.model_path) != artifact.model_sha256 ||
        sha256_file(artifact.head_path) != artifact.head_sha256) {
        throw std::runtime_error(
            "native PPO reward artifact contents fail authentication");
    }
    return artifact;
}

void save_training_checkpoint(
    Parameters& parameters, Runtime& runtime, const Options& options,
    std::int64_t step, std::int64_t sampler_batch,
    std::string_view source_sha, std::string_view tokenizer_sha) {
    const fs::path root = fs::path(options.output_dir) /
        ("checkpoint-step-" + std::to_string(step));
    const fs::path staging = root.string() + ".staging";
    if (fs::exists(staging))
        throw std::runtime_error("refusing to replace an existing Glimmer checkpoint staging directory");
    fs::create_directories(staging);
    runtime.sync();
    parameters.save_model(staging / "model.bf16");
    parameters.save_optimizer(staging / "optimizer.f32");
    const std::string model_sha = sha256_file(staging / "model.bf16");
    const std::string optimizer_sha = sha256_file(staging / "optimizer.f32");
    write_trainer_state(staging, options.geometry, step, sampler_batch,
                        model_sha, optimizer_sha, source_sha, tokenizer_sha, options);
    if (fs::exists(root)) throw std::runtime_error("refusing to replace an existing Glimmer checkpoint");
    fs::rename(staging, root);
    std::cout << "{\"event\":\"checkpoint\",\"step\":" << step
              << ",\"path\":\"" << root.string() << "\",\"model_sha256\":\""
              << model_sha << "\"}\n";
}

fs::path distributed_checkpoint_root(
    const Options& options, std::int64_t step) {
    return fs::path(options.output_dir) /
        ("checkpoint-step-" + std::to_string(step));
}

std::string distributed_rank_name(
    std::string_view kind, std::int64_t rank, std::string_view extension) {
    std::ostringstream out;
    out << kind << "-rank-" << std::setw(5) << std::setfill('0') << rank
        << extension;
    return out.str();
}

std::vector<DistributedTrainerEntryV1> read_distributed_state(
    const fs::path& directory, const Geometry& geometry,
    const Options& options, std::string_view tokenizer_sha,
    std::int64_t* completed_step,
    std::int64_t* sampler_batch) {
    const fs::path path = directory / "distributed_state.v1";
    const fs::path manifest_path = directory / "distributed_manifest.json";
    const fs::path done_path = directory / "DONE";
    if (!fs::is_regular_file(path) || !fs::is_regular_file(manifest_path) ||
        !fs::is_regular_file(done_path)) {
        throw std::runtime_error("distributed Glimmer resume state is absent");
    }
    {
        std::ifstream done(done_path);
        std::string state_digest;
        std::string manifest_digest;
        done >> state_digest >> manifest_digest;
        if (!done || state_digest != sha256_file(path) ||
            manifest_digest != sha256_file(manifest_path)) {
            throw std::runtime_error(
                "distributed Glimmer checkpoint completion marker is invalid");
        }
    }
    std::ifstream in(path, std::ios::binary);
    DistributedTrainerHeaderV1 header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || std::memcmp(header.magic, "NFNGLIMMERDIST1", 15) != 0 ||
        header.version != 1 || header.header_bytes != sizeof(header) ||
        header.entry_bytes != sizeof(DistributedTrainerEntryV1) ||
        header.reserved != 0 || header.world_size != options.pipeline_parallel_size ||
        header.max_sequence != geometry.max_sequence || header.vocab != geometry.vocab ||
        header.layers != geometry.layers || header.dim != geometry.dim ||
        header.intermediate != geometry.intermediate ||
        header.query_heads != geometry.query_heads ||
        header.kv_heads != geometry.kv_heads || header.head_dim != geometry.head_dim ||
        header.window != geometry.window || header.completed_step < 0 ||
        header.sampler_batch < 0 ||
        state_sha(header.source_sha256, "distributed source") !=
            options.checkpoint_sha256 ||
        state_sha(header.tokenizer_sha256, "distributed tokenizer") !=
            tokenizer_sha ||
        state_sha(header.topology_sha256, "distributed topology") !=
            topology_sha256(geometry) ||
        state_sha(header.graph_fingerprint, "distributed graph fingerprint") !=
            (options.graph_fingerprint.empty() ? std::string(64, '0')
                                               : options.graph_fingerprint) ||
        (options.objective == "sft" &&
         state_sha(header.chat_template_sha256, "distributed chat template") !=
             options.chat_template_sha256)) {
        throw std::runtime_error(
            "distributed Glimmer resume topology/lineage mismatch");
    }
    const auto objective_end = std::find(
        std::begin(header.objective), std::end(header.objective), '\0');
    if (objective_end == std::end(header.objective) ||
        std::string(std::begin(header.objective), objective_end) != options.objective) {
        throw std::runtime_error("distributed Glimmer resume objective mismatch");
    }
    std::vector<DistributedTrainerEntryV1> entries(
        static_cast<std::size_t>(header.world_size));
    in.read(
        reinterpret_cast<char*>(entries.data()),
        static_cast<std::streamsize>(entries.size() * sizeof(entries.front())));
    if (!in || in.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("corrupt distributed Glimmer resume state extent");
    }
    for (std::int64_t rank = 0; rank < header.world_size; ++rank) {
        const auto expected = PipelinePartition(geometry, header.world_size, rank);
        const auto& entry = entries.at(static_cast<std::size_t>(rank));
        if (entry.rank != rank || entry.layer_begin != expected.layer_begin ||
            entry.layer_end != expected.layer_end ||
            entry.parameter_elements != expected.parameter_elements(geometry) ||
            entry.model_bytes != entry.parameter_elements * 2 ||
            entry.optimizer_bytes != entry.parameter_elements * 2 *
                static_cast<std::int64_t>(sizeof(float)) ||
            !valid_sha256(state_sha(entry.model_sha256, "distributed model")) ||
            !valid_sha256(state_sha(entry.optimizer_sha256, "distributed optimizer"))) {
            throw std::runtime_error("corrupt distributed Glimmer shard table");
        }
    }
    *completed_step = header.completed_step;
    *sampler_batch = header.sampler_batch;
    return entries;
}

void save_distributed_checkpoint(
    Parameters& parameters, Runtime& runtime, PipelineCollective& collective,
    const Options& options, std::int64_t step, std::int64_t sampler_batch,
    std::string_view source_sha, std::string_view tokenizer_sha) {
    const fs::path root = distributed_checkpoint_root(options, step);
    const fs::path staging = root.string() + ".staging";
    if (collective.rank() == 0) {
        if (fs::exists(root) || fs::exists(staging)) {
            throw std::runtime_error(
                "refusing to replace an existing distributed Glimmer checkpoint");
        }
        fs::create_directories(staging);
    }
    collective.barrier();
    const std::string model_name = distributed_rank_name(
        "model", collective.rank(), ".bf16");
    const std::string optimizer_name = distributed_rank_name(
        "optimizer", collective.rank(), ".f32");
    runtime.sync();
    parameters.save_local_model(staging / model_name);
    parameters.save_local_optimizer(staging / optimizer_name);
    collective.barrier();
    if (collective.rank() == 0) {
        DistributedTrainerHeaderV1 header{};
        std::memcpy(header.magic, "NFNGLIMMERDIST1", 15);
        header.version = 1;
        header.header_bytes = sizeof(header);
        header.entry_bytes = sizeof(DistributedTrainerEntryV1);
        header.world_size = options.pipeline_parallel_size;
        header.max_sequence = options.geometry.max_sequence;
        header.vocab = options.geometry.vocab;
        header.layers = options.geometry.layers;
        header.dim = options.geometry.dim;
        header.intermediate = options.geometry.intermediate;
        header.query_heads = options.geometry.query_heads;
        header.kv_heads = options.geometry.kv_heads;
        header.head_dim = options.geometry.head_dim;
        header.window = options.geometry.window;
        header.completed_step = step;
        header.sampler_batch = sampler_batch;
        copy_sha(header.source_sha256, source_sha, "distributed source");
        copy_sha(header.tokenizer_sha256, tokenizer_sha, "distributed tokenizer");
        copy_sha(
            header.topology_sha256, topology_sha256(options.geometry),
            "distributed topology");
        copy_sha(
            header.graph_fingerprint,
            options.graph_fingerprint.empty() ? std::string(64, '0')
                                              : options.graph_fingerprint,
            "distributed graph fingerprint");
        copy_sha(
            header.chat_template_sha256,
            options.objective == "sft" ? options.chat_template_sha256
                                       : std::string(64, '0'),
            "distributed chat template");
        std::memcpy(
            header.objective, options.objective.data(), options.objective.size());
        std::vector<DistributedTrainerEntryV1> entries(
            static_cast<std::size_t>(options.pipeline_parallel_size));
        for (std::int64_t rank = 0; rank < options.pipeline_parallel_size; ++rank) {
            const PipelinePartition partition(options.geometry,
                                                options.pipeline_parallel_size, rank);
            auto& entry = entries.at(static_cast<std::size_t>(rank));
            entry.rank = rank;
            entry.layer_begin = partition.layer_begin;
            entry.layer_end = partition.layer_end;
            entry.parameter_elements = partition.parameter_elements(options.geometry);
            entry.model_bytes = entry.parameter_elements * 2;
            entry.optimizer_bytes = entry.parameter_elements * 2 *
                static_cast<std::int64_t>(sizeof(float));
            const fs::path model_path = staging /
                distributed_rank_name("model", rank, ".bf16");
            const fs::path optimizer_path = staging /
                distributed_rank_name("optimizer", rank, ".f32");
            if (!fs::is_regular_file(model_path) ||
                static_cast<std::int64_t>(fs::file_size(model_path)) !=
                    entry.model_bytes || !fs::is_regular_file(optimizer_path) ||
                static_cast<std::int64_t>(fs::file_size(optimizer_path)) !=
                    entry.optimizer_bytes) {
                throw std::runtime_error(
                    "distributed Glimmer checkpoint shard extent mismatch");
            }
            copy_sha(
                entry.model_sha256, sha256_file(model_path),
                "distributed model shard");
            copy_sha(
                entry.optimizer_sha256, sha256_file(optimizer_path),
                "distributed optimizer shard");
        }
        const fs::path state_tmp = staging / "distributed_state.v1.tmp";
        const fs::path state = staging / "distributed_state.v1";
        {
            std::ofstream out(state_tmp, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(&header), sizeof(header));
            out.write(
                reinterpret_cast<const char*>(entries.data()),
                static_cast<std::streamsize>(entries.size() * sizeof(entries.front())));
            if (!out) throw std::runtime_error("failed to write distributed Glimmer state");
        }
        fs::rename(state_tmp, state);
        const fs::path manifest_tmp = staging / "distributed_manifest.json.tmp";
        const fs::path manifest = staging / "distributed_manifest.json";
        {
            std::ofstream out(manifest_tmp, std::ios::trunc);
            out << "{\n  \"schema\": "
                << "\"neuralfn.muse_glimmer_distributed_checkpoint.v1\",\n"
                << "  \"world_size\": " << options.pipeline_parallel_size << ",\n"
                << "  \"completed_step\": " << step << ",\n"
                << "  \"sampler_batch\": " << sampler_batch << ",\n"
                << "  \"objective\": \"" << options.objective << "\",\n"
                << "  \"source_sha256\": \"" << source_sha << "\",\n"
                << "  \"tokenizer_sha256\": \"" << tokenizer_sha << "\",\n"
                << "  \"chat_template_sha256\": \""
                << (options.objective == "sft" ? options.chat_template_sha256 : "")
                << "\",\n  \"topology_sha256\": \""
                << topology_sha256(options.geometry) << "\",\n"
                << "  \"graph_fingerprint\": \"" << options.graph_fingerprint
                << "\",\n  \"activation_checkpoint_interval\": "
                << options.activation_checkpoint_interval << ",\n"
                << "  \"transport\": \"nccl\",\n"
                << "  \"stages\": [\n";
            for (std::size_t index = 0; index < entries.size(); ++index) {
                const auto& entry = entries[index];
                out << "    {\"rank\": " << entry.rank
                    << ", \"layer_begin\": " << entry.layer_begin
                    << ", \"layer_end\": " << entry.layer_end
                    << ", \"parameter_elements\": " << entry.parameter_elements
                    << ", \"model_path\": \""
                    << distributed_rank_name("model", entry.rank, ".bf16")
                    << "\", \"model_bytes\": " << entry.model_bytes
                    << ", \"model_sha256\": \"" << entry.model_sha256
                    << "\", \"optimizer_path\": \""
                    << distributed_rank_name("optimizer", entry.rank, ".f32")
                    << "\", \"optimizer_bytes\": " << entry.optimizer_bytes
                    << ", \"optimizer_sha256\": \"" << entry.optimizer_sha256
                    << "\"}" << (index + 1 == entries.size() ? "\n" : ",\n");
            }
            out << "  ]\n}\n";
            if (!out) {
                throw std::runtime_error(
                    "failed to write distributed Glimmer manifest");
            }
        }
        fs::rename(manifest_tmp, manifest);
        {
            std::ofstream done(staging / "DONE.tmp", std::ios::trunc);
            done << sha256_file(state) << ' ' << sha256_file(manifest) << '\n';
            if (!done) throw std::runtime_error("failed to write distributed DONE marker");
        }
        fs::rename(staging / "DONE.tmp", staging / "DONE");
        fs::rename(staging, root);
        std::cout << "{\"event\":\"checkpoint\",\"step\":" << step
                  << ",\"path\":\"" << root.string()
                  << "\",\"distributed\":true,\"world_size\":"
                  << options.pipeline_parallel_size << "}\n";
    }
    collective.barrier();
}

void save_lora_checkpoint(
    LoraParameters& adapters, Runtime& runtime, const Options& options,
    std::int64_t step, std::int64_t sampler_batch,
    std::string_view source_sha, std::string_view tokenizer_sha) {
    const fs::path root = fs::path(options.output_dir) /
        ("checkpoint-step-" + std::to_string(step));
    const fs::path staging = root.string() + ".staging";
    if (fs::exists(staging))
        throw std::runtime_error("refusing to replace an existing Glimmer LoRA staging directory");
    fs::create_directories(staging);
    runtime.sync();
    adapters.save(staging / "adapter.bf16");
    adapters.save_optimizer(staging / "adapter_optimizer.f32");
    const std::string adapter_sha = sha256_file(staging / "adapter.bf16");
    const std::string optimizer_sha = sha256_file(staging / "adapter_optimizer.f32");
    write_lora_trainer_state(
        staging, options.geometry, step, sampler_batch, adapter_sha,
        optimizer_sha, source_sha, tokenizer_sha, adapters, options);
    if (fs::exists(root))
        throw std::runtime_error("refusing to replace an existing Glimmer LoRA checkpoint");
    fs::rename(staging, root);
    std::cout << "{\"event\":\"checkpoint\",\"step\":" << step
              << ",\"path\":\"" << root.string() << "\",\"adapter_sha256\":\""
              << adapter_sha << "\",\"adapter_only\":true}\n";
}

void save_dpo_checkpoint(
    Parameters& parameters,
    LoraParameters* adapters,
    Runtime& runtime,
    const Options& options,
    std::int64_t step,
    std::int64_t sampler_batch,
    std::string_view tokenizer_sha) {
    const fs::path root = fs::path(options.output_dir) /
        ("checkpoint-step-" + std::to_string(step));
    const fs::path staging = root.string() + ".staging";
    if (fs::exists(staging)) {
        throw std::runtime_error(
            "refusing to replace an existing Glimmer DPO staging directory");
    }
    fs::create_directories(staging);
    runtime.sync();
    fs::path artifact_path;
    fs::path optimizer_path;
    if (adapters != nullptr) {
        artifact_path = staging / "adapter.bf16";
        optimizer_path = staging / "adapter_optimizer.f32";
        adapters->save(artifact_path);
        adapters->save_optimizer(optimizer_path);
    } else {
        artifact_path = staging / "model.bf16";
        optimizer_path = staging / "optimizer.f32";
        parameters.save_model(artifact_path);
        parameters.save_optimizer(optimizer_path);
    }
    const std::string artifact_sha = sha256_file(artifact_path);
    const std::string optimizer_sha = sha256_file(optimizer_path);
    write_dpo_trainer_state(
        staging, step, sampler_batch, artifact_sha, optimizer_sha,
        tokenizer_sha, options, adapters);
    if (adapters != nullptr) {
        write_lora_artifact_manifest(
            staging, *adapters, artifact_sha, options.checkpoint_sha256,
            tokenizer_sha, options);
    }
    if (fs::exists(root)) {
        throw std::runtime_error(
            "refusing to replace an existing Glimmer DPO checkpoint");
    }
    fs::rename(staging, root);
    std::cout << "{\"event\":\"checkpoint\",\"step\":" << step
              << ",\"path\":\"" << root.string()
              << "\",\"artifact_sha256\":\"" << artifact_sha
              << "\",\"objective\":\"dpo\",\"reference_sha256\":\""
              << options.reference_checkpoint_sha256 << "\"}\n";
}

void save_reward_checkpoint(
    Parameters& parameters,
    RewardHead& reward_head,
    Runtime& runtime,
    const Options& options,
    std::int64_t step,
    std::int64_t sampler_batch,
    std::string_view tokenizer_sha) {
    const fs::path root = fs::path(options.output_dir) /
        ("checkpoint-step-" + std::to_string(step));
    const fs::path staging = root.string() + ".staging";
    if (fs::exists(staging)) {
        throw std::runtime_error(
            "refusing to replace an existing Glimmer reward staging directory");
    }
    fs::create_directories(staging);
    runtime.sync();
    parameters.save_model(staging / "model.bf16");
    parameters.save_optimizer(staging / "optimizer.f32");
    reward_head.save(staging / "reward_head.bf16");
    reward_head.save_optimizer(staging / "reward_head_optimizer.f32");
    const std::string model_sha = sha256_file(staging / "model.bf16");
    const std::string optimizer_sha = sha256_file(staging / "optimizer.f32");
    const std::string reward_head_sha = sha256_file(staging / "reward_head.bf16");
    const std::string reward_optimizer_sha =
        sha256_file(staging / "reward_head_optimizer.f32");
    write_reward_trainer_state(
        staging, step, sampler_batch, model_sha, optimizer_sha,
        reward_head_sha, reward_optimizer_sha, tokenizer_sha, options);
    const fs::path manifest_tmp = staging / "reward_model_manifest.json.tmp";
    const fs::path manifest = staging / "reward_model_manifest.json";
    {
        std::ofstream out(manifest_tmp, std::ios::trunc);
        out << "{\n  \"format\": \"neuralfn.native_muse_glimmer_reward.bf16.v1\",\n"
            << "  \"architecture\": \"muse_glimmer_reward\",\n"
            << "  \"base_model_path\": \"model.bf16\",\n"
            << "  \"base_model_sha256\": \"" << model_sha << "\",\n"
            << "  \"pretrained_source_sha256\": \""
            << options.checkpoint_sha256 << "\",\n"
            << "  \"reward_head_path\": \"reward_head.bf16\",\n"
            << "  \"reward_head_sha256\": \"" << reward_head_sha << "\",\n"
            << "  \"hidden_size\": " << options.geometry.dim << ",\n"
            << "  \"head_bias\": false,\n"
            << "  \"pool\": \"last_selected_token\",\n"
            << "  \"tokenizer_sha256\": \"" << tokenizer_sha << "\",\n"
            << "  \"chat_template_sha256\": \""
            << options.chat_template_sha256 << "\",\n"
            << "  \"graph_topology_sha256\": \""
            << topology_sha256(options.geometry) << "\",\n"
            << "  \"graph_fingerprint\": \"" << options.graph_fingerprint
            << "\"\n}\n";
        if (!out) {
            throw std::runtime_error(
                "failed to write native Glimmer reward-model manifest");
        }
    }
    fs::rename(manifest_tmp, manifest);
    if (fs::exists(root)) {
        throw std::runtime_error(
            "refusing to replace an existing Glimmer reward checkpoint");
    }
    fs::rename(staging, root);
    std::cout << "{\"event\":\"checkpoint\",\"step\":" << step
              << ",\"path\":\"" << root.string()
              << "\",\"model_sha256\":\"" << model_sha
              << "\",\"reward_head_sha256\":\"" << reward_head_sha
              << "\",\"objective\":\"reward_model\"}\n";
}

void save_ppo_checkpoint(
    Parameters& parameters,
    LoraParameters* adapters,
    RewardHead& value_head,
    Runtime& runtime,
    const Options& options,
    std::int64_t completed_rollout,
    std::int64_t optimizer_step,
    std::int64_t sampler_batch,
    std::string_view tokenizer_sha) {
    const fs::path root = fs::path(options.output_dir) /
        ("checkpoint-rollout-" + std::to_string(completed_rollout));
    const fs::path staging = root.string() + ".staging";
    if (fs::exists(staging)) {
        throw std::runtime_error(
            "refusing to replace an existing Glimmer PPO staging directory");
    }
    fs::create_directories(staging);
    runtime.sync();
    fs::path policy_path;
    fs::path policy_optimizer_path;
    if (adapters != nullptr) {
        policy_path = staging / "adapter.bf16";
        policy_optimizer_path = staging / "adapter_optimizer.f32";
        adapters->save(policy_path);
        adapters->save_optimizer(policy_optimizer_path);
    } else {
        policy_path = staging / "model.bf16";
        policy_optimizer_path = staging / "optimizer.f32";
        parameters.save_model(policy_path);
        parameters.save_optimizer(policy_optimizer_path);
    }
    value_head.save(staging / "value_head.bf16");
    value_head.save_optimizer(staging / "value_head_optimizer.f32");
    const std::string policy_sha = sha256_file(policy_path);
    const std::string policy_optimizer_sha = sha256_file(policy_optimizer_path);
    const std::string value_head_sha = sha256_file(staging / "value_head.bf16");
    const std::string value_optimizer_sha =
        sha256_file(staging / "value_head_optimizer.f32");
    write_ppo_trainer_state(
        staging, completed_rollout, optimizer_step, sampler_batch,
        policy_sha, policy_optimizer_sha, value_head_sha, value_optimizer_sha,
        tokenizer_sha, options, adapters);
    if (adapters != nullptr) {
        write_lora_artifact_manifest(
            staging, *adapters, policy_sha, options.checkpoint_sha256,
            tokenizer_sha, options);
    }
    if (fs::exists(root)) {
        throw std::runtime_error(
            "refusing to replace an existing Glimmer PPO checkpoint");
    }
    fs::rename(staging, root);
    std::cout << "{\"event\":\"checkpoint\",\"completed_rollout\":"
              << completed_rollout << ",\"optimizer_step\":" << optimizer_step
              << ",\"path\":\"" << root.string()
              << "\",\"policy_sha256\":\"" << policy_sha
              << "\",\"value_head_sha256\":\"" << value_head_sha
              << "\",\"objective\":\"ppo\"}\n";
}

std::int32_t sample_ppo_token(
    const float* logits,
    const Options& options,
    std::mt19937_64& generator) {
    if (logits == nullptr) {
        throw std::runtime_error("native PPO sampler received null logits");
    }
    std::vector<std::int32_t> candidates;
    const std::int64_t limit = options.rollout_top_k > 0
        ? std::min(options.rollout_top_k, options.geometry.vocab)
        : options.geometry.vocab;
    candidates.resize(static_cast<std::size_t>(options.geometry.vocab));
    for (std::int64_t token = 0; token < options.geometry.vocab; ++token) {
        if (!std::isfinite(logits[token])) {
            throw std::runtime_error("native PPO policy logits are not finite");
        }
        candidates[static_cast<std::size_t>(token)] =
            static_cast<std::int32_t>(token);
    }
    if (limit < options.geometry.vocab) {
        const auto order = [&](std::int32_t lhs, std::int32_t rhs) {
            if (logits[lhs] != logits[rhs]) return logits[lhs] > logits[rhs];
            return lhs < rhs;
        };
        std::partial_sort(
            candidates.begin(), candidates.begin() + limit, candidates.end(), order);
        candidates.resize(static_cast<std::size_t>(limit));
    }
    double maximum = -std::numeric_limits<double>::infinity();
    for (const std::int32_t token : candidates) {
        maximum = std::max(
            maximum,
            static_cast<double>(logits[token]) / options.rollout_temperature);
    }
    std::vector<double> weights(candidates.size());
    double total = 0.0;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        const double weight = std::exp(
            static_cast<double>(logits[candidates[index]]) /
                options.rollout_temperature -
            maximum);
        weights[index] = weight;
        total += weight;
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        throw std::runtime_error("native PPO sampling distribution is invalid");
    }
    std::uniform_real_distribution<double> uniform(0.0, total);
    const double draw = uniform(generator);
    double cumulative = 0.0;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        cumulative += weights[index];
        if (draw <= cumulative) return candidates[index];
    }
    return candidates.back();
}

int run_ppo(
    const Options& options,
    Runtime& runtime,
    TileOps& ops,
    std::string_view topology) {
    auto dataset = neuralfn::native_train::resolve_structured_ppo_prompt_records(
        options.dataset, options.allow_train_as_validation, true);
    const std::string tokenizer_sha =
        structured_ppo_prompt_tokenizer_sha(dataset, options);
    neuralfn::native_train::SequentialStructuredPpoPromptBatchSampler sampler(
        std::move(dataset.train_files), options.batch_size);
    const std::int64_t total_batches = sampler.total_batches();
    if (total_batches <= 0) {
        throw std::runtime_error(
            "Muse Glimmer PPO prompt dataset has no complete training batch");
    }

    const bool use_lora = options.adapter != "none";
    const bool use_qlora = options.adapter == "qlora";
    const auto kquant = load_kquant_checkpoint(options);
    Parameters parameters(
        runtime, options.geometry, !use_lora, use_qlora, true, kquant);
    std::unique_ptr<LoraParameters> adapters;
    if (use_lora) {
        adapters = std::make_unique<LoraParameters>(
            runtime, options.geometry, options);
    }
    RewardHead value_head(
        runtime, options.geometry.dim, options.ppo_value_head_seed);
    std::int64_t completed_rollout = 0;
    std::int64_t optimizer_step = 0;
    std::int64_t sampler_batch = 0;
    if (!options.resume.empty()) {
        const fs::path resume_path = fs::path(options.resume);
        const PpoTrainerStateV1 state = read_ppo_trainer_state(
            resume_path, options.geometry, options);
        completed_rollout = state.completed_rollout;
        optimizer_step = state.optimizer_step;
        sampler_batch = state.sampler_batch;
        if (state_sha(state.tokenizer_sha256, "PPO tokenizer") != tokenizer_sha) {
            throw std::runtime_error(
                "PPO resume tokenizer digest does not match the prompt dataset");
        }
        if (use_lora) {
            parameters.load(options.checkpoint, options.checkpoint_sha256);
            adapters->load(
                resume_path / "adapter.bf16",
                state_sha(state.policy_sha256, "PPO adapter"));
            adapters->load_optimizer(
                resume_path / "adapter_optimizer.f32",
                state_sha(state.policy_optimizer_sha256, "PPO adapter optimizer"));
        } else {
            parameters.load(
                resume_path / "model.bf16",
                state_sha(state.policy_sha256, "PPO model"));
            parameters.load_optimizer(
                resume_path / "optimizer.f32",
                state_sha(state.policy_optimizer_sha256, "PPO optimizer"));
        }
        value_head.load(
            resume_path / "value_head.bf16",
            state_sha(state.value_head_sha256, "PPO value head"));
        value_head.load_optimizer(
            resume_path / "value_head_optimizer.f32",
            state_sha(state.value_optimizer_sha256, "PPO value optimizer"));
        if (!sampler.seek_batch(sampler_batch % total_batches)) {
            throw std::runtime_error(
                "failed to restore Muse Glimmer PPO prompt sampler cursor");
        }
    } else {
        parameters.load(options.checkpoint, options.checkpoint_sha256);
    }

    std::unique_ptr<Parameters> distinct_reference;
    Parameters* reference_parameters = nullptr;
    if (use_lora &&
        options.checkpoint_sha256 == options.reference_checkpoint_sha256) {
        reference_parameters = &parameters;
    } else {
        distinct_reference = std::make_unique<Parameters>(
            runtime, options.geometry, false, use_qlora, true, kquant);
        distinct_reference->load(
            options.reference_checkpoint,
            options.reference_checkpoint_sha256);
        reference_parameters = distinct_reference.get();
    }

    const FrozenRewardArtifact reward_artifact =
        resolve_frozen_reward_artifact(options, tokenizer_sha);
    Parameters reward_parameters(runtime, options.geometry, false, false, false);
    reward_parameters.load(
        reward_artifact.model_path, reward_artifact.model_sha256);
    RewardHead reward_head(runtime, options.geometry.dim, 0);
    reward_head.load(reward_artifact.head_path, reward_artifact.head_sha256);

    GlimmerTrainer policy(
        runtime, ops, parameters, adapters.get(), options);
    GlimmerTrainer reference(
        runtime, ops, *reference_parameters, nullptr, options);
    GlimmerTrainer reward(
        runtime, ops, reward_parameters, nullptr, options);
    neuralfn::native_train::StructuredPpoPromptBatch prompt_batch;
    if (completed_rollout >= options.max_steps) {
        std::cout << "{\"event\":\"complete\",\"completed_rollout\":"
                  << completed_rollout
                  << ",\"resumed\":true,\"objective\":\"ppo\"}\n";
        return 0;
    }
    const std::int64_t rows = options.batch_size * options.sequence_length;
    std::vector<std::int32_t> sequence_ids(static_cast<std::size_t>(rows), 0);
    for (std::int64_t rollout = completed_rollout + 1;
         rollout <= options.max_steps;
         ++rollout) {
        bool available = sampler.next(prompt_batch);
        if (!available) {
            sampler.reset();
            sampler_batch = 0;
            available = sampler.next(prompt_batch);
            if (!available) {
                throw std::runtime_error(
                    "Muse Glimmer PPO prompt sampler could not produce a batch");
            }
        }
        std::vector<std::uint32_t> tokens = prompt_batch.input_ids;
        std::vector<std::int64_t> prompt_lengths(
            static_cast<std::size_t>(options.batch_size), 0);
        std::vector<std::int64_t> current_lengths(
            static_cast<std::size_t>(options.batch_size), 0);
        std::vector<std::int64_t> action_counts(
            static_cast<std::size_t>(options.batch_size), 0);
        std::vector<bool> alive(static_cast<std::size_t>(options.batch_size), true);
        for (std::int64_t batch = 0; batch < options.batch_size; ++batch) {
            std::int64_t prompt_length = 0;
            for (std::int64_t position = 0;
                 position < options.sequence_length;
                 ++position) {
                prompt_length += prompt_batch.attention_mask[
                    static_cast<std::size_t>(
                        batch * options.sequence_length + position)] > 0.0f;
            }
            if (prompt_length <= 0 ||
                prompt_length + options.rollout_length > options.sequence_length) {
                throw std::runtime_error(
                    "native PPO prompt does not reserve the configured rollout length");
            }
            prompt_lengths[static_cast<std::size_t>(batch)] = prompt_length;
            current_lengths[static_cast<std::size_t>(batch)] = prompt_length;
        }
        std::mt19937_64 generator(splitmix64(
            options.rollout_seed ^ static_cast<std::uint64_t>(rollout)));
        std::vector<float> selected_logits;
        for (std::int64_t generated = 0;
             generated < options.rollout_length;
             ++generated) {
            std::vector<std::int64_t> positions(
                static_cast<std::size_t>(options.batch_size));
            for (std::int64_t batch = 0; batch < options.batch_size; ++batch) {
                positions[static_cast<std::size_t>(batch)] =
                    current_lengths[static_cast<std::size_t>(batch)] - 1;
            }
            policy.rollout_forward(tokens, sequence_ids, nullptr);
            policy.download_logits_at(positions, selected_logits);
            bool any_alive = false;
            for (std::int64_t batch = 0; batch < options.batch_size; ++batch) {
                const std::size_t batch_index = static_cast<std::size_t>(batch);
                if (!alive[batch_index]) continue;
                const std::int32_t token = sample_ppo_token(
                    selected_logits.data() + batch * options.geometry.vocab,
                    options, generator);
                const std::int64_t destination =
                    batch * options.sequence_length + current_lengths[batch_index];
                tokens[static_cast<std::size_t>(destination)] =
                    static_cast<std::uint32_t>(token);
                ++current_lengths[batch_index];
                ++action_counts[batch_index];
                if (std::find(
                        options.eos_token_ids.begin(),
                        options.eos_token_ids.end(), token) !=
                    options.eos_token_ids.end()) {
                    alive[batch_index] = false;
                } else {
                    any_alive = true;
                }
            }
            if (!any_alive) break;
        }

        std::vector<std::int32_t> targets(
            static_cast<std::size_t>(rows), -100);
        std::vector<float> loss_mask(static_cast<std::size_t>(rows), 0.0f);
        for (std::int64_t batch = 0; batch < options.batch_size; ++batch) {
            const std::size_t batch_index = static_cast<std::size_t>(batch);
            if (action_counts[batch_index] <= 0) {
                throw std::runtime_error(
                    "native PPO rollout produced no action for an example");
            }
            const std::int64_t action_start = prompt_lengths[batch_index] - 1;
            for (std::int64_t action = 0;
                 action < action_counts[batch_index];
                 ++action) {
                const std::int64_t position = action_start + action;
                const std::size_t row = static_cast<std::size_t>(
                    batch * options.sequence_length + position);
                targets[row] = static_cast<std::int32_t>(tokens[row + 1]);
                loss_mask[row] = 1.0f;
            }
        }

        std::vector<float> logp_old;
        std::vector<float> value_old;
        policy.evaluate_policy(
            tokens, targets, loss_mask, sequence_ids, &value_head,
            logp_old, &value_old);
        std::vector<float> reference_logp;
        reference.evaluate_policy(
            tokens, targets, loss_mask, sequence_ids, nullptr,
            reference_logp, nullptr);
        const std::vector<float> reward_scores = reward.evaluate_reward(
            tokens, targets, loss_mask, sequence_ids, reward_head);
        std::vector<float> rewards(static_cast<std::size_t>(rows), 0.0f);
        std::vector<float> advantages(static_cast<std::size_t>(rows), 0.0f);
        std::vector<float> returns(static_cast<std::size_t>(rows), 0.0f);
        for (std::int64_t row = 0; row < rows; ++row) {
            if (loss_mask[static_cast<std::size_t>(row)] > 0.0f) {
                rewards[static_cast<std::size_t>(row)] =
                    -options.kl_coef *
                    (logp_old[static_cast<std::size_t>(row)] -
                     reference_logp[static_cast<std::size_t>(row)]);
            }
        }
        for (std::int64_t batch = 0; batch < options.batch_size; ++batch) {
            const std::size_t batch_index = static_cast<std::size_t>(batch);
            const std::int64_t action_start = prompt_lengths[batch_index] - 1;
            const std::int64_t action_stop =
                action_start + action_counts[batch_index];
            const std::int64_t terminal_row =
                batch * options.sequence_length + action_stop - 1;
            rewards[static_cast<std::size_t>(terminal_row)] +=
                reward_scores[batch_index];
            double next_advantage = 0.0;
            double next_value = 0.0;
            for (std::int64_t position = action_stop - 1;
                 position >= action_start;
                 --position) {
                const std::size_t row = static_cast<std::size_t>(
                    batch * options.sequence_length + position);
                const double delta = rewards[row] +
                    options.gae_gamma * next_value - value_old[row];
                next_advantage = delta + options.gae_gamma *
                    options.gae_lambda * next_advantage;
                advantages[row] = static_cast<float>(next_advantage);
                returns[row] = static_cast<float>(next_advantage + value_old[row]);
                next_value = value_old[row];
            }
        }

        double final_loss = 0.0;
        for (std::int64_t epoch = 0;
             epoch < options.ppo_epochs_per_rollout;
             ++epoch) {
            ++optimizer_step;
            final_loss = policy.ppo_step(
                tokens, targets, loss_mask, sequence_ids, logp_old, value_old,
                advantages, returns, value_head, optimizer_step);
            std::cout << std::setprecision(10)
                      << "{\"event\":\"ppo_epoch\",\"rollout\":"
                      << rollout << ",\"epoch\":" << epoch + 1
                      << ",\"optimizer_step\":" << optimizer_step
                      << ",\"loss\":" << final_loss
                      << ",\"objective\":\"ppo\",\"adapter\":\""
                      << options.adapter << "\"}\n";
        }
        ++sampler_batch;
        std::int64_t accepted_actions = 0;
        for (const auto count : action_counts) accepted_actions += count;
        std::cout << std::setprecision(10)
                  << "{\"event\":\"rollout\",\"rollout\":" << rollout
                  << ",\"actions\":" << accepted_actions
                  << ",\"loss\":" << final_loss
                  << ",\"reference_sha256\":\""
                  << options.reference_checkpoint_sha256
                  << "\",\"reward_manifest_sha256\":\""
                  << options.reward_checkpoint_sha256
                  << "\",\"topology_sha256\":\"" << topology << "\"}\n";
        if (rollout == options.max_steps ||
            rollout % options.checkpoint_every_steps == 0) {
            save_ppo_checkpoint(
                parameters, adapters.get(), value_head, runtime, options,
                rollout, optimizer_step, sampler_batch % total_batches,
                tokenizer_sha);
        }
    }
    return 0;
}

int run_dpo(
    const Options& options,
    Runtime& runtime,
    TileOps& ops,
    std::string_view topology) {
    auto dataset = neuralfn::native_train::resolve_structured_preference_records(
        options.dataset, options.allow_train_as_validation, true);
    const std::string tokenizer_sha =
        structured_preference_tokenizer_sha(dataset, options);
    neuralfn::native_train::SequentialStructuredPreferenceBatchSampler sampler(
        std::move(dataset.train_files), options.batch_size);
    const std::int64_t total_batches = sampler.total_batches();
    if (total_batches <= 0) {
        throw std::runtime_error(
            "Muse Glimmer preference dataset has no complete training batch");
    }

    const bool use_lora = options.adapter != "none";
    const bool use_qlora = options.adapter == "qlora";
    const auto kquant = load_kquant_checkpoint(options);
    Parameters parameters(
        runtime, options.geometry, !use_lora, use_qlora, true, kquant);
    std::unique_ptr<LoraParameters> adapters;
    if (use_lora) {
        adapters = std::make_unique<LoraParameters>(
            runtime, options.geometry, options);
    }
    std::int64_t completed_step = 0;
    std::int64_t sampler_batch = 0;
    if (!options.resume.empty()) {
        const fs::path resume_path = fs::path(options.resume);
        const DpoTrainerStateV1 state = read_dpo_trainer_state(
            resume_path, options.geometry, options);
        completed_step = state.completed_step;
        sampler_batch = state.sampler_batch;
        if (state_sha(state.tokenizer_sha256, "DPO tokenizer") != tokenizer_sha) {
            throw std::runtime_error(
                "DPO resume tokenizer digest does not match the dataset");
        }
        if (use_lora) {
            parameters.load(options.checkpoint, options.checkpoint_sha256);
            adapters->load(
                resume_path / "adapter.bf16",
                state_sha(state.artifact_sha256, "DPO adapter"));
            adapters->load_optimizer(
                resume_path / "adapter_optimizer.f32",
                state_sha(state.optimizer_sha256, "DPO adapter optimizer"));
        } else {
            parameters.load(
                resume_path / "model.bf16",
                state_sha(state.artifact_sha256, "DPO model"));
            parameters.load_optimizer(
                resume_path / "optimizer.f32",
                state_sha(state.optimizer_sha256, "DPO optimizer"));
        }
        if (!sampler.seek_batch(sampler_batch % total_batches)) {
            throw std::runtime_error(
                "failed to restore Muse Glimmer DPO sampler cursor");
        }
    } else {
        parameters.load(options.checkpoint, options.checkpoint_sha256);
    }

    std::unique_ptr<Parameters> distinct_reference;
    Parameters* reference_parameters = nullptr;
    if (use_lora &&
        options.checkpoint_sha256 == options.reference_checkpoint_sha256) {
        reference_parameters = &parameters;
    } else {
        distinct_reference = std::make_unique<Parameters>(
            runtime, options.geometry, false, use_qlora, true, kquant);
        distinct_reference->load(
            options.reference_checkpoint,
            options.reference_checkpoint_sha256);
        reference_parameters = distinct_reference.get();
    }

    if (options.batch_size > std::numeric_limits<std::int64_t>::max() / 2) {
        throw std::runtime_error("native Glimmer DPO paired batch size overflows");
    }
    Options paired_options = options;
    paired_options.batch_size *= 2;
    GlimmerTrainer policy(
        runtime, ops, parameters, adapters.get(), paired_options);
    GlimmerTrainer reference(
        runtime, ops, *reference_parameters, nullptr, paired_options);
    neuralfn::native_train::StructuredPreferenceBatch batch;
    if (completed_step >= options.max_steps) {
        std::cout << "{\"event\":\"complete\",\"completed_step\":"
                  << completed_step << ",\"resumed\":true,\"objective\":\"dpo\"}\n";
        return 0;
    }

    const auto pair_vectors = [](const auto& chosen, const auto& rejected) {
        using Value = typename std::decay_t<decltype(chosen)>::value_type;
        std::vector<Value> paired;
        paired.reserve(chosen.size() + rejected.size());
        paired.insert(paired.end(), chosen.begin(), chosen.end());
        paired.insert(paired.end(), rejected.begin(), rejected.end());
        return paired;
    };
    for (std::int64_t step = completed_step + 1;
         step <= options.max_steps;
         ++step) {
        bool available = sampler.next(batch);
        if (!available) {
            sampler.reset();
            sampler_batch = 0;
            available = sampler.next(batch);
            if (!available) {
                throw std::runtime_error(
                    "Muse Glimmer preference sampler could not produce a batch");
            }
        }
        const auto tokens = pair_vectors(
            batch.chosen_input_ids, batch.rejected_input_ids);
        const auto targets = pair_vectors(
            batch.chosen_targets, batch.rejected_targets);
        const auto loss_mask = pair_vectors(
            batch.chosen_loss_mask, batch.rejected_loss_mask);
        const auto sequence_ids = pair_vectors(
            batch.chosen_sequence_ids, batch.rejected_sequence_ids);
        const double loss = policy.dpo_step(
            tokens, targets, loss_mask, sequence_ids, reference,
            options.batch_size, step);
        ++sampler_batch;
        std::cout << std::setprecision(10)
                  << "{\"event\":\"train_step\",\"step\":" << step
                  << ",\"loss\":" << loss << ",\"batch_tokens\":"
                  << tokens.size() << ",\"dtype\":\"uint32\","
                  << "\"objective\":\"dpo\",\"adapter\":\""
                  << options.adapter << "\",\"reference_sha256\":\""
                  << options.reference_checkpoint_sha256
                  << "\",\"topology_sha256\":\"" << topology << "\"}\n";
        if (step == options.max_steps ||
            step % options.checkpoint_every_steps == 0) {
            save_dpo_checkpoint(
                parameters, adapters.get(), runtime, options, step,
                sampler_batch % total_batches, tokenizer_sha);
        }
    }
    return 0;
}

int run_reward_model(
    const Options& options,
    Runtime& runtime,
    TileOps& ops,
    std::string_view topology) {
    auto dataset = neuralfn::native_train::resolve_structured_preference_records(
        options.dataset, options.allow_train_as_validation, true);
    const std::string tokenizer_sha =
        structured_preference_tokenizer_sha(dataset, options);
    neuralfn::native_train::SequentialStructuredPreferenceBatchSampler sampler(
        std::move(dataset.train_files), options.batch_size);
    const std::int64_t total_batches = sampler.total_batches();
    if (total_batches <= 0) {
        throw std::runtime_error(
            "Muse Glimmer preference dataset has no complete reward batch");
    }
    Parameters parameters(runtime, options.geometry, true, false, false);
    RewardHead reward_head(
        runtime, options.geometry.dim, options.reward_head_seed);
    std::int64_t completed_step = 0;
    std::int64_t sampler_batch = 0;
    if (!options.resume.empty()) {
        const fs::path resume_path = fs::path(options.resume);
        const RewardTrainerStateV1 state = read_reward_trainer_state(
            resume_path, options.geometry, options);
        completed_step = state.completed_step;
        sampler_batch = state.sampler_batch;
        if (state_sha(state.tokenizer_sha256, "reward tokenizer") != tokenizer_sha) {
            throw std::runtime_error(
                "reward resume tokenizer digest does not match the dataset");
        }
        parameters.load(
            resume_path / "model.bf16",
            state_sha(state.model_sha256, "reward model"));
        parameters.load_optimizer(
            resume_path / "optimizer.f32",
            state_sha(state.optimizer_sha256, "reward optimizer"));
        reward_head.load(
            resume_path / "reward_head.bf16",
            state_sha(state.reward_head_sha256, "reward head"));
        reward_head.load_optimizer(
            resume_path / "reward_head_optimizer.f32",
            state_sha(state.reward_optimizer_sha256, "reward-head optimizer"));
        if (!sampler.seek_batch(sampler_batch % total_batches)) {
            throw std::runtime_error(
                "failed to restore Muse Glimmer reward sampler cursor");
        }
    } else {
        parameters.load(options.checkpoint, options.checkpoint_sha256);
    }
    if (options.batch_size > std::numeric_limits<std::int64_t>::max() / 2) {
        throw std::runtime_error("native Glimmer reward paired batch size overflows");
    }
    Options paired_options = options;
    paired_options.batch_size *= 2;
    GlimmerTrainer trainer(runtime, ops, parameters, nullptr, paired_options);
    neuralfn::native_train::StructuredPreferenceBatch batch;
    if (completed_step >= options.max_steps) {
        std::cout << "{\"event\":\"complete\",\"completed_step\":"
                  << completed_step
                  << ",\"resumed\":true,\"objective\":\"reward_model\"}\n";
        return 0;
    }
    const auto pair_vectors = [](const auto& chosen, const auto& rejected) {
        using Value = typename std::decay_t<decltype(chosen)>::value_type;
        std::vector<Value> paired;
        paired.reserve(chosen.size() + rejected.size());
        paired.insert(paired.end(), chosen.begin(), chosen.end());
        paired.insert(paired.end(), rejected.begin(), rejected.end());
        return paired;
    };
    for (std::int64_t step = completed_step + 1;
         step <= options.max_steps;
         ++step) {
        bool available = sampler.next(batch);
        if (!available) {
            sampler.reset();
            sampler_batch = 0;
            available = sampler.next(batch);
            if (!available) {
                throw std::runtime_error(
                    "Muse Glimmer reward sampler could not produce a batch");
            }
        }
        const auto tokens = pair_vectors(
            batch.chosen_input_ids, batch.rejected_input_ids);
        const auto targets = pair_vectors(
            batch.chosen_targets, batch.rejected_targets);
        const auto loss_mask = pair_vectors(
            batch.chosen_loss_mask, batch.rejected_loss_mask);
        const auto sequence_ids = pair_vectors(
            batch.chosen_sequence_ids, batch.rejected_sequence_ids);
        const double loss = trainer.reward_step(
            tokens, targets, loss_mask, sequence_ids, reward_head,
            options.batch_size, step);
        ++sampler_batch;
        std::cout << std::setprecision(10)
                  << "{\"event\":\"train_step\",\"step\":" << step
                  << ",\"loss\":" << loss << ",\"batch_tokens\":"
                  << tokens.size() << ",\"dtype\":\"uint32\","
                  << "\"objective\":\"reward_model\",\"adapter\":\"none\","
                  << "\"topology_sha256\":\"" << topology << "\"}\n";
        if (step == options.max_steps ||
            step % options.checkpoint_every_steps == 0) {
            save_reward_checkpoint(
                parameters, reward_head, runtime, options, step,
                sampler_batch % total_batches, tokenizer_sha);
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto layout = parameter_specs(options.geometry);
        if (options.print_layout) {
            std::int64_t offset = 0;
            std::cout << "{\"schema\":\"neuralfn.muse_glimmer_parameter_layout.v1\",\"tensor_count\":"
                      << layout.size() << ",\"parameter_count\":"
                      << parameter_elements(options.geometry) << ",\"tensors\":[";
            for (std::size_t index = 0; index < layout.size(); ++index) {
                if (index) std::cout << ',';
                std::cout << "{\"name\":\"" << layout[index].name << "\",\"rows\":"
                          << layout[index].rows << ",\"cols\":" << layout[index].cols
                          << ",\"element_offset\":" << offset << "}";
                offset += layout[index].elements();
            }
            std::cout << "]}\n";
            return 0;
        }
        const std::string topology = topology_sha256(options.geometry);
        if (options.distributed_plan) {
            print_pipeline_memory_plan(options);
            return 0;
        }
        if (!options.graph_fingerprint.empty() && !valid_sha256(options.graph_fingerprint))
            throw std::runtime_error("--graph-fingerprint must be a lowercase SHA-256 digest");
        if (!options.tiny_geometry && !options.kernel_check && !options.print_layout &&
            options.graph_fingerprint.empty())
            throw std::runtime_error("production Muse Glimmer training requires --graph-fingerprint");
        TileOps ops(options.tile_ops_lib);
        if (options.kernel_check) {
            std::cout
                << "{\"schema\":\"neuralfn.muse_glimmer_native_training_kernel_check.v1\","
                << "\"passed\":true,\"training_abi\":1,\"topology_sha256\":\""
                << topology << "\",\"required_symbols\":31,\"native_lora\":true,"
                << "\"native_qlora_nf4_group64\":true,\"native_dpo\":true,"
                << "\"native_kquant_lora\":true,"
                << "\"kquant_profiles\":[\"k-quant-17gb\",\"k-quant-dynamic\"],"
                << "\"native_reward_model\":true,\"native_ppo\":true,"
                << "\"pipeline_parallel\":true,\"pipeline_transport\":\"nccl\"}\n";
            return 0;
        }
        if (options.dataset.empty()) throw std::runtime_error("--dataset is required");
        if ((options.objective == "dpo" || options.objective == "ppo" ||
             options.resume.empty() ||
             options.adapter != "none") &&
            (options.checkpoint.empty() || options.checkpoint_sha256.empty()))
            throw std::runtime_error("--checkpoint and --checkpoint-sha256 are required for a new run");

        if (options.pipeline_parallel_size > 1 &&
            !valid_sha256(options.checkpoint_sha256)) {
            throw std::runtime_error(
                "pipeline-parallel training requires the original "
                "--checkpoint-sha256 lineage pin");
        }
        Runtime runtime(options.cuda_runtime_lib, options.cuda_device);
        if (options.objective == "dpo") {
            return run_dpo(options, runtime, ops, topology);
        }
        if (options.objective == "reward_model") {
            return run_reward_model(options, runtime, ops, topology);
        }
        if (options.objective == "ppo") {
            return run_ppo(options, runtime, ops, topology);
        }

        std::unique_ptr<neuralfn::native_train::SequentialTokenBatchSampler32> ar_sampler;
        std::unique_ptr<neuralfn::native_train::SequentialStructuredSftBatchSampler> sft_sampler;
        std::string tokenizer_sha;
        if (options.objective == "sft") {
            auto dataset = neuralfn::native_train::resolve_structured_sft_records(
                options.dataset, options.allow_train_as_validation, true);
            tokenizer_sha = structured_sft_tokenizer_sha(dataset, options);
            sft_sampler = std::make_unique<
                neuralfn::native_train::SequentialStructuredSftBatchSampler>(
                    std::move(dataset.train_files), options.batch_size);
        } else {
            auto dataset = neuralfn::native_train::resolve_token_shards(
                options.dataset, options.allow_train_as_validation, true);
            tokenizer_sha = dataset_tokenizer_sha(dataset, options.geometry);
            ar_sampler = std::make_unique<
                neuralfn::native_train::SequentialTokenBatchSampler32>(
                    std::move(dataset.train_shards), options.sequence_length,
                    options.batch_size);
        }
        const auto total_batches = [&]() -> std::int64_t {
            return sft_sampler ? sft_sampler->total_batches() : ar_sampler->total_batches();
        };
        if (total_batches() <= 0)
            throw std::runtime_error("Muse Glimmer dataset has no complete training batch");

        if (options.pipeline_parallel_size > 1) {
            const PipelineMemoryPlan memory_plan = pipeline_memory_plan(
                options.geometry, options, options.pipeline_parallel_rank);
            bool local_fit = false;
            std::size_t free_bytes = 0;
            std::size_t total_bytes = 0;
            std::string memory_error;
            try {
                std::tie(free_bytes, total_bytes) = runtime.memory_info();
                const auto reserve = static_cast<std::uint64_t>(
                    options.distributed_reserve_bytes);
                local_fit = memory_plan.required_bytes <=
                    std::numeric_limits<std::uint64_t>::max() - reserve &&
                    static_cast<std::uint64_t>(free_bytes) >=
                        memory_plan.required_bytes + reserve;
                if (!local_fit) {
                    std::ostringstream message;
                    message << "rank " << options.pipeline_parallel_rank
                            << " has " << free_bytes << " free bytes but requires "
                            << memory_plan.required_bytes << " plus " << reserve
                            << " reserve bytes";
                    memory_error = message.str();
                }
            } catch (const std::exception& exc) {
                memory_error = exc.what();
            }
            PipelineCollective collective(runtime, options);
            const float failed_ranks = collective.all_reduce_sum(
                local_fit ? 0.0f : 1.0f);
            if (failed_ranks != 0.0f) {
                throw std::runtime_error(
                    local_fit
                        ? "one or more pipeline ranks failed the per-device VRAM admission gate"
                        : "pipeline VRAM admission failed: " + memory_error);
            }
            std::cout
                << "{\"event\":\"pipeline_admission\",\"rank\":"
                << options.pipeline_parallel_rank << ",\"cuda_device\":"
                << options.cuda_device << ",\"free_bytes\":" << free_bytes
                << ",\"total_bytes\":" << total_bytes
                << ",\"required_bytes_before_reserve\":"
                << memory_plan.required_bytes << ",\"reserve_bytes\":"
                << options.distributed_reserve_bytes << ",\"layer_begin\":"
                << memory_plan.partition.layer_begin << ",\"layer_end\":"
                << memory_plan.partition.layer_end << "}\n";

            Parameters parameters(
                runtime, options.geometry, true, false, true, nullptr,
                &memory_plan.partition);
            std::int64_t completed_step = 0;
            std::int64_t sampler_batch = 0;
            if (!options.resume.empty()) {
                const fs::path resume_path(options.resume);
                const auto entries = read_distributed_state(
                    resume_path, options.geometry, options, tokenizer_sha,
                    &completed_step, &sampler_batch);
                const auto& entry = entries.at(
                    static_cast<std::size_t>(options.pipeline_parallel_rank));
                const fs::path model_path = resume_path /
                    distributed_rank_name(
                        "model", options.pipeline_parallel_rank, ".bf16");
                const fs::path optimizer_path = resume_path /
                    distributed_rank_name(
                        "optimizer", options.pipeline_parallel_rank, ".f32");
                bool authenticated = false;
                try {
                    authenticated =
                        fs::is_regular_file(model_path) &&
                        static_cast<std::int64_t>(fs::file_size(model_path)) ==
                            entry.model_bytes &&
                        sha256_file(model_path) ==
                            state_sha(entry.model_sha256, "pipeline model") &&
                        fs::is_regular_file(optimizer_path) &&
                        static_cast<std::int64_t>(fs::file_size(optimizer_path)) ==
                            entry.optimizer_bytes &&
                        sha256_file(optimizer_path) ==
                            state_sha(entry.optimizer_sha256, "pipeline optimizer");
                } catch (...) {
                    authenticated = false;
                }
                if (collective.all_reduce_sum(authenticated ? 0.0f : 1.0f) != 0.0f) {
                    throw std::runtime_error(
                        "one or more pipeline resume shards failed authentication");
                }
                parameters.load_local_model(
                    model_path,
                    state_sha(entry.model_sha256, "pipeline model"), false);
                parameters.load_local_optimizer(
                    optimizer_path,
                    state_sha(entry.optimizer_sha256, "pipeline optimizer"), false);
            } else {
                bool source_authenticated = true;
                if (collective.rank() == 0) {
                    try {
                        source_authenticated =
                            fs::is_regular_file(options.checkpoint) &&
                            fs::file_size(options.checkpoint) ==
                                static_cast<std::uintmax_t>(
                                    parameter_elements(options.geometry)) * 2U &&
                            sha256_file(options.checkpoint) ==
                                options.checkpoint_sha256;
                    } catch (...) {
                        source_authenticated = false;
                    }
                }
                if (collective.all_reduce_sum(
                        source_authenticated ? 0.0f : 1.0f) != 0.0f) {
                    throw std::runtime_error(
                        "pipeline source BF16 checkpoint authentication failed");
                }
                parameters.load(
                    options.checkpoint, options.checkpoint_sha256, false);
            }
            const bool restored = completed_step == 0 ||
                (sft_sampler
                    ? sft_sampler->seek_batch(sampler_batch % total_batches())
                    : ar_sampler->seek_batch(sampler_batch % total_batches()));
            if (!restored) {
                throw std::runtime_error(
                    "failed to restore distributed Glimmer sampler cursor");
            }
            GlimmerTrainer trainer(
                runtime, ops, parameters, nullptr, options, &collective);
            neuralfn::native_train::TokenBatch32 ar_batch;
            neuralfn::native_train::StructuredSftBatch sft_batch;
            if (completed_step >= options.max_steps) {
                if (collective.rank() == 0) {
                    std::cout << "{\"event\":\"complete\",\"completed_step\":"
                              << completed_step
                              << ",\"resumed\":true,\"distributed\":true}\n";
                }
                return 0;
            }
            for (std::int64_t step = completed_step + 1;
                 step <= options.max_steps; ++step) {
                bool available = sft_sampler
                    ? sft_sampler->next(sft_batch) : ar_sampler->next(ar_batch);
                if (!available) {
                    if (sft_sampler) sft_sampler->reset();
                    else ar_sampler->reset();
                    sampler_batch = 0;
                    available = sft_sampler
                        ? sft_sampler->next(sft_batch) : ar_sampler->next(ar_batch);
                    if (!available) {
                        throw std::runtime_error(
                            "distributed Glimmer sampler could not produce a batch");
                    }
                }
                std::vector<std::int32_t> ar_targets;
                if (!sft_sampler) {
                    ar_targets.reserve(ar_batch.targets.size());
                    for (const std::uint32_t target : ar_batch.targets) {
                        if (target > static_cast<std::uint32_t>(
                                std::numeric_limits<std::int32_t>::max())) {
                            throw std::runtime_error(
                                "AR target does not fit the native i32 loss ABI");
                        }
                        ar_targets.push_back(static_cast<std::int32_t>(target));
                    }
                }
                const double loss = sft_sampler
                    ? trainer.pipeline_train_step(
                          sft_batch.input_ids, sft_batch.targets,
                          &sft_batch.loss_mask, &sft_batch.sequence_ids, step)
                    : trainer.pipeline_train_step(
                          ar_batch.tokens, ar_targets, nullptr, nullptr, step);
                ++sampler_batch;
                if (collective.rank() == 0) {
                    std::cout << std::setprecision(10)
                              << "{\"event\":\"train_step\",\"step\":" << step
                              << ",\"loss\":" << loss
                              << ",\"objective\":\"" << options.objective
                              << "\",\"pipeline_parallel_size\":"
                              << options.pipeline_parallel_size
                              << ",\"topology_sha256\":\"" << topology
                              << "\"}\n";
                }
                if (step == options.max_steps ||
                    step % options.checkpoint_every_steps == 0) {
                    save_distributed_checkpoint(
                        parameters, runtime, collective, options, step,
                        sampler_batch % total_batches(),
                        options.checkpoint_sha256, tokenizer_sha);
                }
            }
            return 0;
        }

        const bool use_lora = options.adapter != "none";
        const bool use_qlora = options.adapter == "qlora";
        const auto kquant = load_kquant_checkpoint(options);
        Parameters parameters(
            runtime, options.geometry, !use_lora, use_qlora, true, kquant);
        std::unique_ptr<LoraParameters> adapters;
        if (use_lora)
            adapters = std::make_unique<LoraParameters>(runtime, options.geometry, options);
        std::int64_t completed_step = 0;
        std::int64_t sampler_batch = 0;
        std::string source_sha = options.checkpoint_sha256;
        if (!options.resume.empty()) {
            const fs::path resume_path = fs::path(options.resume);
            if (use_lora) {
                const LoraTrainerStateV1 state = read_lora_trainer_state(
                    resume_path, options.geometry, options);
                completed_step = state.completed_step;
                sampler_batch = state.sampler_batch;
                source_sha = state_sha(state.source_sha256, "source");
                if (options.checkpoint_sha256 != source_sha)
                    throw std::runtime_error("LoRA resume base checkpoint digest conflicts with CLI pin");
                if (state_sha(state.tokenizer_sha256, "tokenizer") != tokenizer_sha)
                    throw std::runtime_error("LoRA resume tokenizer digest does not match the dataset");
                parameters.load(options.checkpoint, options.checkpoint_sha256);
                adapters->load(
                    resume_path / "adapter.bf16", state_sha(state.adapter_sha256, "adapter"));
                adapters->load_optimizer(
                    resume_path / "adapter_optimizer.f32",
                    state_sha(state.optimizer_sha256, "adapter optimizer"));
            } else {
                const TrainerStateV2 state = read_trainer_state(
                    resume_path, options.geometry, options);
                completed_step = state.completed_step;
                sampler_batch = state.sampler_batch;
                source_sha = state_sha(state.source_sha256, "source");
                if (!options.checkpoint_sha256.empty() && options.checkpoint_sha256 != source_sha)
                    throw std::runtime_error("resume source checkpoint digest conflicts with CLI pin");
                if (state_sha(state.tokenizer_sha256, "tokenizer") != tokenizer_sha)
                    throw std::runtime_error("resume tokenizer digest does not match the dataset");
                parameters.load(
                    resume_path / "model.bf16", state_sha(state.model_sha256, "model"));
                parameters.load_optimizer(
                    resume_path / "optimizer.f32", state_sha(state.optimizer_sha256, "optimizer"));
            }
            const bool restored = sft_sampler
                ? sft_sampler->seek_batch(sampler_batch % total_batches())
                : ar_sampler->seek_batch(sampler_batch % total_batches());
            if (!restored)
                throw std::runtime_error("failed to restore Muse Glimmer sampler cursor");
        } else {
            parameters.load(options.checkpoint, options.checkpoint_sha256);
        }
        GlimmerTrainer trainer(runtime, ops, parameters, adapters.get(), options);
        neuralfn::native_train::TokenBatch32 ar_batch;
        neuralfn::native_train::StructuredSftBatch sft_batch;
        if (completed_step >= options.max_steps) {
            std::cout << "{\"event\":\"complete\",\"completed_step\":"
                      << completed_step << ",\"resumed\":true}\n";
            return 0;
        }
        for (std::int64_t step = completed_step + 1; step <= options.max_steps; ++step) {
            bool available = sft_sampler ? sft_sampler->next(sft_batch) : ar_sampler->next(ar_batch);
            if (!available) {
                if (sft_sampler) sft_sampler->reset();
                else ar_sampler->reset();
                sampler_batch = 0;
                available = sft_sampler ? sft_sampler->next(sft_batch) : ar_sampler->next(ar_batch);
                if (!available)
                    throw std::runtime_error("Muse Glimmer sampler could not produce a batch");
            }
            std::vector<std::int32_t> ar_targets;
            if (!sft_sampler) {
                ar_targets.reserve(ar_batch.targets.size());
                for (std::uint32_t target : ar_batch.targets) {
                    if (target > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
                        throw std::runtime_error("AR target does not fit the native i32 loss ABI");
                    ar_targets.push_back(static_cast<std::int32_t>(target));
                }
            }
            const double loss = sft_sampler
                ? trainer.train_step(
                      sft_batch.input_ids, sft_batch.targets, &sft_batch.loss_mask,
                      &sft_batch.sequence_ids, step)
                : trainer.train_step(ar_batch.tokens, ar_targets, nullptr, nullptr, step);
            ++sampler_batch;
            std::cout << std::setprecision(10)
                      << "{\"event\":\"train_step\",\"step\":" << step
                      << ",\"loss\":" << loss << ",\"batch_tokens\":"
                      << (sft_sampler ? sft_batch.input_ids.size() : ar_batch.tokens.size())
                      << ",\"dtype\":\"uint32\",\"objective\":\""
                      << options.objective << "\","
                      << "\"adapter\":\"" << options.adapter << "\","
                      << "\"base_weight_precision\":\""
                      << (options.kquant_profile.empty() ? "bf16"
                                                         : options.kquant_profile)
                      << "\","
                      << "\"topology_sha256\":\"" << topology << "\"}\n";
            if (step == options.max_steps || step % options.checkpoint_every_steps == 0) {
                if (use_lora) {
                    save_lora_checkpoint(
                        *adapters, runtime, options, step,
                        sampler_batch % total_batches(), source_sha, tokenizer_sha);
                } else {
                    save_training_checkpoint(
                        parameters, runtime, options, step,
                        sampler_batch % total_batches(), source_sha, tokenizer_sha);
                }
            }
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return 2;
    }
}
