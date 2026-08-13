#include "resident_glimmer.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::int64_t> tokens(const std::string& text) {
    std::vector<std::int64_t> result;
    std::stringstream input(text);
    std::string item;
    while (std::getline(input, item, ',')) {
        if (!item.empty()) {
            result.push_back(std::stoll(item));
        }
    }
    return result;
}

void print_logits(const std::vector<float>& logits) {
    std::cout.precision(9);
    for (std::size_t index = 0; index < logits.size(); ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << logits[index];
    }
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 5 || argc > 7) {
            throw std::runtime_error(
                "usage: probe CHECKPOINT SHA off|full TOKENS [TRUNCATE APPEND]");
        }
        neuralfn::resident_glimmer::GlimmerInferenceConfig config;
        config.max_seq_len = 8;
        config.vocab_size = 13;
        config.num_layers = 4;
        config.model_dim = 8;
        config.intermediate_dim = 16;
        config.num_heads = 2;
        config.num_kv_heads = 1;
        config.head_dim = 2;
        config.sliding_window = 3;
        config.rope_theta = 500000.0;
        config.norm_eps = 1.0e-5;
        config.post_norm_eps = 1.0e-8;
        config.q_scale_factor = 3.87;
        config.output_multiplier = 0.19611613513818404;
        config.logit_softcap = 20.0;
        config.container = neuralfn::resident_glimmer::WeightContainer::NativeBf16;
        config.checkpoint_sha256 = argv[2];
        const char* cuda_runtime = std::getenv("NFN_TEST_GLIMMER_CUDA_RUNTIME");
        const char* tile_ops = std::getenv("NFN_TEST_GLIMMER_TILE_OPS");
        if ((cuda_runtime == nullptr) != (tile_ops == nullptr)) {
            throw std::runtime_error("both fake Glimmer CUDA libraries must be configured");
        }
        if (cuda_runtime != nullptr) {
            config.whole_model_cuda = true;
            config.cuda_device = 0;
            config.cuda_runtime_lib = cuda_runtime;
            config.tile_ops_lib = tile_ops;
        }
        auto model = neuralfn::resident_glimmer::GlimmerModel::load(argv[1], config);
        const char* lora_path = std::getenv("NFN_TEST_GLIMMER_LORA");
        if (lora_path != nullptr) {
            const char* lora_sha = std::getenv("NFN_TEST_GLIMMER_LORA_SHA256");
            const char* lora_rank = std::getenv("NFN_TEST_GLIMMER_LORA_RANK");
            const char* lora_alpha = std::getenv("NFN_TEST_GLIMMER_LORA_ALPHA");
            const char* lora_mask = std::getenv("NFN_TEST_GLIMMER_LORA_TARGET_MASK");
            if (lora_sha == nullptr || lora_rank == nullptr ||
                lora_alpha == nullptr || lora_mask == nullptr) {
                throw std::runtime_error("all Glimmer LoRA test fields must be configured");
            }
            model->load_lora_adapter(
                lora_path,
                lora_sha,
                std::stoll(lora_rank),
                std::stod(lora_alpha),
                static_cast<std::uint32_t>(std::stoul(lora_mask)));
        }
        const auto mode = std::string(argv[3]) == "full"
            ? neuralfn::resident_dense::KVCacheMode::Full
            : neuralfn::resident_dense::KVCacheMode::Off;
        auto session = model->create_session(17, mode);
        const auto prompt = tokens(argv[4]);
        session->prefill(prompt, 0);
        if (argc == 7) {
            const std::int64_t count = std::stoll(argv[5]);
            session->truncate(count);
            const auto suffix = tokens(argv[6]);
            session->prefill(suffix, count);
        }
        print_logits(session->current_logits());
        const auto stats = session->stats();
        std::cout << stats.token_count << ',' << stats.cached_tokens << ','
                  << stats.cache_bytes << ',' << stats.uncompressed_cache_bytes << '\n';
        neuralfn::resident_dense::GenerationConfig generation;
        generation.temperature = 0.0;
        generation.top_k = 0;
        generation.top_p = 1.0;
        generation.stop_token_ids = {1, 8};
        const auto decoded = session->decode_one(generation);
        std::cout << decoded.token_id << ',' << decoded.selected_logit << ','
                  << decoded.finish_reason << '\n';
        session->cancel();
        bool cancelled = false;
        try {
            (void)session->current_logits();
        } catch (const neuralfn::resident_dense::ResidentCancellationError&) {
            cancelled = true;
        }
        std::cout << (cancelled ? "cancelled" : "not-cancelled") << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
