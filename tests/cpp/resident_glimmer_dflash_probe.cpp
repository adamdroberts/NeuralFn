#include "resident_glimmer.h"
#include "resident_glimmer_assistant.h"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::int64_t> parse_tokens(const std::string& text) {
    std::vector<std::int64_t> result;
    std::stringstream input(text);
    std::string item;
    while (std::getline(input, item, ',')) {
        if (!item.empty()) result.push_back(std::stoll(item));
    }
    return result;
}

void print_tokens(const std::vector<neuralfn::resident_dense::DecodeResult>& rows) {
    for (std::size_t index = 0; index < rows.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << rows[index].token_id;
    }
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 6) {
            throw std::runtime_error(
                "usage: dflash-probe TARGET TARGET_SHA ASSISTANT ASSISTANT_SHA TOKENS");
        }
        neuralfn::resident_glimmer::GlimmerInferenceConfig target_config;
        target_config.max_seq_len = 12;
        target_config.vocab_size = 13;
        target_config.num_layers = 4;
        target_config.model_dim = 8;
        target_config.intermediate_dim = 16;
        target_config.num_heads = 2;
        target_config.num_kv_heads = 1;
        target_config.head_dim = 2;
        target_config.sliding_window = 3;
        target_config.rope_theta = 500000.0;
        target_config.norm_eps = 1.0e-5;
        target_config.post_norm_eps = 1.0e-8;
        target_config.q_scale_factor = 3.87;
        target_config.output_multiplier = 0.19611613513818404;
        target_config.logit_softcap = 20.0;
        target_config.container = neuralfn::resident_glimmer::WeightContainer::NativeBf16;
        target_config.checkpoint_sha256 = argv[2];
        const char* cuda_runtime = std::getenv("NFN_TEST_GLIMMER_CUDA_RUNTIME");
        const char* tile_ops = std::getenv("NFN_TEST_GLIMMER_TILE_OPS");
        if ((cuda_runtime == nullptr) != (tile_ops == nullptr)) {
            throw std::runtime_error("both fake Glimmer CUDA libraries must be configured");
        }
        if (cuda_runtime != nullptr) {
            target_config.whole_model_cuda = true;
            target_config.cuda_device = 0;
            target_config.cuda_runtime_lib = cuda_runtime;
            target_config.tile_ops_lib = tile_ops;
        }
        auto target = neuralfn::resident_glimmer::GlimmerModel::load(argv[1], target_config);

        neuralfn::resident_glimmer_assistant::Config assistant_config;
        assistant_config.max_seq_len = 12;
        assistant_config.model_dim = 8;
        assistant_config.intermediate_dim = 16;
        assistant_config.num_layers = 2;
        assistant_config.num_heads = 2;
        assistant_config.num_kv_heads = 1;
        assistant_config.head_dim = 2;
        assistant_config.block_size = 4;
        assistant_config.mask_token_id = 12;
        assistant_config.sliding_window = 3;
        assistant_config.rope_theta = 500000.0;
        assistant_config.norm_eps = 1.0e-5;
        assistant_config.target_layer_ids = {0, 2};
        assistant_config.checkpoint_sha256 = argv[4];
        auto assistant = neuralfn::resident_glimmer_assistant::Model::load(
            argv[3], assistant_config, target);
        auto session = target->create_speculative_session(
            29, neuralfn::resident_dense::KVCacheMode::Full, assistant);
        session->prefill(parse_tokens(argv[5]), 0);
        neuralfn::resident_dense::GenerationConfig generation;
        generation.temperature = 0.0;
        generation.top_k = 0;
        generation.top_p = 1.0;
        generation.seed = 29;
        generation.stop_token_ids = {};
        const auto warmup = session->decode_speculative_block(generation, 1);
        const auto block = session->decode_speculative_block(generation, 4);
        print_tokens(warmup.tokens);
        print_tokens(block.tokens);
        std::cout << block.proposed_tokens << ',' << block.accepted_tokens << ','
                  << block.rejected_tokens << ',' << block.target_rows << ','
                  << block.assistant_blocks << '\n';
        const auto stats = session->stats();
        std::cout << stats.token_count << ',' << stats.cached_tokens << ','
                  << stats.speculative_blocks << ',' << stats.speculative_proposed_tokens << ','
                  << stats.speculative_accepted_tokens << ','
                  << stats.speculative_rejected_tokens << ','
                  << stats.assistant_cache_bytes << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
