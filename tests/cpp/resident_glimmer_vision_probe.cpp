#include "resident_glimmer_vision.h"
#include "resident_glimmer_cuda.h"
#include "resident_dense.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class WeightBank final {
public:
    neuralfn::resident_glimmer_cuda::HostWeightView take(
        std::int64_t rows, std::int64_t cols, bool centered = false) {
        storage_.emplace_back(static_cast<std::size_t>(rows * cols), 0);
        neuralfn::resident_glimmer_cuda::HostWeightView result;
        result.data = reinterpret_cast<const std::uint8_t*>(storage_.back().data());
        result.rows = rows;
        result.cols = cols;
        result.row_stride_bytes = cols * 2;
        result.nbytes = rows * result.row_stride_bytes;
        result.encoding = 30;
        result.centered = centered;
        return result;
    }

private:
    std::vector<std::vector<std::uint16_t>> storage_;
};

neuralfn::resident_glimmer_cuda::HostWeightPlan target_weights(
    WeightBank* bank) {
    using neuralfn::resident_glimmer_cuda::HostLayerWeights;
    neuralfn::resident_glimmer_cuda::HostWeightPlan result;
    result.token_embedding = bank->take(13, 8);
    result.final_norm = bank->take(1, 8);
    result.lm_head = bank->take(13, 8);
    for (int index = 0; index < 4; ++index) {
        HostLayerWeights layer;
        layer.input_norm = bank->take(1, 8, true);
        layer.post_attention_norm = bank->take(1, 8, true);
        layer.pre_feedforward_norm = bank->take(1, 8, true);
        layer.post_feedforward_norm = bank->take(1, 8, true);
        layer.q = bank->take(4, 8);
        layer.k = bank->take(2, 8);
        layer.v = bank->take(2, 8);
        layer.gate = bank->take(4, 8);
        layer.output = bank->take(8, 4);
        layer.mlp_gate = bank->take(16, 8);
        layer.mlp_up = bank->take(16, 8);
        layer.mlp_down = bank->take(8, 16);
        result.layers.push_back(std::move(layer));
    }
    return result;
}

void print_row(const std::vector<float>& output) {
    std::cout.precision(9);
    for (std::size_t index = 0; index < output.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << output[index];
    }
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--gguf") {
            auto model = neuralfn::resident_glimmer_vision::Model::load_gguf(argv[2]);
            std::cout << model->output_size() << ',' << model->weight_bytes() << '\n';
            return 0;
        }
        if (argc != 2) {
            throw std::runtime_error(
                "usage: resident_glimmer_vision_probe PAYLOAD | --gguf MMPROJ");
        }
        std::ifstream input(argv[1], std::ios::binary);
        if (!input) {
            throw std::runtime_error("unable to open vision payload");
        }
        std::vector<std::uint8_t> payload(
            (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        neuralfn::resident_glimmer_vision::Config config;
        config.hidden_size = 8;
        config.intermediate_size = 12;
        config.num_layers = 2;
        config.num_heads = 2;
        config.patch_width = 6;
        config.merge_size = 2;
        config.position_side = 2;
        config.adapter_size = 5;
        config.output_size = 8;
        config.rope_theta = 10000.0f;
        config.norm_eps = 1.0e-5f;
        auto model = neuralfn::resident_glimmer_vision::Model::load_bf16(
            payload.data(), static_cast<std::int64_t>(payload.size()), config);
        std::vector<float> patches(8 * 6);
        for (std::size_t index = 0; index < patches.size(); ++index) {
            patches[index] = static_cast<float>(
                0.19 * std::sin(static_cast<double>(index) * 0.17) +
                0.07 * std::cos(static_cast<double>(index) * 0.11));
        }
        std::atomic<bool> cancelled{false};
        const std::vector<float> output = model->encode(patches, {1, 2, 4}, cancelled);
        print_row(output);
        cancelled.store(true);
        bool cancellation_observed = false;
        try {
            (void)model->encode(patches, {1, 2, 4}, cancelled);
        } catch (const neuralfn::resident_dense::ResidentCancellationError&) {
            cancellation_observed = true;
        }
        std::cout << (cancellation_observed ? "cancelled" : "not-cancelled") << '\n';
        const char* cuda_runtime = std::getenv("NFN_TEST_GLIMMER_CUDA_RUNTIME");
        const char* tile_ops = std::getenv("NFN_TEST_GLIMMER_TILE_OPS");
        if ((cuda_runtime == nullptr) != (tile_ops == nullptr)) {
            throw std::runtime_error(
                "both fake Glimmer CUDA libraries must be configured");
        }
        if (cuda_runtime != nullptr) {
            neuralfn::resident_glimmer_cuda::Config target;
            target.max_seq_len = 8;
            target.vocab_size = 13;
            target.num_layers = 4;
            target.model_dim = 8;
            target.intermediate_dim = 16;
            target.num_heads = 2;
            target.num_kv_heads = 1;
            target.head_dim = 2;
            target.sliding_window = 3;
            target.cuda_runtime_lib = cuda_runtime;
            target.tile_ops_lib = tile_ops;
            WeightBank bank;
            auto plan = target_weights(&bank);
            auto cuda_model = neuralfn::resident_glimmer_cuda::Model::load(
                target, plan);
            cuda_model->load_vision(
                model->cuda_config(), model->cuda_weight_plan());
            cancelled.store(false);
            const std::vector<float> cuda_output = cuda_model->encode_vision(
                patches, {1, 2, 4}, cancelled);
            print_row(cuda_output);
            std::cout << cuda_model->vision_weight_bytes() << ','
                      << cuda_model->workspace_bytes() << ','
                      << cuda_model->kernel_launches() << '\n';
            cancelled.store(true);
            bool cuda_cancellation_observed = false;
            try {
                (void)cuda_model->encode_vision(patches, {1, 2, 4}, cancelled);
            } catch (const std::runtime_error&) {
                cuda_cancellation_observed = true;
            }
            std::cout << (cuda_cancellation_observed
                ? "cuda-cancelled" : "cuda-not-cancelled") << '\n';
            cancellation_observed = cancellation_observed &&
                cuda_cancellation_observed;
        }
        return cancellation_observed ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
