#include "resident_glimmer_vision.h"
#include "resident_dense.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

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
        std::cout.precision(9);
        for (std::size_t index = 0; index < output.size(); ++index) {
            if (index != 0) {
                std::cout << ',';
            }
            std::cout << output[index];
        }
        std::cout << '\n';
        cancelled.store(true);
        bool cancellation_observed = false;
        try {
            (void)model->encode(patches, {1, 2, 4}, cancelled);
        } catch (const neuralfn::resident_dense::ResidentCancellationError&) {
            cancellation_observed = true;
        }
        std::cout << (cancellation_observed ? "cancelled" : "not-cancelled") << '\n';
        return cancellation_observed ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
