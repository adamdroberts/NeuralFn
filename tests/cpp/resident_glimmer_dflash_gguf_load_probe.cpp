#include "resident_glimmer.h"
#include "resident_glimmer_assistant.h"

#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error("usage: dflash-gguf-load-probe TARGET DFLASH");
        }
        neuralfn::resident_glimmer::GlimmerInferenceConfig target_config;
        target_config.checkpoint_sha256 = std::string(64, '0');
        target_config.checkpoint_sha256_preverified = true;
        auto target = neuralfn::resident_glimmer::GlimmerModel::load(argv[1], target_config);

        neuralfn::resident_glimmer_assistant::Config assistant_config;
        assistant_config.container =
            neuralfn::resident_glimmer_assistant::WeightContainer::GgufKQuant;
        assistant_config.checkpoint_sha256 = std::string(64, '1');
        assistant_config.checkpoint_sha256_preverified = true;
        auto assistant = neuralfn::resident_glimmer_assistant::Model::load(
            argv[2], assistant_config, target);
        std::cout << assistant->parameter_count() << ',' << assistant->weight_bytes()
                  << ',' << assistant->target_layer_ids().front() << ','
                  << assistant->target_layer_ids().back() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
