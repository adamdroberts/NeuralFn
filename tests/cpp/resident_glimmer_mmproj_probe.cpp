#include "resident_glimmer_vision.h"

#include <exception>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("usage: resident_glimmer_mmproj_probe MMPROJ");
        }
        auto model = neuralfn::resident_glimmer_vision::Model::load_gguf(argv[1]);
        std::cout << model->output_size() << ',' << model->weight_bytes() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
