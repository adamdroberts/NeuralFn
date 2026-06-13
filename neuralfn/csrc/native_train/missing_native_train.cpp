#include <iostream>
#include <string>

#ifndef NFN_NATIVE_MODEL_FAMILY
#define NFN_NATIVE_MODEL_FAMILY "unknown"
#endif

#ifndef NFN_NATIVE_TARGET_NAME
#define NFN_NATIVE_TARGET_NAME "nfn_unknown_native_train"
#endif

#ifndef NFN_NATIVE_REQUIRED_KERNELS
#define NFN_NATIVE_REQUIRED_KERNELS "model-specific CUDA Tile kernels"
#endif

namespace {

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [native options]\n\n"
        << NFN_NATIVE_TARGET_NAME << " is a compiled NeuralFn native training placeholder for "
        << NFN_NATIVE_MODEL_FAMILY << ".\n"
        << "It intentionally fails until the required CUDA Tile C++ trainer kernels are implemented.\n\n"
        << "Required native work:\n"
        << "  " << NFN_NATIVE_REQUIRED_KERNELS << "\n\n"
        << "This command exists so CLI, SDK, and install paths stay on compiled native boundaries\n"
        << "instead of entering graph-backed TorchTrainer code.\n";
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::cerr
        << NFN_NATIVE_TARGET_NAME << ": native CUDA Tile trainer for " << NFN_NATIVE_MODEL_FAMILY
        << " is not implemented yet.\n"
        << "Required native work: " << NFN_NATIVE_REQUIRED_KERNELS << "\n"
        << "Do not use the graph-backed TorchTrainer path for production training; implement this "
        << "family's CUDA Tile C++ kernels first, or set NFN_ALLOW_TORCH_TRAINING=1 only for local debugging.\n";
    return 2;
}
