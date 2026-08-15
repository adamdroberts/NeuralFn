#include "resident_glimmer_vision.h"
#include "resident_glimmer_cuda.h"

#include "resident_dense.h"
#include "resident_sha256.h"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace neuralfn::resident_glimmer_vision {
namespace {

class Cursor final {
public:
    Cursor(const std::uint8_t* data, std::size_t size) : data_(data), size_(size) {}

    std::size_t offset() const noexcept { return offset_; }
    const std::uint8_t* pointer() const noexcept { return data_ + offset_; }

    void require(std::size_t count, const char* label) const {
        if (count > size_ - std::min(size_, offset_) || offset_ + count > 64u * 1024u * 1024u) {
            throw std::runtime_error(std::string("Muse Glimmer mmproj is truncated at ") + label);
        }
    }

    void skip(std::size_t count, const char* label) {
        require(count, label);
        offset_ += count;
    }

    template <typename Value>
    Value read(const char* label) {
        require(sizeof(Value), label);
        Value value{};
        std::memcpy(&value, data_ + offset_, sizeof(Value));
        offset_ += sizeof(Value);
        return value;
    }

    std::string string(const char* label) {
        const std::uint64_t length = read<std::uint64_t>(label);
        if (length == 0 || length > 32u * 1024u * 1024u ||
            length > std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("Muse Glimmer mmproj has an invalid string");
        }
        require(static_cast<std::size_t>(length), label);
        std::string result(
            reinterpret_cast<const char*>(data_ + offset_),
            static_cast<std::size_t>(length));
        offset_ += static_cast<std::size_t>(length);
        if (result.find('\0') != std::string::npos) {
            throw std::runtime_error("Muse Glimmer mmproj string contains NUL");
        }
        return result;
    }

private:
    const std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t offset_ = 0;
};

enum GgufValueType : std::uint32_t {
    GgufUint8 = 0,
    GgufInt8 = 1,
    GgufUint16 = 2,
    GgufInt16 = 3,
    GgufUint32 = 4,
    GgufInt32 = 5,
    GgufFloat32 = 6,
    GgufBool = 7,
    GgufString = 8,
    GgufArray = 9,
    GgufUint64 = 10,
    GgufInt64 = 11,
    GgufFloat64 = 12,
};

std::size_t align_up(std::size_t value, std::size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0 ||
        value > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
        throw std::runtime_error("Muse Glimmer mmproj alignment is invalid");
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

std::uint64_t read_integer(Cursor* cursor, std::uint32_t type, const char* label) {
    if (type == GgufUint32) return cursor->read<std::uint32_t>(label);
    if (type == GgufUint64) return cursor->read<std::uint64_t>(label);
    throw std::runtime_error(std::string("Muse Glimmer mmproj metadata type mismatch at ") + label);
}

void require_string(Cursor* cursor, std::uint32_t type, const char* expected, const char* label) {
    if (type != GgufString || cursor->string(label) != expected) {
        throw std::runtime_error(std::string("Muse Glimmer mmproj metadata mismatch at ") + label);
    }
}

struct TensorContract {
    std::vector<std::uint64_t> dimensions;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::uint32_t type = 0;
};

TensorContract mmproj_tensor_contract(const std::string& name) {
    const std::unordered_map<std::string, TensorContract> fixed = {
        {"v.patch_embd.weight", {{14, 14, 3, 1536}, 1536, 588, 0}},
        {"v.position_embd.weight", {{1536, 1024}, 1024, 1536, 0}},
        {"v.pre_ln.weight", {{1536}, 1, 1536, 0}},
        {"v.pre_ln.bias", {{1536}, 1, 1536, 0}},
        {"v.post_ln.weight", {{1536}, 1, 1536, 0}},
        {"v.post_ln.bias", {{1536}, 1, 1536, 0}},
        {"mm.0.weight", {{6144, 4096}, 4096, 6144, 30}},
        {"mm.1.weight", {{4096, 4096}, 4096, 4096, 30}},
        {"mm.2.weight", {{4096, 6656}, 6656, 4096, 30}},
    };
    const auto found = fixed.find(name);
    if (found != fixed.end()) return found->second;
    if (!name.starts_with("v.blk.")) {
        throw std::runtime_error("Muse Glimmer mmproj contains an unexpected tensor: " + name);
    }
    const std::size_t layer_end = name.find('.', 6);
    if (layer_end == std::string::npos) {
        throw std::runtime_error("Muse Glimmer mmproj tensor name is malformed");
    }
    const std::string layer_text = name.substr(6, layer_end - 6);
    if (layer_text.empty() ||
        !std::all_of(layer_text.begin(), layer_text.end(), [](unsigned char value) {
            return value >= '0' && value <= '9';
        })) {
        throw std::runtime_error("Muse Glimmer mmproj layer index is malformed");
    }
    const int layer = std::stoi(layer_text);
    if (layer < 0 || layer >= 50) {
        throw std::runtime_error("Muse Glimmer mmproj layer index is out of range");
    }
    const std::string suffix = name.substr(layer_end + 1);
    if (suffix.ends_with(".bias")) {
        const std::string operation = suffix.substr(0, suffix.size() - 5);
        const std::int64_t width = operation == "ffn_up" ? 8960 : 1536;
        return {{static_cast<std::uint64_t>(width)}, 1, width, 0};
    }
    if (!suffix.ends_with(".weight")) {
        throw std::runtime_error("Muse Glimmer mmproj tensor parameter is unsupported");
    }
    const std::string operation = suffix.substr(0, suffix.size() - 7);
    if (operation == "ln1" || operation == "ln2") {
        return {{1536}, 1, 1536, 0};
    }
    if (operation == "attn_q" || operation == "attn_k" ||
        operation == "attn_v" || operation == "attn_out") {
        return {{1536, 1536}, 1536, 1536, operation == "attn_v" ? 14u : 12u};
    }
    if (operation == "ffn_up") {
        return {{1536, 8960}, 8960, 1536, 12};
    }
    if (operation == "ffn_down") {
        return {{8960, 1536}, 1536, 8960, 14};
    }
    throw std::runtime_error("Muse Glimmer mmproj tensor operation is unsupported: " + name);
}

std::unordered_set<std::string> expected_mmproj_names() {
    std::unordered_set<std::string> names = {
        "v.patch_embd.weight", "v.position_embd.weight",
        "v.pre_ln.weight", "v.pre_ln.bias", "v.post_ln.weight", "v.post_ln.bias",
        "mm.0.weight", "mm.1.weight", "mm.2.weight",
    };
    const std::vector<std::string> operations = {
        "attn_k", "attn_out", "attn_q", "attn_v",
        "ffn_up", "ffn_down", "ln1", "ln2",
    };
    for (int layer = 0; layer < 50; ++layer) {
        for (const std::string& operation : operations) {
            names.insert("v.blk." + std::to_string(layer) + "." + operation + ".weight");
            names.insert("v.blk." + std::to_string(layer) + "." + operation + ".bias");
        }
    }
    return names;
}

std::uint32_t validated_mmproj_encoding(std::uint32_t type) {
    switch (type) {
        case 0:
        case 12:
        case 14:
        case 30:
            return type;
        default: throw std::runtime_error("Muse Glimmer mmproj tensor encoding is unsupported");
    }
}

using neuralfn::resident_dense::ResidentCancellationError;

std::int64_t checked_add(std::int64_t left, std::int64_t right, const char* label) {
    if (left < 0 || right < 0 || left > std::numeric_limits<std::int64_t>::max() - right) {
        throw std::runtime_error(std::string("native Glimmer vision size overflow at ") + label);
    }
    return left + right;
}

std::int64_t checked_mul(std::int64_t left, std::int64_t right, const char* label) {
    if (left < 0 || right < 0 ||
        (left != 0 && right > std::numeric_limits<std::int64_t>::max() / left)) {
        throw std::runtime_error(std::string("native Glimmer vision size overflow at ") + label);
    }
    return left * right;
}

std::size_t checked_size(std::int64_t value, const char* label) {
    if (value < 0 || static_cast<std::uint64_t>(value) >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(std::string("native Glimmer vision host-size overflow at ") + label);
    }
    return static_cast<std::size_t>(value);
}

void throw_if_cancelled(const std::atomic<bool>& cancelled) {
    if (cancelled.load(std::memory_order_relaxed)) {
        throw ResidentCancellationError("resident Glimmer vision encoding was cancelled");
    }
}

float bf16_to_float(const std::uint8_t* source) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, source, sizeof(bits));
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16u);
}

float fp16_to_float(const std::uint8_t* source) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, source, sizeof(bits));
    const bool negative = (bits & 0x8000u) != 0;
    const int exponent = static_cast<int>((bits >> 10u) & 0x1fu);
    const int mantissa = static_cast<int>(bits & 0x03ffu);
    double value = 0.0;
    if (exponent == 0) {
        value = std::ldexp(static_cast<double>(mantissa), -24);
    } else if (exponent == 31) {
        value = mantissa == 0
            ? std::numeric_limits<double>::infinity()
            : std::numeric_limits<double>::quiet_NaN();
    } else {
        value = std::ldexp(
            1.0 + static_cast<double>(mantissa) / 1024.0,
            exponent - 15);
    }
    return static_cast<float>(negative ? -value : value);
}

void k_scale_min(const std::uint8_t* scales, int index, int* scale, int* minimum) {
    if (index < 4) {
        *scale = scales[index] & 63;
        *minimum = scales[index + 4] & 63;
        return;
    }
    *scale = (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4);
    *minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
}

template <typename Function>
void parallel_rows(std::int64_t rows, const std::atomic<bool>& cancelled, Function&& function) {
    if (rows <= 0) {
        return;
    }
    const unsigned available = std::max(1u, std::thread::hardware_concurrency());
    const std::int64_t workers = std::min<std::int64_t>(rows, available);
    if (workers == 1 || rows < 4) {
        for (std::int64_t row = 0; row < rows; ++row) {
            throw_if_cancelled(cancelled);
            function(row);
        }
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(checked_size(workers, "worker count"));
    std::exception_ptr error;
    std::mutex error_mutex;
    for (std::int64_t worker = 0; worker < workers; ++worker) {
        const std::int64_t begin = rows * worker / workers;
        const std::int64_t end = rows * (worker + 1) / workers;
        threads.emplace_back([&, begin, end]() {
            try {
                for (std::int64_t row = begin; row < end; ++row) {
                    throw_if_cancelled(cancelled);
                    function(row);
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) {
                    error = std::current_exception();
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

}  // namespace

struct Model::Weight {
    enum class Encoding { F32, BF16, Q4K, Q6K };

    const std::uint8_t* data = nullptr;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::int64_t row_stride_bytes = 0;
    std::int64_t nbytes = 0;
    Encoding encoding = Encoding::BF16;

    float at(std::int64_t row, std::int64_t col) const {
        if (data == nullptr || row < 0 || row >= rows || col < 0 || col >= cols) {
            throw std::runtime_error("native Glimmer vision weight index is invalid");
        }
        const std::uint8_t* row_data = data + checked_size(
            checked_mul(row, row_stride_bytes, "weight row"), "weight row");
        if (encoding == Encoding::F32) {
            float value = 0.0f;
            std::memcpy(&value, row_data + checked_size(col * 4, "F32 weight index"), 4);
            return value;
        }
        if (encoding == Encoding::BF16) {
            return bf16_to_float(row_data + checked_size(col * 2, "BF16 weight index"));
        }
        const std::int64_t block_index = col / 256;
        const int within = static_cast<int>(col % 256);
        if (encoding == Encoding::Q4K) {
            const std::uint8_t* block = row_data + block_index * 144;
            const float d = fp16_to_float(block);
            const float dmin = fp16_to_float(block + 2);
            const std::uint8_t* scales = block + 4;
            const std::uint8_t* quants = block + 16;
            const int group = within / 64;
            const int lane = within % 32;
            const bool high = (within % 64) >= 32;
            int scale = 0;
            int minimum = 0;
            k_scale_min(scales, group * 2 + (high ? 1 : 0), &scale, &minimum);
            const int quant = high ? (quants[group * 32 + lane] >> 4)
                                   : (quants[group * 32 + lane] & 0x0f);
            return d * static_cast<float>(scale * quant) -
                dmin * static_cast<float>(minimum);
        }
        const std::uint8_t* block = row_data + block_index * 210;
        const std::uint8_t* low_bits = block;
        const std::uint8_t* high_bits = block + 128;
        const auto* scales = reinterpret_cast<const std::int8_t*>(block + 192);
        const float d = fp16_to_float(block + 208);
        const int half = within / 128;
        const int position = within % 128;
        const int quarter = position / 32;
        const int lane = position % 32;
        const std::uint8_t low = low_bits[
            half * 64 + lane + ((quarter == 1 || quarter == 3) ? 32 : 0)];
        const std::uint8_t high = high_bits[half * 32 + lane];
        int quant = 0;
        if (quarter == 0) {
            quant = (low & 0x0f) | (((high >> 0) & 3) << 4);
        } else if (quarter == 1) {
            quant = (low & 0x0f) | (((high >> 2) & 3) << 4);
        } else if (quarter == 2) {
            quant = (low >> 4) | (((high >> 4) & 3) << 4);
        } else {
            quant = (low >> 4) | (((high >> 6) & 3) << 4);
        }
        return d * static_cast<float>(scales[half * 8 + 2 * quarter]) *
            static_cast<float>(quant - 32);
    }
};

struct Model::Layer {
    Weight* q_weight = nullptr;
    Weight* q_bias = nullptr;
    Weight* k_weight = nullptr;
    Weight* k_bias = nullptr;
    Weight* v_weight = nullptr;
    Weight* v_bias = nullptr;
    Weight* output_weight = nullptr;
    Weight* output_bias = nullptr;
    Weight* norm1_weight = nullptr;
    Weight* norm1_bias = nullptr;
    Weight* norm2_weight = nullptr;
    Weight* norm2_bias = nullptr;
    Weight* fc1_weight = nullptr;
    Weight* fc1_bias = nullptr;
    Weight* fc2_weight = nullptr;
    Weight* fc2_bias = nullptr;
};

class Model::MappedFile final {
public:
    explicit MappedFile(const std::string& path) {
#if defined(_WIN32)
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input) {
            throw std::runtime_error("failed to open Muse Glimmer mmproj: " + path);
        }
        const auto end = input.tellg();
        if (end <= 0 || static_cast<std::uint64_t>(end) >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error("Muse Glimmer mmproj has an invalid byte length");
        }
        owned_.resize(static_cast<std::size_t>(end));
        input.seekg(0);
        input.read(reinterpret_cast<char*>(owned_.data()), end);
        if (!input) {
            throw std::runtime_error("Muse Glimmer mmproj is truncated");
        }
        data_ = owned_.data();
        size_ = owned_.size();
#else
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) {
            throw std::runtime_error(
                "failed to open Muse Glimmer mmproj: " + std::string(std::strerror(errno)));
        }
        struct stat status {};
        if (::fstat(fd_, &status) != 0 || !S_ISREG(status.st_mode) || status.st_size <= 0 ||
            static_cast<std::uint64_t>(status.st_size) >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            const std::string message = std::strerror(errno);
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error(
                "Muse Glimmer mmproj has an invalid byte length: " + message);
        }
        size_ = static_cast<std::size_t>(status.st_size);
        void* mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapped == MAP_FAILED) {
            const std::string message = std::strerror(errno);
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("failed to mmap Muse Glimmer mmproj: " + message);
        }
        data_ = static_cast<const std::uint8_t*>(mapped);
#endif
    }

    ~MappedFile() {
#if !defined(_WIN32)
        if (data_ != nullptr) {
            ::munmap(const_cast<std::uint8_t*>(data_), size_);
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
#endif
    }

    const std::uint8_t* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

private:
    const std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
#if defined(_WIN32)
    std::vector<std::uint8_t> owned_;
#else
    int fd_ = -1;
#endif
};

namespace {

void validate_config(const Config& config) {
    if (config.hidden_size <= 0 || config.intermediate_size <= 0 ||
        config.num_layers <= 0 || config.num_heads <= 0 ||
        config.hidden_size % config.num_heads != 0 ||
        (config.hidden_size / config.num_heads) % 4 != 0 ||
        config.patch_width <= 0 || config.merge_size <= 0 ||
        config.position_side <= 0 || config.adapter_size <= 0 ||
        config.output_size <= 0 || !std::isfinite(config.rope_theta) ||
        !(config.rope_theta > 0.0f) || !std::isfinite(config.norm_eps) ||
        !(config.norm_eps > 0.0f)) {
        throw std::runtime_error("native Glimmer vision geometry is invalid");
    }
}

void linear(
    const std::vector<float>& input,
    std::int64_t input_rows,
    std::int64_t input_width,
    const Model::Weight& weight,
    const Model::Weight* bias,
    std::vector<float>* output,
    const std::atomic<bool>& cancelled) {
    if (weight.cols != input_width ||
        static_cast<std::int64_t>(input.size()) != checked_mul(input_rows, input_width, "linear input")) {
        throw std::runtime_error("native Glimmer vision linear geometry mismatch");
    }
    if (bias != nullptr && (bias->rows != 1 || bias->cols != weight.rows)) {
        throw std::runtime_error("native Glimmer vision linear bias geometry mismatch");
    }
    output->assign(
        checked_size(checked_mul(input_rows, weight.rows, "linear output"), "linear output"),
        0.0f);
    parallel_rows(input_rows, cancelled, [&](std::int64_t row) {
        const float* source = input.data() + row * input_width;
        float* destination = output->data() + row * weight.rows;
        for (std::int64_t out = 0; out < weight.rows; ++out) {
            double sum = bias == nullptr ? 0.0 : static_cast<double>(bias->at(0, out));
            for (std::int64_t in = 0; in < input_width; ++in) {
                sum += static_cast<double>(source[in]) * static_cast<double>(weight.at(out, in));
            }
            destination[out] = static_cast<float>(sum);
        }
    });
}

void layer_norm_inplace(
    std::vector<float>* values,
    std::int64_t rows,
    std::int64_t width,
    const Model::Weight& weight,
    const Model::Weight& bias,
    float epsilon,
    const std::atomic<bool>& cancelled) {
    if (weight.rows != 1 || bias.rows != 1 || weight.cols != width || bias.cols != width ||
        static_cast<std::int64_t>(values->size()) != checked_mul(rows, width, "layer norm values")) {
        throw std::runtime_error("native Glimmer vision LayerNorm geometry mismatch");
    }
    parallel_rows(rows, cancelled, [&](std::int64_t row) {
        float* data = values->data() + row * width;
        double mean = 0.0;
        for (std::int64_t index = 0; index < width; ++index) {
            mean += data[index];
        }
        mean /= static_cast<double>(width);
        double variance = 0.0;
        for (std::int64_t index = 0; index < width; ++index) {
            const double centered = static_cast<double>(data[index]) - mean;
            variance += centered * centered;
        }
        const double inverse = 1.0 / std::sqrt(variance / static_cast<double>(width) + epsilon);
        for (std::int64_t index = 0; index < width; ++index) {
            data[index] = static_cast<float>(
                (static_cast<double>(data[index]) - mean) * inverse * weight.at(0, index) +
                bias.at(0, index));
        }
    });
}

void scaleless_rms_inplace(
    std::vector<float>* values,
    std::int64_t rows,
    std::int64_t width,
    float epsilon,
    const std::atomic<bool>& cancelled) {
    parallel_rows(rows, cancelled, [&](std::int64_t row) {
        float* data = values->data() + row * width;
        double square_sum = 0.0;
        for (std::int64_t index = 0; index < width; ++index) {
            square_sum += static_cast<double>(data[index]) * data[index];
        }
        const double inverse = 1.0 / std::sqrt(square_sum / static_cast<double>(width) + epsilon);
        for (std::int64_t index = 0; index < width; ++index) {
            data[index] = static_cast<float>(static_cast<double>(data[index]) * inverse);
        }
    });
}

void gelu_inplace(std::vector<float>* values, const std::atomic<bool>& cancelled) {
    const std::int64_t count = static_cast<std::int64_t>(values->size());
    parallel_rows(count, cancelled, [&](std::int64_t index) {
        const double value = (*values)[checked_size(index, "GELU index")];
        (*values)[checked_size(index, "GELU index")] = static_cast<float>(
            0.5 * value * (1.0 + std::erf(value / std::sqrt(2.0))));
    });
}

struct GridRow {
    std::int64_t temporal = 0;
    std::int64_t height = 0;
    std::int64_t width = 0;
    std::int64_t offset = 0;
};

struct VisionLayout {
    std::vector<GridRow> grid;
    std::vector<std::int64_t> window_index;
    std::vector<std::int64_t> window_boundaries{0};
    std::vector<std::int64_t> full_boundaries{0};
    std::vector<std::int64_t> position_width;
    std::vector<std::int64_t> position_height;
    std::int64_t patch_rows = 0;
    std::int64_t merged_rows = 0;
};

VisionLayout make_layout(
    const std::vector<std::int64_t>& grid_thw,
    const Config& config,
    std::int64_t supplied_rows) {
    if (grid_thw.empty() || grid_thw.size() % 3 != 0) {
        throw std::runtime_error("native Glimmer vision grid must be a nonempty flattened [media,3] array");
    }
    VisionLayout layout;
    std::int64_t offset = 0;
    for (std::size_t media = 0; media < grid_thw.size(); media += 3) {
        const std::int64_t temporal = grid_thw[media];
        const std::int64_t height = grid_thw[media + 1];
        const std::int64_t width = grid_thw[media + 2];
        if (temporal <= 0 || height <= 0 || width <= 0 ||
            height % config.merge_size != 0 || width % config.merge_size != 0) {
            throw std::runtime_error("native Glimmer vision grid has invalid or unmergeable dimensions");
        }
        const std::int64_t spatial = checked_mul(height, width, "grid spatial size");
        const std::int64_t rows = checked_mul(temporal, spatial, "grid patch rows");
        layout.grid.push_back({temporal, height, width, offset});
        offset = checked_add(offset, rows, "grid row offset");
        layout.full_boundaries.push_back(offset);
        layout.merged_rows = checked_add(
            layout.merged_rows,
            checked_mul(
                temporal,
                checked_mul(height / config.merge_size, width / config.merge_size, "merged spatial size"),
                "merged media rows"),
            "merged rows");

        const std::int64_t window = config.position_side;
        const std::int64_t windows_h = (height + window - 1) / window;
        const std::int64_t windows_w = (width + window - 1) / window;
        for (std::int64_t time = 0; time < temporal; ++time) {
            for (std::int64_t window_h = 0; window_h < windows_h; ++window_h) {
                for (std::int64_t window_w = 0; window_w < windows_w; ++window_w) {
                    std::int64_t window_rows = 0;
                    for (std::int64_t local_h = 0; local_h < window; ++local_h) {
                        const std::int64_t h = window_h * window + local_h;
                        if (h >= height) {
                            continue;
                        }
                        for (std::int64_t local_w = 0; local_w < window; ++local_w) {
                            const std::int64_t w = window_w * window + local_w;
                            if (w >= width) {
                                continue;
                            }
                            layout.window_index.push_back(
                                layout.grid.back().offset + time * spatial + h * width + w);
                            layout.position_width.push_back(w + 1);
                            layout.position_height.push_back(h + 1);
                            ++window_rows;
                        }
                    }
                    if (window_rows > 0) {
                        layout.window_boundaries.push_back(
                            checked_add(layout.window_boundaries.back(), window_rows, "window boundary"));
                    }
                }
            }
        }
    }
    layout.patch_rows = offset;
    if (layout.patch_rows != supplied_rows ||
        static_cast<std::int64_t>(layout.window_index.size()) != supplied_rows ||
        layout.full_boundaries.back() != supplied_rows ||
        layout.window_boundaries.back() != supplied_rows) {
        throw std::runtime_error("native Glimmer vision grid and packed-patch rows differ");
    }
    return layout;
}

std::vector<float> position_embedding(
    const Model::Weight& table,
    const VisionLayout& layout,
    const Config& config,
    const std::atomic<bool>& cancelled) {
    const std::int64_t hidden = config.hidden_size;
    std::vector<float> result(
        checked_size(checked_mul(layout.patch_rows, hidden, "position output"), "position output"),
        0.0f);
    for (const GridRow& media : layout.grid) {
        throw_if_cancelled(cancelled);
        const float side = static_cast<float>(config.position_side);
        const std::int64_t spatial = media.height * media.width;
        for (std::int64_t h = 0; h < media.height; ++h) {
            const float h_coord = (static_cast<float>(h) + 0.5f) *
                (side / static_cast<float>(media.height)) - 0.5f;
            const std::int64_t h_floor_raw = static_cast<std::int64_t>(std::floor(h_coord));
            const std::int64_t h_ceil_raw = h_floor_raw + 1;
            const float h_fraction = h_coord - static_cast<float>(h_floor_raw);
            for (std::int64_t w = 0; w < media.width; ++w) {
                const float w_coord = (static_cast<float>(w) + 0.5f) *
                    (side / static_cast<float>(media.width)) - 0.5f;
                const std::int64_t w_floor_raw = static_cast<std::int64_t>(std::floor(w_coord));
                const std::int64_t w_ceil_raw = w_floor_raw + 1;
                const float w_fraction = w_coord - static_cast<float>(w_floor_raw);
                const std::int64_t hs[4] = {h_floor_raw, h_floor_raw, h_ceil_raw, h_ceil_raw};
                const std::int64_t ws[4] = {w_floor_raw, w_ceil_raw, w_floor_raw, w_ceil_raw};
                const float weights[4] = {
                    (1.0f - h_fraction) * (1.0f - w_fraction),
                    (1.0f - h_fraction) * w_fraction,
                    h_fraction * (1.0f - w_fraction),
                    h_fraction * w_fraction,
                };
                for (std::int64_t time = 0; time < media.temporal; ++time) {
                    float* destination = result.data() +
                        (media.offset + time * spatial + h * media.width + w) * hidden;
                    for (int corner = 0; corner < 4; ++corner) {
                        if (hs[corner] < 0 || hs[corner] >= config.position_side ||
                            ws[corner] < 0 || ws[corner] >= config.position_side) {
                            continue;
                        }
                        const std::int64_t table_row = hs[corner] * config.position_side + ws[corner];
                        for (std::int64_t dim = 0; dim < hidden; ++dim) {
                            destination[dim] += weights[corner] * table.at(table_row, dim);
                        }
                    }
                }
            }
        }
    }
    return result;
}

void apply_rope_2d(
    std::vector<float>* values,
    const VisionLayout& layout,
    const Config& config,
    bool interleaved,
    const std::atomic<bool>& cancelled) {
    const std::int64_t head_dim = config.hidden_size / config.num_heads;
    const std::int64_t spatial_dim = head_dim / 2;
    std::vector<float> inverse_frequency(checked_size(spatial_dim / 2, "RoPE frequency count"));
    for (std::int64_t index = 0; index < spatial_dim / 2; ++index) {
        inverse_frequency[checked_size(index, "RoPE frequency index")] = static_cast<float>(
            1.0 / std::pow(config.rope_theta, static_cast<double>(index * 2) / spatial_dim));
    }
    parallel_rows(layout.patch_rows, cancelled, [&](std::int64_t row) {
        float* row_data = values->data() + row * config.hidden_size;
        for (std::int64_t head = 0; head < config.num_heads; ++head) {
            float* data = row_data + head * head_dim;
            if (interleaved) {
                for (std::int64_t pair = 0; pair < head_dim / 2; ++pair) {
                    const bool width_axis = pair < spatial_dim / 2;
                    const float position = static_cast<float>(width_axis
                        ? layout.position_width[checked_size(row, "position row")]
                        : layout.position_height[checked_size(row, "position row")]);
                    const float angle = position * inverse_frequency[checked_size(
                        pair % (spatial_dim / 2), "frequency index")];
                    const float first = data[2 * pair];
                    const float second = data[2 * pair + 1];
                    data[2 * pair] = first * std::cos(angle) - second * std::sin(angle);
                    data[2 * pair + 1] = second * std::cos(angle) + first * std::sin(angle);
                }
                continue;
            }
            std::vector<float> original(data, data + head_dim);
            for (std::int64_t dim = 0; dim < head_dim; ++dim) {
                const std::int64_t quarter_index = dim % (spatial_dim / 2);
                const bool width_axis = dim < spatial_dim / 2 ||
                    (dim >= spatial_dim && dim < spatial_dim + spatial_dim / 2);
                const float position = static_cast<float>(width_axis
                    ? layout.position_width[checked_size(row, "position row")]
                    : layout.position_height[checked_size(row, "position row")]);
                const float angle = position * inverse_frequency[checked_size(quarter_index, "frequency index")];
                const float rotated = dim < head_dim / 2
                    ? -original[checked_size(dim + head_dim / 2, "RoPE second-half index")]
                    : original[checked_size(dim - head_dim / 2, "RoPE first-half index")];
                data[dim] = original[checked_size(dim, "RoPE value index")] * std::cos(angle) +
                    rotated * std::sin(angle);
            }
        }
    });
}

std::vector<float> attention(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    const std::vector<std::int64_t>& boundaries,
    const Config& config,
    const std::atomic<bool>& cancelled) {
    const std::int64_t rows = static_cast<std::int64_t>(q.size()) / config.hidden_size;
    const std::int64_t head_dim = config.hidden_size / config.num_heads;
    std::vector<float> output(q.size(), 0.0f);
    for (std::size_t segment = 0; segment + 1 < boundaries.size(); ++segment) {
        const std::int64_t begin = boundaries[segment];
        const std::int64_t end = boundaries[segment + 1];
        if (begin < 0 || end <= begin || end > rows) {
            throw std::runtime_error("native Glimmer vision attention boundaries are invalid");
        }
        parallel_rows((end - begin) * config.num_heads, cancelled, [&](std::int64_t lane) {
            const std::int64_t query_row = begin + lane / config.num_heads;
            const std::int64_t head = lane % config.num_heads;
            const float* query = q.data() + query_row * config.hidden_size + head * head_dim;
            std::vector<double> scores(checked_size(end - begin, "attention scores"));
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::int64_t key_row = begin; key_row < end; ++key_row) {
                const float* key = k.data() + key_row * config.hidden_size + head * head_dim;
                double score = 0.0;
                for (std::int64_t dim = 0; dim < head_dim; ++dim) {
                    score += static_cast<double>(query[dim]) * key[dim];
                }
                score /= std::sqrt(static_cast<double>(head_dim));
                scores[checked_size(key_row - begin, "attention score index")] = score;
                maximum = std::max(maximum, score);
            }
            double denominator = 0.0;
            for (double& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            float* destination = output.data() + query_row * config.hidden_size + head * head_dim;
            for (std::int64_t key_row = begin; key_row < end; ++key_row) {
                const double probability = scores[checked_size(key_row - begin, "attention probability index")] /
                    denominator;
                const float* value = v.data() + key_row * config.hidden_size + head * head_dim;
                for (std::int64_t dim = 0; dim < head_dim; ++dim) {
                    destination[dim] += static_cast<float>(probability * value[dim]);
                }
            }
        });
    }
    return output;
}

}  // namespace

Model::Model(const std::uint8_t* payload, std::int64_t payload_nbytes, Config config)
    : payload_(payload), payload_nbytes_(payload_nbytes), config_(std::move(config)) {
    validate_config(config_);
    if (payload_ == nullptr || payload_nbytes_ <= 0) {
        throw std::runtime_error("native Glimmer vision BF16 payload is empty");
    }
    build_layout();
}

Model::Model(std::unique_ptr<MappedFile> mapped, Config config)
    : mapped_(std::move(mapped)), config_(std::move(config)), interleaved_rope_(true) {
    config_.patch_width = 588;
    validate_config(config_);
    if (!mapped_ || mapped_->data() == nullptr || mapped_->size() == 0 ||
        mapped_->size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::runtime_error("native Glimmer mmproj payload is empty");
    }
    payload_ = mapped_->data();
    payload_nbytes_ = static_cast<std::int64_t>(mapped_->size());
    build_gguf_layout();
}

std::shared_ptr<Model> Model::load_bf16(
    const std::uint8_t* payload,
    std::int64_t payload_nbytes,
    Config config) {
    return std::shared_ptr<Model>(new Model(payload, payload_nbytes, std::move(config)));
}

std::shared_ptr<Model> Model::load_gguf(
    const std::string& checkpoint_path,
    Config config) {
    auto mapped = std::make_unique<MappedFile>(checkpoint_path);
    return std::shared_ptr<Model>(new Model(std::move(mapped), std::move(config)));
}

void Model::build_layout() {
    std::int64_t offset = 0;
    auto take = [&](std::int64_t rows, std::int64_t cols) -> Weight* {
        const std::int64_t elements = checked_mul(rows, cols, "vision tensor elements");
        const std::int64_t bytes = checked_mul(elements, 2, "vision tensor bytes");
        if (offset > payload_nbytes_ || bytes > payload_nbytes_ - offset) {
            throw std::runtime_error("native Glimmer vision tensor layout exceeds the BF16 payload");
        }
        auto weight = std::make_unique<Weight>();
        weight->data = payload_ + checked_size(offset, "vision tensor offset");
        weight->rows = rows;
        weight->cols = cols;
        weight->row_stride_bytes = checked_mul(cols, 2, "vision BF16 row stride");
        weight->nbytes = bytes;
        weight->encoding = Weight::Encoding::BF16;
        Weight* result = weight.get();
        weights_.push_back(std::move(weight));
        offset = checked_add(offset, bytes, "vision tensor layout");
        return result;
    };

    patch_ = take(config_.hidden_size, config_.patch_width);
    position_ = take(checked_mul(config_.position_side, config_.position_side, "position rows"), config_.hidden_size);
    pre_norm_weight_ = take(1, config_.hidden_size);
    pre_norm_bias_ = take(1, config_.hidden_size);
    post_norm_weight_ = take(1, config_.hidden_size);
    post_norm_bias_ = take(1, config_.hidden_size);
    layers_.reserve(checked_size(config_.num_layers, "vision layer count"));
    for (std::int64_t index = 0; index < config_.num_layers; ++index) {
        Layer layer;
        layer.q_weight = take(config_.hidden_size, config_.hidden_size);
        layer.q_bias = take(1, config_.hidden_size);
        layer.k_weight = take(config_.hidden_size, config_.hidden_size);
        layer.k_bias = take(1, config_.hidden_size);
        layer.v_weight = take(config_.hidden_size, config_.hidden_size);
        layer.v_bias = take(1, config_.hidden_size);
        layer.output_weight = take(config_.hidden_size, config_.hidden_size);
        layer.output_bias = take(1, config_.hidden_size);
        layer.norm1_weight = take(1, config_.hidden_size);
        layer.norm1_bias = take(1, config_.hidden_size);
        layer.norm2_weight = take(1, config_.hidden_size);
        layer.norm2_bias = take(1, config_.hidden_size);
        layer.fc1_weight = take(config_.intermediate_size, config_.hidden_size);
        layer.fc1_bias = take(1, config_.intermediate_size);
        layer.fc2_weight = take(config_.hidden_size, config_.intermediate_size);
        layer.fc2_bias = take(1, config_.hidden_size);
        layers_.push_back(layer);
    }
    adapter_fc1_ = take(config_.adapter_size, checked_mul(
        checked_mul(config_.merge_size, config_.merge_size, "merge area"),
        config_.hidden_size,
        "merged hidden width"));
    adapter_fc2_ = take(config_.adapter_size, config_.adapter_size);
    projection_ = take(config_.output_size, config_.adapter_size);
    if (offset != payload_nbytes_) {
        throw std::runtime_error("native Glimmer vision BF16 payload has trailing or missing tensors");
    }
}

void Model::build_gguf_layout() {
    constexpr std::size_t kCanonicalBytes = 1'400'328'928u;
    if (mapped_->size() != kCanonicalBytes) {
        throw std::runtime_error("Muse Glimmer mmproj has the wrong canonical byte length");
    }
    Cursor cursor(mapped_->data(), mapped_->size());
    cursor.require(4, "magic");
    if (std::memcmp(cursor.pointer(), "GGUF", 4) != 0) {
        throw std::runtime_error("Muse Glimmer mmproj magic is invalid");
    }
    cursor.skip(4, "magic");
    if (cursor.read<std::uint32_t>("version") != 3 ||
        cursor.read<std::uint64_t>("tensor count") != 809 ||
        cursor.read<std::uint64_t>("metadata count") != 19) {
        throw std::runtime_error("Muse Glimmer mmproj GGUF header contract is invalid");
    }
    const std::unordered_set<std::string> expected_metadata = {
        "general.architecture", "general.type", "general.name", "general.size_label",
        "clip.has_vision_encoder", "clip.vision.projection_dim",
        "clip.vision.image_size", "clip.vision.patch_size",
        "clip.vision.embedding_length", "clip.vision.feed_forward_length",
        "clip.vision.block_count", "clip.vision.attention.head_count",
        "clip.vision.image_mean", "clip.vision.image_std", "clip.projector_type",
        "clip.vision.attention.layer_norm_epsilon",
        "clip.vision.spatial_merge_size", "general.quantization_version",
        "general.file_type",
    };
    const std::unordered_map<std::string, std::uint64_t> expected_integers = {
        {"clip.vision.projection_dim", 6656}, {"clip.vision.image_size", 896},
        {"clip.vision.patch_size", 14}, {"clip.vision.embedding_length", 1536},
        {"clip.vision.feed_forward_length", 8960}, {"clip.vision.block_count", 50},
        {"clip.vision.attention.head_count", 16},
        {"clip.vision.spatial_merge_size", 2}, {"general.quantization_version", 2},
        {"general.file_type", 15},
    };
    std::unordered_set<std::string> seen_metadata;
    for (int entry = 0; entry < 19; ++entry) {
        const std::string key = cursor.string("metadata key");
        if (!expected_metadata.contains(key) || !seen_metadata.insert(key).second) {
            throw std::runtime_error("Muse Glimmer mmproj metadata allowlist mismatch at " + key);
        }
        const std::uint32_t type = cursor.read<std::uint32_t>("metadata type");
        if (key == "general.architecture") {
            require_string(&cursor, type, "clip", key.c_str());
        } else if (key == "general.type") {
            require_string(&cursor, type, "mmproj", key.c_str());
        } else if (key == "general.name") {
            require_string(&cursor, type, "Muse Glimmer Hf", key.c_str());
        } else if (key == "general.size_label") {
            require_string(&cursor, type, "1.9B", key.c_str());
        } else if (key == "clip.projector_type") {
            require_string(&cursor, type, "muse-glimmer", key.c_str());
        } else if (key == "clip.has_vision_encoder") {
            if (type != GgufBool || cursor.read<std::uint8_t>(key.c_str()) != 1) {
                throw std::runtime_error("Muse Glimmer mmproj has no canonical vision encoder");
            }
        } else if (key == "clip.vision.attention.layer_norm_epsilon") {
            if (type != GgufFloat32) {
                throw std::runtime_error("Muse Glimmer mmproj LayerNorm epsilon type is invalid");
            }
            const float value = cursor.read<float>(key.c_str());
            if (!std::isfinite(value) || std::abs(value - 1.0e-5f) > 1.0e-9f) {
                throw std::runtime_error("Muse Glimmer mmproj LayerNorm epsilon is invalid");
            }
        } else if (key == "clip.vision.image_mean" || key == "clip.vision.image_std") {
            if (type != GgufArray ||
                cursor.read<std::uint32_t>("image normalization type") != GgufFloat32 ||
                cursor.read<std::uint64_t>("image normalization length") != 3) {
                throw std::runtime_error("Muse Glimmer mmproj image normalization is malformed");
            }
            for (int channel = 0; channel < 3; ++channel) {
                const float value = cursor.read<float>("image normalization value");
                if (!std::isfinite(value) || std::abs(value - 0.5f) > 1.0e-7f) {
                    throw std::runtime_error("Muse Glimmer mmproj image normalization is invalid");
                }
            }
        } else {
            const auto expected = expected_integers.find(key);
            if (expected == expected_integers.end() ||
                read_integer(&cursor, type, key.c_str()) != expected->second) {
                throw std::runtime_error("Muse Glimmer mmproj scalar metadata mismatch at " + key);
            }
        }
    }
    if (seen_metadata != expected_metadata) {
        throw std::runtime_error("Muse Glimmer mmproj metadata allowlist is incomplete");
    }

    struct RawTensor {
        std::string name;
        std::vector<std::uint64_t> dimensions;
        std::uint32_t type = 0;
        std::uint64_t offset = 0;
    };
    const std::size_t tensor_table_start = cursor.offset();
    std::vector<RawTensor> raw;
    raw.reserve(809);
    std::unordered_set<std::string> names;
    for (int index = 0; index < 809; ++index) {
        RawTensor tensor;
        tensor.name = cursor.string("tensor name");
        if (!names.insert(tensor.name).second) {
            throw std::runtime_error("Muse Glimmer mmproj contains a duplicate tensor");
        }
        const std::uint32_t rank = cursor.read<std::uint32_t>("tensor rank");
        if (rank < 1 || rank > 4) {
            throw std::runtime_error("Muse Glimmer mmproj tensor rank is unsupported");
        }
        for (std::uint32_t dimension = 0; dimension < rank; ++dimension) {
            const std::uint64_t value = cursor.read<std::uint64_t>("tensor dimension");
            if (value == 0 || value > static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max())) {
                throw std::runtime_error("Muse Glimmer mmproj tensor dimension is invalid");
            }
            tensor.dimensions.push_back(value);
        }
        tensor.type = cursor.read<std::uint32_t>("tensor encoding");
        tensor.offset = cursor.read<std::uint64_t>("tensor offset");
        raw.push_back(std::move(tensor));
    }
    if (names != expected_mmproj_names()) {
        throw std::runtime_error("Muse Glimmer mmproj tensor allowlist mismatch");
    }
    const std::size_t tensor_table_end = cursor.offset();
    if (neuralfn::resident_support::sha256_hex(
            mapped_->data() + tensor_table_start,
            tensor_table_end - tensor_table_start) !=
        "47a880e1fde666694bf591879b3e8bbab6cff1a72ba883d959d3bf3cae4bea78") {
        throw std::runtime_error("Muse Glimmer mmproj tensor-table SHA-256 mismatch");
    }
    const std::size_t data_offset = align_up(tensor_table_end, 32);
    cursor.require(data_offset - tensor_table_end, "header padding");
    for (std::size_t index = tensor_table_end; index < data_offset; ++index) {
        if (mapped_->data()[index] != 0) {
            throw std::runtime_error("Muse Glimmer mmproj header padding is nonzero");
        }
    }

    struct Extent {
        std::uint64_t offset = 0;
        std::uint64_t nbytes = 0;
    };
    std::vector<Extent> extents;
    extents.reserve(raw.size());
    std::unordered_map<std::string, Weight*> tensors;
    std::unordered_map<std::uint32_t, int> inventory;
    for (const RawTensor& tensor : raw) {
        const TensorContract contract = mmproj_tensor_contract(tensor.name);
        if (tensor.dimensions != contract.dimensions || tensor.type != contract.type) {
            throw std::runtime_error("Muse Glimmer mmproj tensor contract mismatch at " + tensor.name);
        }
        const std::uint32_t encoding_type = validated_mmproj_encoding(tensor.type);
        const Weight::Encoding encoding = encoding_type == 0
            ? Weight::Encoding::F32
            : encoding_type == 12
            ? Weight::Encoding::Q4K
            : encoding_type == 14
            ? Weight::Encoding::Q6K
            : Weight::Encoding::BF16;
        std::int64_t block_elements = 1;
        std::int64_t block_bytes = tensor.type == 0 ? 4 : 2;
        if (tensor.type == 12) { block_elements = 256; block_bytes = 144; }
        if (tensor.type == 14) { block_elements = 256; block_bytes = 210; }
        if (contract.cols % block_elements != 0 || tensor.offset % 32 != 0) {
            throw std::runtime_error("Muse Glimmer mmproj block/alignment contract failed");
        }
        const std::int64_t row_stride = checked_mul(
            contract.cols / block_elements, block_bytes, "mmproj row stride");
        const std::int64_t nbytes = checked_mul(contract.rows, row_stride, "mmproj tensor bytes");
        if (tensor.offset > mapped_->size() - data_offset ||
            static_cast<std::uint64_t>(nbytes) >
                mapped_->size() - data_offset - static_cast<std::size_t>(tensor.offset)) {
            throw std::runtime_error("Muse Glimmer mmproj tensor exceeds its file");
        }
        auto weight = std::make_unique<Weight>();
        weight->data = mapped_->data() + data_offset + static_cast<std::size_t>(tensor.offset);
        weight->rows = contract.rows;
        weight->cols = contract.cols;
        weight->row_stride_bytes = row_stride;
        weight->nbytes = nbytes;
        weight->encoding = encoding;
        Weight* view = weight.get();
        weights_.push_back(std::move(weight));
        tensors.emplace(tensor.name, view);
        extents.push_back({tensor.offset, static_cast<std::uint64_t>(nbytes)});
        ++inventory[tensor.type];
    }
    if (inventory[0] != 506 || inventory[12] != 200 || inventory[14] != 100 ||
        inventory[30] != 3 || inventory.size() != 4) {
        throw std::runtime_error("Muse Glimmer mmproj tensor encoding inventory is invalid");
    }
    std::sort(extents.begin(), extents.end(), [](const Extent& left, const Extent& right) {
        return left.offset < right.offset;
    });
    std::uint64_t expected_offset = 0;
    for (const Extent& extent : extents) {
        if (extent.offset != expected_offset) {
            throw std::runtime_error("Muse Glimmer mmproj tensor data is not contiguous");
        }
        expected_offset = align_up(
            checked_size(checked_add(
                static_cast<std::int64_t>(extent.offset),
                static_cast<std::int64_t>(extent.nbytes),
                "mmproj extent"), "mmproj extent"),
            32);
    }
    const Extent& last = extents.back();
    if (data_offset + last.offset + last.nbytes != mapped_->size()) {
        throw std::runtime_error("Muse Glimmer mmproj has trailing or missing tensor bytes");
    }

    patch_ = tensors.at("v.patch_embd.weight");
    position_ = tensors.at("v.position_embd.weight");
    pre_norm_weight_ = tensors.at("v.pre_ln.weight");
    pre_norm_bias_ = tensors.at("v.pre_ln.bias");
    post_norm_weight_ = tensors.at("v.post_ln.weight");
    post_norm_bias_ = tensors.at("v.post_ln.bias");
    adapter_fc1_ = tensors.at("mm.0.weight");
    adapter_fc2_ = tensors.at("mm.1.weight");
    projection_ = tensors.at("mm.2.weight");
    layers_.reserve(50);
    for (int index = 0; index < 50; ++index) {
        const std::string prefix = "v.blk." + std::to_string(index) + ".";
        Layer layer;
        layer.q_weight = tensors.at(prefix + "attn_q.weight");
        layer.q_bias = tensors.at(prefix + "attn_q.bias");
        layer.k_weight = tensors.at(prefix + "attn_k.weight");
        layer.k_bias = tensors.at(prefix + "attn_k.bias");
        layer.v_weight = tensors.at(prefix + "attn_v.weight");
        layer.v_bias = tensors.at(prefix + "attn_v.bias");
        layer.output_weight = tensors.at(prefix + "attn_out.weight");
        layer.output_bias = tensors.at(prefix + "attn_out.bias");
        layer.norm1_weight = tensors.at(prefix + "ln1.weight");
        layer.norm1_bias = tensors.at(prefix + "ln1.bias");
        layer.norm2_weight = tensors.at(prefix + "ln2.weight");
        layer.norm2_bias = tensors.at(prefix + "ln2.bias");
        layer.fc1_weight = tensors.at(prefix + "ffn_up.weight");
        layer.fc1_bias = tensors.at(prefix + "ffn_up.bias");
        layer.fc2_weight = tensors.at(prefix + "ffn_down.weight");
        layer.fc2_bias = tensors.at(prefix + "ffn_down.bias");
        layers_.push_back(layer);
    }
}

neuralfn::resident_glimmer_cuda::VisionConfig Model::cuda_config() const {
    neuralfn::resident_glimmer_cuda::VisionConfig result;
    result.hidden_size = config_.hidden_size;
    result.intermediate_size = config_.intermediate_size;
    result.num_layers = config_.num_layers;
    result.num_heads = config_.num_heads;
    result.patch_width = config_.patch_width;
    result.merge_size = config_.merge_size;
    result.position_side = config_.position_side;
    result.adapter_size = config_.adapter_size;
    result.output_size = config_.output_size;
    result.rope_theta = config_.rope_theta;
    result.norm_eps = config_.norm_eps;
    result.interleaved_rope = interleaved_rope_;
    return result;
}

neuralfn::resident_glimmer_cuda::VisionHostWeightPlan
Model::cuda_weight_plan() const {
    using neuralfn::resident_glimmer_cuda::HostWeightView;
    using neuralfn::resident_glimmer_cuda::VisionHostLayer;
    using neuralfn::resident_glimmer_cuda::VisionHostLinear;
    using neuralfn::resident_glimmer_cuda::VisionHostWeightPlan;
    const auto view = [](const Weight& source) {
        HostWeightView result;
        result.data = source.data;
        result.rows = source.rows;
        result.cols = source.cols;
        result.row_stride_bytes = source.row_stride_bytes;
        result.nbytes = source.nbytes;
        switch (source.encoding) {
            case Weight::Encoding::F32: result.encoding = 0; break;
            case Weight::Encoding::BF16: result.encoding = 30; break;
            case Weight::Encoding::Q4K: result.encoding = 12; break;
            case Weight::Encoding::Q6K: result.encoding = 14; break;
        }
        return result;
    };
    const auto dense = [](const Weight& source) {
        std::vector<float> result(checked_size(
            checked_mul(source.rows, source.cols, "vision dense export"),
            "vision dense export"));
        for (std::int64_t row = 0; row < source.rows; ++row) {
            for (std::int64_t col = 0; col < source.cols; ++col) {
                result[checked_size(row * source.cols + col, "vision dense export")] =
                    source.at(row, col);
            }
        }
        return result;
    };
    const auto linear = [&](const Weight& weight, const Weight* bias) {
        VisionHostLinear result;
        result.weight = view(weight);
        if (bias != nullptr) result.bias = dense(*bias);
        return result;
    };
    VisionHostWeightPlan result;
    result.patch = view(*patch_);
    result.position = dense(*position_);
    result.pre_norm_weight = dense(*pre_norm_weight_);
    result.pre_norm_bias = dense(*pre_norm_bias_);
    result.post_norm_weight = dense(*post_norm_weight_);
    result.post_norm_bias = dense(*post_norm_bias_);
    result.layers.reserve(layers_.size());
    for (const Layer& source : layers_) {
        VisionHostLayer layer;
        layer.query = linear(*source.q_weight, source.q_bias);
        layer.key = linear(*source.k_weight, source.k_bias);
        layer.value = linear(*source.v_weight, source.v_bias);
        layer.output = linear(*source.output_weight, source.output_bias);
        layer.norm1_weight = dense(*source.norm1_weight);
        layer.norm1_bias = dense(*source.norm1_bias);
        layer.norm2_weight = dense(*source.norm2_weight);
        layer.norm2_bias = dense(*source.norm2_bias);
        layer.fc1 = linear(*source.fc1_weight, source.fc1_bias);
        layer.fc2 = linear(*source.fc2_weight, source.fc2_bias);
        result.layers.push_back(std::move(layer));
    }
    result.adapter_fc1 = linear(*adapter_fc1_, nullptr);
    result.adapter_fc2 = linear(*adapter_fc2_, nullptr);
    result.projection = linear(*projection_, nullptr);
    return result;
}

std::vector<float> Model::encode(
    const std::vector<float>& packed_patches,
    const std::vector<std::int64_t>& grid_thw,
    const std::atomic<bool>& cancelled) const {
    throw_if_cancelled(cancelled);
    if (packed_patches.empty() || packed_patches.size() % checked_size(config_.patch_width, "patch width") != 0 ||
        !std::all_of(packed_patches.begin(), packed_patches.end(), [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("native Glimmer vision patches must be finite packed rows of the exact patch width");
    }
    const std::int64_t patch_rows = static_cast<std::int64_t>(
        packed_patches.size() / checked_size(config_.patch_width, "patch width"));
    const VisionLayout layout = make_layout(grid_thw, config_, patch_rows);

    std::vector<float> hidden;
    linear(packed_patches, patch_rows, config_.patch_width, *patch_, nullptr, &hidden, cancelled);
    std::vector<float> positions = position_embedding(*position_, layout, config_, cancelled);
    for (std::size_t index = 0; index < hidden.size(); ++index) {
        hidden[index] += positions[index];
    }
    layer_norm_inplace(
        &hidden,
        patch_rows,
        config_.hidden_size,
        *pre_norm_weight_,
        *pre_norm_bias_,
        config_.norm_eps,
        cancelled);

    std::vector<float> reordered(hidden.size());
    for (std::int64_t row = 0; row < patch_rows; ++row) {
        const std::int64_t source = layout.window_index[checked_size(row, "window index")];
        std::copy_n(
            hidden.data() + source * config_.hidden_size,
            config_.hidden_size,
            reordered.data() + row * config_.hidden_size);
    }
    hidden.swap(reordered);

    for (std::int64_t layer_index = 0; layer_index < config_.num_layers; ++layer_index) {
        throw_if_cancelled(cancelled);
        const Layer& layer = layers_[checked_size(layer_index, "vision layer index")];
        std::vector<float> normalized = hidden;
        layer_norm_inplace(
            &normalized,
            patch_rows,
            config_.hidden_size,
            *layer.norm1_weight,
            *layer.norm1_bias,
            config_.norm_eps,
            cancelled);
        std::vector<float> q;
        std::vector<float> k;
        std::vector<float> v;
        linear(normalized, patch_rows, config_.hidden_size, *layer.q_weight, layer.q_bias, &q, cancelled);
        linear(normalized, patch_rows, config_.hidden_size, *layer.k_weight, layer.k_bias, &k, cancelled);
        linear(normalized, patch_rows, config_.hidden_size, *layer.v_weight, layer.v_bias, &v, cancelled);
        apply_rope_2d(&q, layout, config_, interleaved_rope_, cancelled);
        apply_rope_2d(&k, layout, config_, interleaved_rope_, cancelled);
        const bool full_attention = (layer_index + 1) % 4 == 0 || layer_index + 1 == config_.num_layers;
        std::vector<float> attended = attention(
            q,
            k,
            v,
            full_attention ? layout.full_boundaries : layout.window_boundaries,
            config_,
            cancelled);
        std::vector<float> projected;
        linear(attended, patch_rows, config_.hidden_size, *layer.output_weight, layer.output_bias, &projected, cancelled);
        for (std::size_t index = 0; index < hidden.size(); ++index) {
            hidden[index] += projected[index];
        }

        normalized = hidden;
        layer_norm_inplace(
            &normalized,
            patch_rows,
            config_.hidden_size,
            *layer.norm2_weight,
            *layer.norm2_bias,
            config_.norm_eps,
            cancelled);
        std::vector<float> mlp;
        linear(normalized, patch_rows, config_.hidden_size, *layer.fc1_weight, layer.fc1_bias, &mlp, cancelled);
        gelu_inplace(&mlp, cancelled);
        linear(mlp, patch_rows, config_.intermediate_size, *layer.fc2_weight, layer.fc2_bias, &projected, cancelled);
        for (std::size_t index = 0; index < hidden.size(); ++index) {
            hidden[index] += projected[index];
        }
    }

    reordered.assign(hidden.size(), 0.0f);
    for (std::int64_t row = 0; row < patch_rows; ++row) {
        const std::int64_t destination = layout.window_index[checked_size(row, "inverse window index")];
        std::copy_n(
            hidden.data() + row * config_.hidden_size,
            config_.hidden_size,
            reordered.data() + destination * config_.hidden_size);
    }
    hidden.swap(reordered);
    layer_norm_inplace(
        &hidden,
        patch_rows,
        config_.hidden_size,
        *post_norm_weight_,
        *post_norm_bias_,
        config_.norm_eps,
        cancelled);

    const std::int64_t merge_area = config_.merge_size * config_.merge_size;
    const std::int64_t merged_width = merge_area * config_.hidden_size;
    std::vector<float> merged(checked_size(
        checked_mul(layout.merged_rows, merged_width, "pixel shuffle output"),
        "pixel shuffle output"));
    std::int64_t output_row = 0;
    for (const GridRow& media : layout.grid) {
        const std::int64_t spatial = media.height * media.width;
        for (std::int64_t time = 0; time < media.temporal; ++time) {
            for (std::int64_t block_h = 0; block_h < media.height / config_.merge_size; ++block_h) {
                for (std::int64_t block_w = 0; block_w < media.width / config_.merge_size; ++block_w) {
                    float* destination = merged.data() + output_row * merged_width;
                    for (std::int64_t dim = 0; dim < config_.hidden_size; ++dim) {
                        std::int64_t slot = 0;
                        for (std::int64_t local_h = 0; local_h < config_.merge_size; ++local_h) {
                            for (std::int64_t local_w = 0; local_w < config_.merge_size; ++local_w) {
                                const std::int64_t source_row = media.offset + time * spatial +
                                    (block_h * config_.merge_size + local_h) * media.width +
                                    block_w * config_.merge_size + local_w;
                                destination[dim * merge_area + slot] =
                                    hidden[checked_size(source_row * config_.hidden_size + dim, "pixel shuffle source")];
                                ++slot;
                            }
                        }
                    }
                    ++output_row;
                }
            }
        }
    }
    if (output_row != layout.merged_rows) {
        throw std::runtime_error("native Glimmer vision pixel-shuffle row count is inconsistent");
    }

    std::vector<float> adapted;
    linear(merged, layout.merged_rows, merged_width, *adapter_fc1_, nullptr, &adapted, cancelled);
    gelu_inplace(&adapted, cancelled);
    std::vector<float> adapter_output;
    linear(adapted, layout.merged_rows, config_.adapter_size, *adapter_fc2_, nullptr, &adapter_output, cancelled);
    gelu_inplace(&adapter_output, cancelled);
    std::vector<float> output;
    linear(adapter_output, layout.merged_rows, config_.adapter_size, *projection_, nullptr, &output, cancelled);
    scaleless_rms_inplace(&output, layout.merged_rows, config_.output_size, config_.norm_eps, cancelled);
    return output;
}

}  // namespace neuralfn::resident_glimmer_vision
