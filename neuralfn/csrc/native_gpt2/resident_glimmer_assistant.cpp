#include "resident_glimmer_assistant.h"

#include "resident_glimmer.h"
#include "resident_glimmer_cuda.h"
#include "../native_train/tile_ops.h"
#include "resident_sha256.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#if defined(_WIN32)
#include <fstream>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace neuralfn::resident_glimmer_assistant {
namespace {

using neuralfn::resident_dense::ResidentCancellationError;

std::int64_t checked_add(std::int64_t left, std::int64_t right, const char* label) {
    if (left < 0 || right < 0 || left > std::numeric_limits<std::int64_t>::max() - right) {
        throw std::runtime_error(std::string("DFlash size overflow at ") + label);
    }
    return left + right;
}

std::int64_t checked_mul(std::int64_t left, std::int64_t right, const char* label) {
    if (left < 0 || right < 0 ||
        (left != 0 && right > std::numeric_limits<std::int64_t>::max() / left)) {
        throw std::runtime_error(std::string("DFlash size overflow at ") + label);
    }
    return left * right;
}

std::size_t checked_size(std::int64_t value, const char* label) {
    if (value < 0 || static_cast<std::uint64_t>(value) >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(std::string("DFlash host-size overflow at ") + label);
    }
    return static_cast<std::size_t>(value);
}

void throw_if_cancelled(const std::atomic<bool>& cancelled) {
    if (cancelled.load(std::memory_order_relaxed)) {
        throw ResidentCancellationError("resident inference session was cancelled");
    }
}

bool valid_sha256(const std::string& value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char ch) {
        return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
    });
}

float bf16_to_float(const std::uint8_t* source) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, source, sizeof(bits));
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16u);
}

float fp16_to_float(const std::uint8_t* source) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, source, sizeof(bits));
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16u;
    const std::uint32_t exponent = (bits >> 10u) & 0x1fu;
    const std::uint32_t mantissa = bits & 0x03ffu;
    std::uint32_t result = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            std::uint32_t normalized = mantissa;
            int shift = 0;
            while ((normalized & 0x0400u) == 0) {
                normalized <<= 1u;
                ++shift;
            }
            normalized &= 0x03ffu;
            result = sign | static_cast<std::uint32_t>(113 - shift) << 23u |
                normalized << 13u;
        }
    } else if (exponent == 0x1fu) {
        result = sign | 0x7f800000u | (mantissa << 13u);
    } else {
        result = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
    }
    return std::bit_cast<float>(result);
}

void k_scale_min(const std::uint8_t* scales, int index, int* scale, int* minimum) {
    if (index < 4) {
        *scale = scales[index] & 63;
        *minimum = scales[index + 4] & 63;
    } else {
        *scale = (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4);
        *minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
    }
}

class Cursor {
public:
    Cursor(const std::uint8_t* data, std::size_t size) : data_(data), size_(size) {}

    void require(std::size_t count, const char* label) const {
        if (count > size_ - std::min(offset_, size_)) {
            throw std::runtime_error(std::string("DFlash GGUF is truncated at ") + label);
        }
    }
    template <typename T>
    T read(const char* label) {
        require(sizeof(T), label);
        T value{};
        std::memcpy(&value, data_ + offset_, sizeof(T));
        offset_ += sizeof(T);
        return value;
    }
    void skip(std::size_t count, const char* label) {
        require(count, label);
        offset_ += count;
    }
    std::string string(const char* label) {
        const std::uint64_t length = read<std::uint64_t>(label);
        if (length > 32u * 1024u * 1024u || length > size_ - std::min(offset_, size_)) {
            throw std::runtime_error(std::string("DFlash GGUF string is invalid at ") + label);
        }
        std::string value(
            reinterpret_cast<const char*>(data_ + offset_), static_cast<std::size_t>(length));
        offset_ += static_cast<std::size_t>(length);
        if (value.empty() || value.find('\0') != std::string::npos) {
            throw std::runtime_error(std::string("DFlash GGUF string is invalid at ") + label);
        }
        return value;
    }
    const std::uint8_t* pointer() const noexcept { return data_ + offset_; }
    std::size_t offset() const noexcept { return offset_; }

private:
    const std::uint8_t* data_;
    std::size_t size_;
    std::size_t offset_ = 0;
};

enum GgufValueType : std::uint32_t {
    GgufUint8 = 0, GgufInt8 = 1, GgufUint16 = 2, GgufInt16 = 3,
    GgufUint32 = 4, GgufInt32 = 5, GgufFloat32 = 6, GgufBool = 7,
    GgufString = 8, GgufArray = 9, GgufUint64 = 10, GgufInt64 = 11,
    GgufFloat64 = 12,
};

void skip_gguf_value(Cursor* cursor, std::uint32_t type, int depth = 0) {
    if (depth > 1) throw std::runtime_error("DFlash GGUF nested arrays are unsupported");
    const std::unordered_map<std::uint32_t, std::size_t> widths = {
        {GgufUint8, 1}, {GgufInt8, 1}, {GgufUint16, 2}, {GgufInt16, 2},
        {GgufUint32, 4}, {GgufInt32, 4}, {GgufFloat32, 4}, {GgufBool, 1},
        {GgufUint64, 8}, {GgufInt64, 8}, {GgufFloat64, 8},
    };
    const auto width = widths.find(type);
    if (width != widths.end()) {
        cursor->skip(width->second, "metadata scalar");
        return;
    }
    if (type == GgufString) {
        (void)cursor->string("metadata string");
        return;
    }
    if (type != GgufArray) throw std::runtime_error("DFlash GGUF metadata type is unsupported");
    const std::uint32_t element_type = cursor->read<std::uint32_t>("array element type");
    if (element_type == GgufArray) throw std::runtime_error("DFlash GGUF nested arrays are unsupported");
    const std::uint64_t count = cursor->read<std::uint64_t>("array length");
    if (count > 1'000'000) throw std::runtime_error("DFlash GGUF array is too large");
    for (std::uint64_t index = 0; index < count; ++index) {
        skip_gguf_value(cursor, element_type, depth + 1);
    }
}

std::uint64_t read_integer_metadata(Cursor* cursor, std::uint32_t type, const char* label) {
    if (type == GgufUint32) return cursor->read<std::uint32_t>(label);
    if (type == GgufInt32) {
        const std::int32_t value = cursor->read<std::int32_t>(label);
        if (value < 0) throw std::runtime_error("DFlash GGUF integer is negative");
        return static_cast<std::uint64_t>(value);
    }
    if (type == GgufUint64) return cursor->read<std::uint64_t>(label);
    throw std::runtime_error(std::string("DFlash GGUF integer has wrong type at ") + label);
}

double read_float_metadata(Cursor* cursor, std::uint32_t type, const char* label) {
    double value = 0.0;
    if (type == GgufFloat32) value = cursor->read<float>(label);
    else if (type == GgufFloat64) value = cursor->read<double>(label);
    else throw std::runtime_error(std::string("DFlash GGUF float has wrong type at ") + label);
    if (!std::isfinite(value)) throw std::runtime_error("DFlash GGUF float is non-finite");
    return value;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0 ||
        value > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
        throw std::runtime_error("DFlash GGUF alignment is invalid");
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

template <typename Function>
void parallel_for(std::int64_t count, std::int64_t minimum_per_thread, Function&& function) {
    if (count <= 0) return;
    const unsigned hardware = std::max(1u, std::thread::hardware_concurrency());
    const std::int64_t wanted = (count + minimum_per_thread - 1) / minimum_per_thread;
    const std::int64_t workers = std::max<std::int64_t>(
        1, std::min<std::int64_t>({count, wanted, static_cast<std::int64_t>(hardware), 32}));
    if (workers == 1) {
        for (std::int64_t index = 0; index < count; ++index) function(index);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers));
    std::mutex error_mutex;
    std::exception_ptr error;
    for (std::int64_t worker = 0; worker < workers; ++worker) {
        const std::int64_t begin = count * worker / workers;
        const std::int64_t end = count * (worker + 1) / workers;
        threads.emplace_back([&, begin, end]() {
            try {
                for (std::int64_t index = begin; index < end; ++index) function(index);
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) error = std::current_exception();
            }
        });
    }
    for (auto& thread : threads) thread.join();
    if (error) std::rethrow_exception(error);
}

float silu(float value) {
    const double source = value;
    if (source >= 0.0) return static_cast<float>(source / (1.0 + std::exp(-source)));
    const double exponential = std::exp(source);
    return static_cast<float>(source * exponential / (1.0 + exponential));
}

void validate_config(
    const Config& config,
    const neuralfn::resident_glimmer::GlimmerModel& target) {
    if (config.max_seq_len <= 0 || config.model_dim <= 0 || config.intermediate_dim <= 0 ||
        config.num_layers <= 0 || config.num_heads <= 0 || config.num_kv_heads <= 0 ||
        config.num_heads % config.num_kv_heads != 0 || config.head_dim <= 0 ||
        config.head_dim % 2 != 0 || config.block_size < 2 ||
        config.sliding_window <= 0 || config.sliding_window > config.max_seq_len ||
        config.target_layer_ids.empty() ||
        !std::is_sorted(config.target_layer_ids.begin(), config.target_layer_ids.end()) ||
        std::adjacent_find(config.target_layer_ids.begin(), config.target_layer_ids.end()) !=
            config.target_layer_ids.end() ||
        config.target_layer_ids.front() < 0 ||
        config.target_layer_ids.back() >= target.num_layers() ||
        config.model_dim != target.model_dim() || config.max_seq_len != target.max_seq_len() ||
        config.mask_token_id < 0 || config.mask_token_id >= target.vocab_size()) {
        throw std::runtime_error("DFlash assistant/target geometry is invalid");
    }
    if (!std::isfinite(config.rope_theta) || !(config.rope_theta > 0.0) ||
        !std::isfinite(config.norm_eps) || !(config.norm_eps > 0.0) ||
        !valid_sha256(config.checkpoint_sha256)) {
        throw std::runtime_error("DFlash assistant numeric/fingerprint contract is invalid");
    }
    if (config.container == WeightContainer::GgufKQuant &&
        (config.max_seq_len != 131072 || config.model_dim != 6656 ||
         config.intermediate_dim != 19968 || config.num_layers != 5 ||
         config.num_heads != 32 || config.num_kv_heads != 8 ||
         config.head_dim != 128 || config.block_size != 16 ||
         config.mask_token_id != 201818 || config.sliding_window != 2048 ||
         config.rope_theta != 500000.0 || config.norm_eps != 1.0e-5 ||
         config.target_layer_ids != std::vector<std::int64_t>({1, 13, 25, 37, 49}))) {
        throw std::runtime_error("DFlash GGUF execution requires the pinned production geometry");
    }
}

}  // namespace

class Model::MappedFile final {
public:
    explicit MappedFile(const std::string& path) {
#if defined(_WIN32)
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error("failed to open DFlash checkpoint: " + path);
        const auto end = input.tellg();
        if (end <= 0 || static_cast<std::uint64_t>(end) >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error("DFlash checkpoint has an invalid byte length");
        }
        owned_.resize(static_cast<std::size_t>(end));
        input.seekg(0);
        input.read(reinterpret_cast<char*>(owned_.data()), end);
        if (!input) throw std::runtime_error("DFlash checkpoint is truncated");
        data_ = owned_.data();
        size_ = owned_.size();
#else
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) {
            throw std::runtime_error("failed to open DFlash checkpoint: " +
                                     std::string(std::strerror(errno)));
        }
        struct stat status {};
        if (::fstat(fd_, &status) != 0 || !S_ISREG(status.st_mode) || status.st_size <= 0 ||
            static_cast<std::uint64_t>(status.st_size) >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("DFlash checkpoint has an invalid byte length");
        }
        size_ = static_cast<std::size_t>(status.st_size);
        void* mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapped == MAP_FAILED) {
            const std::string message = std::strerror(errno);
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("failed to mmap DFlash checkpoint: " + message);
        }
        data_ = static_cast<const std::uint8_t*>(mapped);
#endif
    }

    ~MappedFile() {
#if !defined(_WIN32)
        if (data_ != nullptr) ::munmap(const_cast<std::uint8_t*>(data_), size_);
        if (fd_ >= 0) ::close(fd_);
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

struct Model::WeightView {
    enum class Encoding { F32, BF16, Q4K, Q5K, Q6K };
    const std::uint8_t* data = nullptr;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::int64_t row_stride_bytes = 0;
    std::int64_t nbytes = 0;
    Encoding encoding = Encoding::BF16;

    float value(std::int64_t row, std::int64_t col) const {
        if (data == nullptr || row < 0 || row >= rows || col < 0 || col >= cols) {
            throw std::runtime_error("DFlash weight access exceeds tensor extent");
        }
        const std::uint8_t* row_data = data + row * row_stride_bytes;
        if (encoding == Encoding::F32) {
            float value = 0.0f;
            std::memcpy(&value, row_data + col * 4, sizeof(value));
            return value;
        }
        if (encoding == Encoding::BF16) return bf16_to_float(row_data + col * 2);
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
            const int quant = high ? quants[group * 32 + lane] >> 4
                                   : quants[group * 32 + lane] & 0x0f;
            return d * static_cast<float>(scale * quant) - dmin * static_cast<float>(minimum);
        }
        if (encoding == Encoding::Q5K) {
            const std::uint8_t* block = row_data + block_index * 176;
            const float d = fp16_to_float(block);
            const float dmin = fp16_to_float(block + 2);
            const std::uint8_t* scales = block + 4;
            const std::uint8_t* high_bits = block + 16;
            const std::uint8_t* low_bits = block + 48;
            const int group = within / 64;
            const int lane = within % 32;
            const bool high_half = (within % 64) >= 32;
            int scale = 0;
            int minimum = 0;
            k_scale_min(scales, group * 2 + (high_half ? 1 : 0), &scale, &minimum);
            const int low = high_half ? low_bits[group * 32 + lane] >> 4
                                      : low_bits[group * 32 + lane] & 0x0f;
            const int mask = (high_half ? 2 : 1) << (2 * group);
            const int quant = low + ((high_bits[lane] & mask) ? 16 : 0);
            return d * static_cast<float>(scale * quant) - dmin * static_cast<float>(minimum);
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
        if (quarter == 0) quant = (low & 0x0f) | (((high >> 0) & 3) << 4);
        else if (quarter == 1) quant = (low & 0x0f) | (((high >> 2) & 3) << 4);
        else if (quarter == 2) quant = (low >> 4) | (((high >> 4) & 3) << 4);
        else quant = (low >> 4) | (((high >> 6) & 3) << 4);
        return d * static_cast<float>(scales[half * 8 + 2 * quarter]) *
            static_cast<float>(quant - 32);
    }
};

struct Model::Layer {
    WeightView q;
    WeightView k;
    WeightView v;
    WeightView output;
    WeightView q_norm;
    WeightView k_norm;
    WeightView mlp_gate;
    WeightView mlp_up;
    WeightView mlp_down;
    WeightView post_attention_norm;
    WeightView input_norm;
};

struct Session::LayerCache {
    std::vector<float> keys;
    std::vector<float> values;
};

namespace {

void linear_rows(
    const std::vector<float>& input,
    std::int64_t rows,
    const Model::WeightView& weight,
    std::vector<float>* output,
    const std::atomic<bool>& cancelled) {
    if (rows <= 0 || static_cast<std::int64_t>(input.size()) != rows * weight.cols) {
        throw std::runtime_error("DFlash linear input geometry is invalid");
    }
    output->assign(checked_size(checked_mul(rows, weight.rows, "linear output"),
                                "linear output"), 0.0f);
    parallel_for(checked_mul(rows, weight.rows, "linear rows"), 128, [&](std::int64_t flat) {
        if ((flat & 31) == 0) throw_if_cancelled(cancelled);
        const std::int64_t input_row = flat / weight.rows;
        const std::int64_t output_row = flat % weight.rows;
        double sum = 0.0;
        const float* source = input.data() + input_row * weight.cols;
        for (std::int64_t col = 0; col < weight.cols; ++col) {
            sum += static_cast<double>(source[col]) * weight.value(output_row, col);
        }
        const float result = static_cast<float>(sum);
        if (!std::isfinite(result)) throw std::runtime_error("DFlash linear produced non-finite output");
        (*output)[static_cast<std::size_t>(flat)] = result;
    });
}

void rms_rows(
    const std::vector<float>& input,
    std::int64_t rows,
    std::int64_t width,
    const Model::WeightView* weight,
    double eps,
    std::vector<float>* output,
    const std::atomic<bool>& cancelled) {
    if (rows <= 0 || width <= 0 || static_cast<std::int64_t>(input.size()) != rows * width ||
        (weight != nullptr && (weight->rows != 1 || weight->cols != width))) {
        throw std::runtime_error("DFlash RMSNorm geometry is invalid");
    }
    output->resize(input.size());
    parallel_for(rows, 2, [&](std::int64_t row) {
        throw_if_cancelled(cancelled);
        const float* source = input.data() + row * width;
        float* target = output->data() + row * width;
        double squares = 0.0;
        for (std::int64_t dim = 0; dim < width; ++dim) {
            squares += static_cast<double>(source[dim]) * source[dim];
        }
        const double inverse = 1.0 / std::sqrt(squares / static_cast<double>(width) + eps);
        for (std::int64_t dim = 0; dim < width; ++dim) {
            const double learned = weight == nullptr ? 1.0 : weight->value(0, dim);
            target[dim] = static_cast<float>(source[dim] * inverse * learned);
        }
    });
}

void qk_norm(
    std::vector<float>* value,
    std::int64_t rows,
    std::int64_t heads,
    std::int64_t head_dim,
    const Model::WeightView& weight,
    double eps,
    const std::atomic<bool>& cancelled) {
    if (static_cast<std::int64_t>(value->size()) != rows * heads * head_dim ||
        weight.rows != 1 || weight.cols != head_dim) {
        throw std::runtime_error("DFlash Q/K RMSNorm geometry is invalid");
    }
    parallel_for(rows * heads, 4, [&](std::int64_t flat) {
        throw_if_cancelled(cancelled);
        float* source = value->data() + flat * head_dim;
        double squares = 0.0;
        for (std::int64_t dim = 0; dim < head_dim; ++dim) {
            squares += static_cast<double>(source[dim]) * source[dim];
        }
        const double inverse = 1.0 / std::sqrt(squares / static_cast<double>(head_dim) + eps);
        for (std::int64_t dim = 0; dim < head_dim; ++dim) {
            source[dim] = static_cast<float>(source[dim] * inverse * weight.value(0, dim));
        }
    });
}

void rope_rows(
    std::vector<float>* value,
    std::int64_t rows,
    std::int64_t heads,
    std::int64_t head_dim,
    std::int64_t first_position,
    double theta,
    const std::atomic<bool>& cancelled) {
    parallel_for(rows * heads, 4, [&](std::int64_t flat) {
        throw_if_cancelled(cancelled);
        const std::int64_t row = flat / heads;
        float* source = value->data() + flat * head_dim;
        const std::int64_t half = head_dim / 2;
        for (std::int64_t index = 0; index < half; ++index) {
            const double inverse_frequency = 1.0 / std::pow(
                theta, static_cast<double>(2 * index) / static_cast<double>(head_dim));
            const double angle = static_cast<double>(first_position + row) * inverse_frequency;
            const double cosine = std::cos(angle);
            const double sine = std::sin(angle);
            const float first = source[index];
            const float second = source[index + half];
            source[index] = static_cast<float>(first * cosine - second * sine);
            source[index + half] = static_cast<float>(second * cosine + first * sine);
        }
    });
}

Model::WeightView::Encoding assistant_encoding_from_ggml(std::uint32_t type) {
    switch (type) {
        case 0: return Model::WeightView::Encoding::F32;
        case 12: return Model::WeightView::Encoding::Q4K;
        case 13: return Model::WeightView::Encoding::Q5K;
        case 14: return Model::WeightView::Encoding::Q6K;
        case 30: return Model::WeightView::Encoding::BF16;
        default: throw std::runtime_error("DFlash GGUF tensor encoding is unsupported");
    }
}

std::pair<std::int64_t, std::int64_t> expected_dflash_gguf_shape(
    const std::string& name) {
    if (name == "fc.weight") return {6656, 33280};
    if (name == "enc.output_norm.weight" || name == "output_norm.weight") {
        return {1, 6656};
    }
    if (!name.starts_with("blk.") || !name.ends_with(".weight")) {
        throw std::runtime_error("DFlash GGUF contains an unexpected tensor: " + name);
    }
    const std::size_t dot = name.find('.', 4);
    if (dot == std::string::npos) throw std::runtime_error("DFlash layer name is malformed");
    const std::string layer_text = name.substr(4, dot - 4);
    if (layer_text.empty() ||
        !std::all_of(layer_text.begin(), layer_text.end(), [](unsigned char value) {
            return std::isdigit(value) != 0;
        })) {
        throw std::runtime_error("DFlash layer index is malformed");
    }
    const int layer = std::stoi(layer_text);
    if (layer < 0 || layer >= 5) throw std::runtime_error("DFlash layer index is invalid");
    const std::string suffix = name.substr(dot + 1, name.size() - dot - 8);
    const std::unordered_map<std::string, std::pair<std::int64_t, std::int64_t>> shapes = {
        {"attn_norm", {1, 6656}}, {"ffn_down", {6656, 19968}},
        {"ffn_gate", {19968, 6656}}, {"ffn_up", {19968, 6656}},
        {"ffn_norm", {1, 6656}}, {"attn_k_norm", {1, 128}},
        {"attn_k", {1024, 6656}}, {"attn_output", {6656, 4096}},
        {"attn_q_norm", {1, 128}}, {"attn_q", {4096, 6656}},
        {"attn_v", {1024, 6656}},
    };
    const auto found = shapes.find(suffix);
    if (found == shapes.end()) throw std::runtime_error("DFlash layer tensor is unexpected");
    return found->second;
}

std::unordered_set<std::string> expected_dflash_gguf_names() {
    std::unordered_set<std::string> names = {
        "fc.weight", "enc.output_norm.weight", "output_norm.weight"};
    const std::vector<std::string> suffixes = {
        "attn_norm", "ffn_down", "ffn_gate", "ffn_up", "ffn_norm",
        "attn_k_norm", "attn_k", "attn_output", "attn_q_norm", "attn_q", "attn_v"};
    for (int layer = 0; layer < 5; ++layer) {
        for (const std::string& suffix : suffixes) {
            names.insert("blk." + std::to_string(layer) + "." + suffix + ".weight");
        }
    }
    return names;
}

}  // namespace

Model::Model(
    std::string checkpoint_path,
    Config config,
    std::shared_ptr<neuralfn::resident_glimmer::GlimmerModel> target,
    std::unique_ptr<MappedFile> mapped)
    : checkpoint_path_(std::move(checkpoint_path)),
      config_(std::move(config)),
      target_(std::move(target)),
      mapped_(std::move(mapped)) {
    build_layout();
    initialize_cuda_backend();
}

Model::~Model() { close(); }

std::shared_ptr<Model> Model::load(
    const std::string& checkpoint_path,
    Config config,
    std::shared_ptr<neuralfn::resident_glimmer::GlimmerModel> target) {
    if (!target) throw std::runtime_error("DFlash assistant requires a target model");
    validate_config(config, *target);
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error("DFlash checkpoints require a little-endian host");
    }
    auto mapped = std::make_unique<MappedFile>(checkpoint_path);
    if (!config.checkpoint_sha256_preverified) {
        const std::string digest = neuralfn::resident_support::sha256_hex(
            mapped->data(), mapped->size());
        if (digest != config.checkpoint_sha256) {
            throw std::runtime_error("DFlash checkpoint SHA-256 does not match its manifest");
        }
    }
    return std::shared_ptr<Model>(new Model(
        checkpoint_path, std::move(config), std::move(target), std::move(mapped)));
}

void Model::build_layout() {
    if (config_.container == WeightContainer::NativeBf16) {
        build_native_bf16_layout();
    } else {
        build_gguf_layout();
    }
}

void Model::build_native_bf16_layout() {
    std::int64_t offset = 0;
    auto take = [&](std::int64_t rows, std::int64_t cols) {
        WeightView result;
        result.data = mapped_->data() + checked_size(offset, "tensor offset");
        result.rows = rows;
        result.cols = cols;
        result.row_stride_bytes = checked_mul(cols, 2, "tensor row stride");
        result.nbytes = checked_mul(rows, result.row_stride_bytes, "tensor bytes");
        result.encoding = WeightView::Encoding::BF16;
        offset = checked_add(offset, result.nbytes, "tensor layout");
        if (offset > static_cast<std::int64_t>(mapped_->size())) {
            throw std::runtime_error("DFlash checkpoint tensor layout exceeds payload");
        }
        return result;
    };
    const std::int64_t query_width = config_.num_heads * config_.head_dim;
    const std::int64_t kv_width = config_.num_kv_heads * config_.head_dim;
    layers_.reserve(static_cast<std::size_t>(config_.num_layers));
    for (std::int64_t index = 0; index < config_.num_layers; ++index) {
        Layer layer;
        layer.q = take(query_width, config_.model_dim);
        layer.k = take(kv_width, config_.model_dim);
        layer.v = take(kv_width, config_.model_dim);
        layer.output = take(config_.model_dim, query_width);
        layer.q_norm = take(1, config_.head_dim);
        layer.k_norm = take(1, config_.head_dim);
        layer.mlp_gate = take(config_.intermediate_dim, config_.model_dim);
        layer.mlp_up = take(config_.intermediate_dim, config_.model_dim);
        layer.mlp_down = take(config_.model_dim, config_.intermediate_dim);
        layer.post_attention_norm = take(1, config_.model_dim);
        layer.input_norm = take(1, config_.model_dim);
        layers_.push_back(std::move(layer));
    }
    final_norm_ = std::make_unique<WeightView>(take(1, config_.model_dim));
    context_projection_ = std::make_unique<WeightView>(take(
        config_.model_dim,
        checked_mul(static_cast<std::int64_t>(config_.target_layer_ids.size()),
                    config_.model_dim, "target tap width")));
    context_norm_ = std::make_unique<WeightView>(take(1, config_.model_dim));
    if (offset != static_cast<std::int64_t>(mapped_->size())) {
        throw std::runtime_error("DFlash checkpoint has trailing or missing payload bytes");
    }
    parameter_count_ = offset / 2;
    weight_bytes_ = offset;
}

void Model::build_gguf_layout() {
    Cursor cursor(mapped_->data(), mapped_->size());
    cursor.require(4, "magic");
    if (std::memcmp(cursor.pointer(), "GGUF", 4) != 0) {
        throw std::runtime_error("DFlash GGUF magic is invalid");
    }
    cursor.skip(4, "magic");
    if (cursor.read<std::uint32_t>("version") != 3 ||
        cursor.read<std::uint64_t>("tensor count") != 58 ||
        cursor.read<std::uint64_t>("metadata count") != 33) {
        throw std::runtime_error("DFlash GGUF requires version 3, 58 tensors, and 33 metadata rows");
    }
    const std::unordered_set<std::string> expected_metadata = {
        "general.architecture", "general.type", "general.name", "general.size_label",
        "dflash.block_count", "dflash.context_length", "dflash.embedding_length",
        "dflash.feed_forward_length", "dflash.attention.head_count",
        "dflash.attention.head_count_kv", "dflash.rope.freq_base",
        "dflash.attention.layer_norm_rms_epsilon", "dflash.attention.key_length",
        "dflash.attention.value_length", "dflash.block_size", "dflash.target_layers",
        "dflash.attention.sliding_window", "dflash.attention.sliding_window_pattern",
        "general.quantization_version", "tokenizer.ggml.model", "tokenizer.ggml.pre",
        "tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges",
        "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id", "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_sep_token", "tokenizer.ggml.eot_token_id",
        "tokenizer.ggml.mask_token_id", "general.file_type", "tokenizer.chat_template",
    };
    std::unordered_set<std::string> seen_metadata;
    for (int entry = 0; entry < 33; ++entry) {
        const std::string key = cursor.string("metadata key");
        if (!seen_metadata.insert(key).second || !expected_metadata.contains(key)) {
            throw std::runtime_error("DFlash GGUF metadata allowlist mismatch at " + key);
        }
        const std::uint32_t type = cursor.read<std::uint32_t>("metadata type");
        if (key == "general.architecture") {
            if (type != GgufString || cursor.string("architecture") != "dflash") {
                throw std::runtime_error("DFlash GGUF architecture is not dflash");
            }
        } else if (key == "dflash.target_layers") {
            if (type != GgufArray ||
                cursor.read<std::uint32_t>("target layer type") != GgufInt32 ||
                cursor.read<std::uint64_t>("target layer count") != 5) {
                throw std::runtime_error("DFlash GGUF target layer metadata is malformed");
            }
            const std::array<std::int32_t, 5> expected = {2, 14, 26, 38, 50};
            for (std::int32_t value : expected) {
                if (cursor.read<std::int32_t>("target layer") != value) {
                    throw std::runtime_error("DFlash GGUF target layer metadata is incorrect");
                }
            }
        } else if (key == "dflash.attention.sliding_window_pattern") {
            if (type != GgufArray ||
                cursor.read<std::uint32_t>("window pattern type") != GgufBool ||
                cursor.read<std::uint64_t>("window pattern count") != 5) {
                throw std::runtime_error("DFlash GGUF window pattern is malformed");
            }
            for (int layer = 0; layer < 5; ++layer) {
                if (cursor.read<std::uint8_t>("window pattern") != 1) {
                    throw std::runtime_error("DFlash GGUF layers must all use sliding attention");
                }
            }
        } else if (
            key == "dflash.block_count" || key == "dflash.context_length" ||
            key == "dflash.embedding_length" || key == "dflash.feed_forward_length" ||
            key == "dflash.attention.head_count" || key == "dflash.attention.head_count_kv" ||
            key == "dflash.attention.key_length" || key == "dflash.attention.value_length" ||
            key == "dflash.block_size" || key == "dflash.attention.sliding_window" ||
            key == "general.quantization_version" || key == "general.file_type" ||
            key == "tokenizer.ggml.bos_token_id" || key == "tokenizer.ggml.eos_token_id" ||
            key == "tokenizer.ggml.padding_token_id" || key == "tokenizer.ggml.eot_token_id" ||
            key == "tokenizer.ggml.mask_token_id") {
            const std::unordered_map<std::string, std::uint64_t> expected = {
                {"dflash.block_count", 5}, {"dflash.context_length", 131072},
                {"dflash.embedding_length", 6656}, {"dflash.feed_forward_length", 19968},
                {"dflash.attention.head_count", 32}, {"dflash.attention.head_count_kv", 8},
                {"dflash.attention.key_length", 128}, {"dflash.attention.value_length", 128},
                {"dflash.block_size", 16}, {"dflash.attention.sliding_window", 2048},
                {"general.quantization_version", 2}, {"general.file_type", 15},
                {"tokenizer.ggml.bos_token_id", 200000},
                {"tokenizer.ggml.eos_token_id", 200001},
                {"tokenizer.ggml.padding_token_id", 200018},
                {"tokenizer.ggml.eot_token_id", 200008},
                {"tokenizer.ggml.mask_token_id", 201818},
            };
            if (read_integer_metadata(&cursor, type, key.c_str()) != expected.at(key)) {
                throw std::runtime_error("DFlash GGUF scalar metadata mismatch at " + key);
            }
        } else if (key == "dflash.rope.freq_base" ||
                   key == "dflash.attention.layer_norm_rms_epsilon") {
            const double expected = key == "dflash.rope.freq_base" ? 500000.0 : 1.0e-5;
            const double value = read_float_metadata(&cursor, type, key.c_str());
            if (std::abs(value - expected) > std::max(1.0e-9, std::abs(expected) * 1.0e-6)) {
                throw std::runtime_error("DFlash GGUF float metadata mismatch at " + key);
            }
        } else {
            skip_gguf_value(&cursor, type);
        }
    }
    if (seen_metadata != expected_metadata) {
        throw std::runtime_error("DFlash GGUF metadata allowlist is incomplete");
    }

    struct RawTensor {
        std::string name;
        std::vector<std::uint64_t> dimensions;
        std::uint32_t type = 0;
        std::uint64_t offset = 0;
    };
    std::vector<RawTensor> raw;
    raw.reserve(58);
    std::unordered_set<std::string> names;
    for (int index = 0; index < 58; ++index) {
        RawTensor tensor;
        tensor.name = cursor.string("tensor name");
        if (!names.insert(tensor.name).second) {
            throw std::runtime_error("DFlash GGUF contains duplicate tensors");
        }
        const std::uint32_t rank = cursor.read<std::uint32_t>("tensor rank");
        if (rank < 1 || rank > 2) throw std::runtime_error("DFlash GGUF tensor rank is invalid");
        for (std::uint32_t dimension = 0; dimension < rank; ++dimension) {
            const std::uint64_t value = cursor.read<std::uint64_t>("tensor dimension");
            if (value == 0 || value > static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max())) {
                throw std::runtime_error("DFlash GGUF tensor dimension is invalid");
            }
            tensor.dimensions.push_back(value);
        }
        tensor.type = cursor.read<std::uint32_t>("tensor type");
        tensor.offset = cursor.read<std::uint64_t>("tensor offset");
        raw.push_back(std::move(tensor));
    }
    if (names != expected_dflash_gguf_names()) {
        throw std::runtime_error("DFlash GGUF tensor allowlist mismatch");
    }
    const std::size_t data_offset = align_up(cursor.offset(), 32);
    cursor.require(data_offset - cursor.offset(), "header padding");
    for (std::size_t offset = cursor.offset(); offset < data_offset; ++offset) {
        if (mapped_->data()[offset] != 0) throw std::runtime_error("DFlash GGUF padding is nonzero");
    }

    std::unordered_map<std::string, WeightView> tensors;
    struct Extent { std::uint64_t offset; std::uint64_t nbytes; std::string name; };
    std::vector<Extent> extents;
    std::unordered_map<std::uint32_t, std::int64_t> inventory;
    std::int64_t logical_parameters = 0;
    std::int64_t resident_bytes = 0;
    for (const RawTensor& tensor : raw) {
        const auto [rows, cols] = expected_dflash_gguf_shape(tensor.name);
        if (tensor.dimensions[0] != static_cast<std::uint64_t>(cols) ||
            (tensor.dimensions.size() == 1 ? rows != 1
                                          : tensor.dimensions[1] != static_cast<std::uint64_t>(rows))) {
            throw std::runtime_error("DFlash GGUF tensor shape mismatch at " + tensor.name);
        }
        const WeightView::Encoding encoding = assistant_encoding_from_ggml(tensor.type);
        std::int64_t block_elements = 1;
        std::int64_t block_bytes = tensor.type == 0 ? 4 : 2;
        if (tensor.type == 12) { block_elements = 256; block_bytes = 144; }
        if (tensor.type == 13) { block_elements = 256; block_bytes = 176; }
        if (tensor.type == 14) { block_elements = 256; block_bytes = 210; }
        if (cols % block_elements != 0 || tensor.offset % 32 != 0) {
            throw std::runtime_error("DFlash GGUF tensor block/alignment contract failed");
        }
        const std::int64_t stride = checked_mul(cols / block_elements, block_bytes, "GGUF stride");
        const std::int64_t nbytes = checked_mul(rows, stride, "GGUF tensor bytes");
        if (data_offset > mapped_->size() || tensor.offset > mapped_->size() - data_offset ||
            static_cast<std::uint64_t>(nbytes) > mapped_->size() - data_offset - tensor.offset) {
            throw std::runtime_error("DFlash GGUF tensor exceeds its artifact");
        }
        WeightView view;
        view.data = mapped_->data() + data_offset + static_cast<std::size_t>(tensor.offset);
        view.rows = rows;
        view.cols = cols;
        view.row_stride_bytes = stride;
        view.nbytes = nbytes;
        view.encoding = encoding;
        tensors.emplace(tensor.name, view);
        extents.push_back({tensor.offset, static_cast<std::uint64_t>(nbytes), tensor.name});
        ++inventory[tensor.type];
        logical_parameters = checked_add(
            logical_parameters, checked_mul(rows, cols, "GGUF parameters"), "GGUF parameters");
        resident_bytes = checked_add(resident_bytes, nbytes, "GGUF resident bytes");
    }
    std::sort(extents.begin(), extents.end(), [](const Extent& left, const Extent& right) {
        return left.offset < right.offset;
    });
    std::uint64_t expected_offset = 0;
    for (const Extent& extent : extents) {
        if (extent.offset != expected_offset) {
            throw std::runtime_error("DFlash GGUF tensors are not canonical/contiguous");
        }
        expected_offset = align_up(static_cast<std::size_t>(extent.offset + extent.nbytes), 32);
    }
    const Extent& last = extents.back();
    if (data_offset + last.offset + last.nbytes != mapped_->size() ||
        inventory[0] != 22 || inventory[12] != 26 || inventory[13] != 0 ||
        inventory[14] != 10 || inventory[30] != 0) {
        throw std::runtime_error("DFlash GGUF extent or encoding inventory is not canonical");
    }

    context_projection_ = std::make_unique<WeightView>(tensors.at("fc.weight"));
    context_norm_ = std::make_unique<WeightView>(tensors.at("enc.output_norm.weight"));
    final_norm_ = std::make_unique<WeightView>(tensors.at("output_norm.weight"));
    layers_.reserve(5);
    for (int index = 0; index < 5; ++index) {
        const std::string prefix = "blk." + std::to_string(index) + ".";
        Layer layer;
        layer.input_norm = tensors.at(prefix + "attn_norm.weight");
        layer.post_attention_norm = tensors.at(prefix + "ffn_norm.weight");
        layer.q = tensors.at(prefix + "attn_q.weight");
        layer.k = tensors.at(prefix + "attn_k.weight");
        layer.v = tensors.at(prefix + "attn_v.weight");
        layer.output = tensors.at(prefix + "attn_output.weight");
        layer.q_norm = tensors.at(prefix + "attn_q_norm.weight");
        layer.k_norm = tensors.at(prefix + "attn_k_norm.weight");
        layer.mlp_gate = tensors.at(prefix + "ffn_gate.weight");
        layer.mlp_up = tensors.at(prefix + "ffn_up.weight");
        layer.mlp_down = tensors.at(prefix + "ffn_down.weight");
        layers_.push_back(std::move(layer));
    }
    parameter_count_ = logical_parameters;
    weight_bytes_ = resident_bytes;
}

void Model::require_open() const {
    if (closed()) throw std::runtime_error("DFlash assistant model is closed");
    if (target_->closed()) throw std::runtime_error("DFlash target model is closed");
}

void Model::initialize_cuda_backend() {
    if (!target_->whole_model_cuda()) return;
    using neuralfn::resident_glimmer_cuda::DFlashConfig;
    using neuralfn::resident_glimmer_cuda::DFlashHostLayerWeights;
    using neuralfn::resident_glimmer_cuda::DFlashHostWeightPlan;
    using neuralfn::resident_glimmer_cuda::HostWeightView;
    const auto convert = [](const WeightView& source) {
        HostWeightView result;
        result.data = source.data;
        result.rows = source.rows;
        result.cols = source.cols;
        result.row_stride_bytes = source.row_stride_bytes;
        result.nbytes = source.nbytes;
        switch (source.encoding) {
            case WeightView::Encoding::F32:
                result.encoding = NFN_NATIVE_TILE_PACKED_WEIGHT_F32;
                break;
            case WeightView::Encoding::BF16:
                result.encoding = NFN_NATIVE_TILE_PACKED_WEIGHT_BF16;
                break;
            case WeightView::Encoding::Q4K:
                result.encoding = NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K;
                break;
            case WeightView::Encoding::Q5K:
                result.encoding = NFN_NATIVE_TILE_PACKED_WEIGHT_Q5_K;
                break;
            case WeightView::Encoding::Q6K:
                result.encoding = NFN_NATIVE_TILE_PACKED_WEIGHT_Q6_K;
                break;
        }
        return result;
    };
    DFlashHostWeightPlan plan;
    plan.context_projection = convert(*context_projection_);
    plan.context_norm = convert(*context_norm_);
    plan.final_norm = convert(*final_norm_);
    plan.layers.reserve(layers_.size());
    for (const Layer& source : layers_) {
        DFlashHostLayerWeights layer;
        layer.input_norm = convert(source.input_norm);
        layer.post_attention_norm = convert(source.post_attention_norm);
        layer.q = convert(source.q);
        layer.k = convert(source.k);
        layer.v = convert(source.v);
        layer.output = convert(source.output);
        layer.q_norm = convert(source.q_norm);
        layer.k_norm = convert(source.k_norm);
        layer.mlp_gate = convert(source.mlp_gate);
        layer.mlp_up = convert(source.mlp_up);
        layer.mlp_down = convert(source.mlp_down);
        plan.layers.push_back(std::move(layer));
    }
    DFlashConfig cuda;
    cuda.max_seq_len = config_.max_seq_len;
    cuda.model_dim = config_.model_dim;
    cuda.intermediate_dim = config_.intermediate_dim;
    cuda.num_layers = config_.num_layers;
    cuda.num_heads = config_.num_heads;
    cuda.num_kv_heads = config_.num_kv_heads;
    cuda.head_dim = config_.head_dim;
    cuda.block_size = config_.block_size;
    cuda.tap_count = static_cast<std::int64_t>(config_.target_layer_ids.size());
    cuda.sliding_window = config_.sliding_window;
    cuda.rope_theta = static_cast<float>(config_.rope_theta);
    cuda.norm_eps = static_cast<float>(config_.norm_eps);
    // The official Muse-Glimmer target uses consecutive-pair RoPE, but its
    // legacy DFlash backbone is explicitly NeoX/half-split.  Container layout
    // does not change that architecture contract: treating the GGUF assistant
    // like the target rotates every draft Q/K head incorrectly and destroys
    // speculative acceptance.
    cuda.gguf_interleaved = false;
    cuda.cuda_device = target_->cuda_device();
    cuda.tile_ops_lib = target_->cuda_tile_ops_library();
    cuda.cuda_runtime_lib = target_->cuda_runtime_library();
    cuda_model_ = neuralfn::resident_glimmer_cuda::DFlashModel::load(cuda, plan);
}

std::unique_ptr<Session> Model::create_session() {
    require_open();
    return std::make_unique<Session>(shared_from_this());
}

void Model::close() noexcept {
    closed_.store(true);
    if (cuda_model_) cuda_model_->close();
}

std::int64_t Model::cuda_resident_weight_bytes() const noexcept {
    return cuda_model_ ? cuda_model_->resident_weight_bytes() : 0;
}

std::int64_t Model::cuda_workspace_bytes() const noexcept {
    return cuda_model_ ? cuda_model_->workspace_bytes() : 0;
}

std::int64_t Model::cuda_kernel_launches() const noexcept {
    return cuda_model_ ? cuda_model_->kernel_launches() : 0;
}

std::int64_t Model::cuda_k_quant_mmq_linears() const noexcept {
    return cuda_model_ ? cuda_model_->k_quant_mmq_linears() : 0;
}

bool Model::cuda_device_tap_pack() const noexcept {
    return cuda_model_ && cuda_model_->has_device_tap_pack();
}

Session::Session(std::shared_ptr<Model> model) : model_(std::move(model)) {
    if (!model_) throw std::runtime_error("DFlash session requires a model");
    layers_.resize(static_cast<std::size_t>(model_->config_.num_layers));
    if (model_->cuda_model_) cuda_cache_ = model_->cuda_model_->create_cache();
}

Session::~Session() = default;

void Session::reset() {
    for (LayerCache& layer : layers_) {
        layer.keys.clear();
        layer.values.clear();
    }
    pending_taps_.clear();
    pending_position_ = -1;
    lagged_anchor_position_ = -1;
    context_length_ = 0;
    if (model_->cuda_model_) cuda_cache_ = model_->cuda_model_->create_cache();
}

void Session::record_target_taps(
    std::int64_t position,
    const std::vector<float>& concatenated_taps,
    const std::atomic<bool>& cancelled) {
    model_->require_open();
    throw_if_cancelled(cancelled);
    const bool lagged_anchor = lagged_anchor_position_ >= 0;
    const bool contiguous = lagged_anchor
        ? (pending_position_ < 0 && pending_taps_.empty() &&
           context_length_ == lagged_anchor_position_ &&
           position == lagged_anchor_position_)
        : ((pending_position_ >= 0 && position == pending_position_ + 1) ||
           (pending_position_ < 0 && context_length_ == 0 && position == 0));
    if (position < 0 || position >= model_->config_.max_seq_len ||
        static_cast<std::int64_t>(concatenated_taps.size()) != model_->target_tap_width() ||
        !contiguous) {
        throw std::runtime_error("DFlash target tap stream is not contiguous/canonical");
    }
    if (pending_position_ >= 0) append_pending_context(cancelled);
    pending_taps_ = concatenated_taps;
    pending_position_ = position;
    lagged_anchor_position_ = -1;
}

void Session::record_target_taps_batch(
    std::int64_t start_position,
    const float* concatenated_taps,
    std::int64_t rows,
    const std::atomic<bool>& cancelled) {
    model_->require_open();
    throw_if_cancelled(cancelled);
    const std::int64_t tap_width = model_->target_tap_width();
    const bool lagged_anchor = lagged_anchor_position_ >= 0;
    const std::int64_t expected_start = lagged_anchor
        ? lagged_anchor_position_
        : (pending_position_ >= 0 ? pending_position_ + 1 : 0);
    if (concatenated_taps == nullptr || rows <= 0 ||
        rows > model_->config_.block_size || start_position < 0 ||
        start_position + rows > model_->config_.max_seq_len ||
        start_position != expected_start ||
        (lagged_anchor && (pending_position_ >= 0 || !pending_taps_.empty() ||
                           context_length_ != lagged_anchor_position_))) {
        throw std::runtime_error(
            "DFlash target tap batch is not contiguous/canonical");
    }
    if (!model_->cuda_model_) {
        for (std::int64_t row = 0; row < rows; ++row) {
            const float* first = concatenated_taps + row * tap_width;
            record_target_taps(
                start_position + row,
                std::vector<float>(first, first + tap_width), cancelled);
        }
        return;
    }
    if (!cuda_cache_) {
        throw std::runtime_error("DFlash CUDA cache is unavailable");
    }
    const bool had_pending = pending_position_ >= 0;
    const std::int64_t append_rows = had_pending ? rows : rows - 1;
    if (append_rows > 0) {
        std::vector<float> contexts;
        contexts.reserve(checked_size(checked_mul(
            append_rows, tap_width, "DFlash context tap batch"),
            "DFlash context tap batch"));
        if (had_pending) {
            contexts.insert(
                contexts.end(), pending_taps_.begin(), pending_taps_.end());
        }
        const std::int64_t incoming_rows = append_rows - (had_pending ? 1 : 0);
        contexts.insert(
            contexts.end(), concatenated_taps,
            concatenated_taps + incoming_rows * tap_width);
        model_->cuda_model_->append_contexts(
            contexts.data(), append_rows, context_length_, cuda_cache_, cancelled);
        context_length_ += append_rows;
    }
    const float* last = concatenated_taps + (rows - 1) * tap_width;
    pending_taps_.assign(last, last + tap_width);
    pending_position_ = start_position + rows - 1;
    lagged_anchor_position_ = -1;
    if (cuda_cache_->logical_length() != context_length_) {
        throw std::runtime_error(
            "DFlash CUDA batched context cache length is inconsistent");
    }
}

void Session::record_target_taps_batch_device(
    std::int64_t start_position,
    const float* tap_major_device,
    int source_cuda_device,
    std::int64_t source_rows,
    std::int64_t rows,
    const std::atomic<bool>& cancelled) {
    model_->require_open();
    throw_if_cancelled(cancelled);
    const bool lagged_anchor = lagged_anchor_position_ >= 0;
    const std::int64_t expected_start = lagged_anchor
        ? lagged_anchor_position_
        : (pending_position_ >= 0 ? pending_position_ + 1 : 0);
    if (!model_->cuda_model_ || !model_->cuda_model_->has_device_tap_pack() ||
        !cuda_cache_ || tap_major_device == nullptr || source_rows <= 0 ||
        source_rows > model_->config_.block_size || rows <= 0 || rows > source_rows ||
        start_position < 0 || start_position + rows > model_->config_.max_seq_len ||
        start_position != expected_start ||
        (lagged_anchor && (pending_position_ >= 0 || !pending_taps_.empty() ||
                           context_length_ != lagged_anchor_position_))) {
        throw std::runtime_error(
            "DFlash device target tap batch is not contiguous/canonical");
    }
    if (pending_position_ >= 0) {
        append_pending_context(cancelled);
        pending_taps_.clear();
        pending_position_ = -1;
    }
    pending_taps_ = model_->cuda_model_->append_contexts_device_tap_major(
        tap_major_device, source_cuda_device, source_rows, rows,
        context_length_, cuda_cache_, cancelled);
    context_length_ += rows - 1;
    pending_position_ = start_position + rows - 1;
    lagged_anchor_position_ = -1;
    if (cuda_cache_->logical_length() != context_length_) {
        throw std::runtime_error(
            "DFlash CUDA packed context cache length is inconsistent");
    }
}

void Session::record_target_taps_batch_device_and_prepare_lagged_anchor(
    std::int64_t start_position,
    const float* tap_major_device,
    int source_cuda_device,
    std::int64_t source_rows,
    std::int64_t rows,
    const std::atomic<bool>& cancelled) {
    model_->require_open();
    throw_if_cancelled(cancelled);
    if (!model_->cuda_model_ || !model_->cuda_model_->has_device_tap_pack() ||
        !cuda_cache_ || tap_major_device == nullptr || source_rows <= 0 ||
        source_rows > model_->config_.block_size || rows <= 0 ||
        rows > source_rows || start_position < 0 ||
        start_position + rows > model_->config_.max_seq_len ||
        lagged_anchor_position_ != start_position || pending_position_ >= 0 ||
        !pending_taps_.empty() || context_length_ != start_position) {
        throw std::runtime_error(
            "DFlash fused lagged target-tap batch is not contiguous/canonical");
    }
    model_->cuda_model_->append_contexts_device_tap_major_all(
        tap_major_device, source_cuda_device, source_rows, rows,
        start_position, cuda_cache_, cancelled);
    context_length_ += rows;
    lagged_anchor_position_ = context_length_;
    if (cuda_cache_->logical_length() != context_length_) {
        throw std::runtime_error(
            "DFlash fused lagged context cache length is inconsistent");
    }
}

void Session::prepare_lagged_anchor(
    std::int64_t anchor_position,
    const std::atomic<bool>& cancelled) {
    model_->require_open();
    throw_if_cancelled(cancelled);
    if (lagged_anchor_position_ >= 0) {
        if (lagged_anchor_position_ != anchor_position ||
            context_length_ != anchor_position || pending_position_ >= 0 ||
            !pending_taps_.empty()) {
            throw std::runtime_error("DFlash lagged anchor state is inconsistent");
        }
        return;
    }
    if (anchor_position <= 0 || pending_position_ != anchor_position - 1 ||
        pending_position_ != context_length_ || pending_taps_.empty()) {
        throw std::runtime_error("DFlash lagged anchor is not contiguous/canonical");
    }
    append_pending_context(cancelled);
    pending_taps_.clear();
    pending_position_ = -1;
    lagged_anchor_position_ = anchor_position;
    if (context_length_ != anchor_position ||
        (cuda_cache_ && cuda_cache_->logical_length() != context_length_)) {
        throw std::runtime_error("DFlash lagged anchor cache length is inconsistent");
    }
}

void Session::append_pending_context(const std::atomic<bool>& cancelled) {
    if (pending_position_ != context_length_ || pending_taps_.empty()) {
        throw std::runtime_error("DFlash pending target context is inconsistent");
    }
    if (model_->cuda_model_) {
        if (!cuda_cache_) throw std::runtime_error("DFlash CUDA cache is unavailable");
        model_->cuda_model_->append_context(
            pending_taps_.data(), pending_position_, cuda_cache_, cancelled);
        ++context_length_;
        return;
    }
    std::vector<float> projected;
    std::vector<float> normalized;
    linear_rows(pending_taps_, 1, *model_->context_projection_, &projected, cancelled);
    rms_rows(projected, 1, model_->config_.model_dim, model_->context_norm_.get(),
             model_->config_.norm_eps, &normalized, cancelled);
    const std::int64_t kv_width = model_->config_.num_kv_heads * model_->config_.head_dim;
    for (std::int64_t layer_index = 0; layer_index < model_->config_.num_layers; ++layer_index) {
        const Model::Layer& layer = model_->layers_[static_cast<std::size_t>(layer_index)];
        std::vector<float> key;
        std::vector<float> value;
        linear_rows(normalized, 1, layer.k, &key, cancelled);
        linear_rows(normalized, 1, layer.v, &value, cancelled);
        qk_norm(&key, 1, model_->config_.num_kv_heads, model_->config_.head_dim,
                layer.k_norm, model_->config_.norm_eps, cancelled);
        rope_rows(&key, 1, model_->config_.num_kv_heads, model_->config_.head_dim,
                  pending_position_, model_->config_.rope_theta, cancelled);
        LayerCache& cache = layers_[static_cast<std::size_t>(layer_index)];
        if (static_cast<std::int64_t>(cache.keys.size()) != context_length_ * kv_width ||
            cache.values.size() != cache.keys.size()) {
            throw std::runtime_error("DFlash accepted-context cache is inconsistent");
        }
        cache.keys.insert(cache.keys.end(), key.begin(), key.end());
        cache.values.insert(cache.values.end(), value.begin(), value.end());
    }
    ++context_length_;
}

Proposal Session::propose(
    std::int64_t anchor_token,
    std::int64_t proposal_tokens,
    const std::atomic<bool>& cancelled,
    bool require_logits,
    bool fast_k_quant) const {
    model_->require_open();
    throw_if_cancelled(cancelled);
    const bool ordinary_anchor = pending_position_ == context_length_ &&
        !pending_taps_.empty() && lagged_anchor_position_ < 0;
    const bool lagged_anchor = pending_position_ < 0 && pending_taps_.empty() &&
        lagged_anchor_position_ == context_length_;
    if ((!ordinary_anchor && !lagged_anchor) ||
        proposal_tokens <= 0 || proposal_tokens > model_->proposal_tokens() ||
        context_length_ + model_->config_.block_size > model_->config_.max_seq_len) {
        throw std::runtime_error("DFlash proposal state/count exceeds its exact block contract");
    }
    const std::int64_t block = model_->config_.block_size;
    const std::int64_t d = model_->config_.model_dim;
    const std::int64_t q_width = model_->config_.num_heads * model_->config_.head_dim;
    const std::int64_t kv_width = model_->config_.num_kv_heads * model_->config_.head_dim;
    std::vector<std::int64_t> input_tokens;
    input_tokens.reserve(static_cast<std::size_t>(block));
    for (std::int64_t row = 0; row < block; ++row) {
        input_tokens.push_back(
            row == 0 ? anchor_token : model_->config_.mask_token_id);
    }
    if (model_->cuda_model_) {
        if (!cuda_cache_ || cuda_cache_->logical_length() != context_length_) {
            throw std::runtime_error("DFlash CUDA proposal cache length is inconsistent");
        }
        Proposal proposal;
        proposal.token_ids.reserve(static_cast<std::size_t>(proposal_tokens));
        if (!require_logits) {
            const float* device_embeddings =
                model_->target_->raw_token_embeddings_device(input_tokens);
            const float* device_normalized = model_->cuda_model_->forward_block_device(
                device_embeddings, model_->target_->cuda_device(), block,
                cuda_cache_, cancelled);
            proposal.token_ids = model_->target_->raw_argmax_rows_from_device_hidden(
                device_normalized + d, model_->target_->cuda_device(),
                proposal_tokens, fast_k_quant);
            return proposal;
        }
        std::vector<float> hidden = model_->target_->raw_token_embeddings(input_tokens);
        std::vector<float> cuda_normalized = model_->cuda_model_->forward_block(
            hidden.data(), block, cuda_cache_, cancelled);
        proposal.logits = model_->target_->raw_logits_rows_from_hidden(
            cuda_normalized.data() + d, proposal_tokens, fast_k_quant);
        const std::int64_t vocab = model_->target_->vocab_size();
        for (std::int64_t row = 0; row < proposal_tokens; ++row) {
            const float* logits = proposal.logits.data() + row * vocab;
            const auto best = std::max_element(logits, logits + vocab);
            proposal.token_ids.push_back(static_cast<std::int64_t>(
                std::distance(logits, best)));
        }
        return proposal;
    }

    std::vector<float> hidden = model_->target_->raw_token_embeddings(input_tokens);

    std::vector<float> normalized;
    std::vector<float> query;
    std::vector<float> block_key;
    std::vector<float> block_value;
    std::vector<float> attention(checked_size(checked_mul(block, q_width, "attention"),
                                              "attention"));
    std::vector<float> projected;
    std::vector<float> mlp_gate;
    std::vector<float> mlp_up;
    std::vector<float> activated;
    std::vector<float> down;
    const double attention_scale = 1.0 / std::sqrt(static_cast<double>(model_->config_.head_dim));

    for (std::int64_t layer_index = 0; layer_index < model_->config_.num_layers; ++layer_index) {
        throw_if_cancelled(cancelled);
        const Model::Layer& layer = model_->layers_[static_cast<std::size_t>(layer_index)];
        const std::vector<float> residual = hidden;
        rms_rows(hidden, block, d, &layer.input_norm, model_->config_.norm_eps,
                 &normalized, cancelled);
        linear_rows(normalized, block, layer.q, &query, cancelled);
        linear_rows(normalized, block, layer.k, &block_key, cancelled);
        linear_rows(normalized, block, layer.v, &block_value, cancelled);
        qk_norm(&query, block, model_->config_.num_heads, model_->config_.head_dim,
                layer.q_norm, model_->config_.norm_eps, cancelled);
        qk_norm(&block_key, block, model_->config_.num_kv_heads, model_->config_.head_dim,
                layer.k_norm, model_->config_.norm_eps, cancelled);
        rope_rows(&query, block, model_->config_.num_heads, model_->config_.head_dim,
                  context_length_, model_->config_.rope_theta, cancelled);
        rope_rows(&block_key, block, model_->config_.num_kv_heads, model_->config_.head_dim,
                  context_length_, model_->config_.rope_theta, cancelled);
        const LayerCache& cache = layers_[static_cast<std::size_t>(layer_index)];
        if (static_cast<std::int64_t>(cache.keys.size()) != context_length_ * kv_width ||
            cache.values.size() != cache.keys.size()) {
            throw std::runtime_error("DFlash proposal cache length is inconsistent");
        }
        std::fill(attention.begin(), attention.end(), 0.0f);
        parallel_for(block * model_->config_.num_heads, 4, [&](std::int64_t flat) {
            throw_if_cancelled(cancelled);
            const std::int64_t query_row = flat / model_->config_.num_heads;
            const std::int64_t query_head = flat % model_->config_.num_heads;
            const std::int64_t kv_head = query_head * model_->config_.num_kv_heads /
                model_->config_.num_heads;
            const std::int64_t query_position = context_length_ + query_row;
            const std::int64_t first_position = std::max<std::int64_t>(
                0, query_position - model_->config_.sliding_window);
            const std::int64_t last_position = std::min<std::int64_t>(
                context_length_ + block - 1,
                query_position + model_->config_.sliding_window);
            std::vector<double> scores(static_cast<std::size_t>(last_position - first_position + 1));
            const float* query_head_row = query.data() +
                (query_row * model_->config_.num_heads + query_head) * model_->config_.head_dim;
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::int64_t key_position = first_position; key_position <= last_position;
                 ++key_position) {
                const float* key_row = key_position < context_length_
                    ? cache.keys.data() + key_position * kv_width +
                        kv_head * model_->config_.head_dim
                    : block_key.data() +
                        ((key_position - context_length_) * model_->config_.num_kv_heads + kv_head) *
                            model_->config_.head_dim;
                double score = 0.0;
                for (std::int64_t dim = 0; dim < model_->config_.head_dim; ++dim) {
                    score += static_cast<double>(query_head_row[dim]) * key_row[dim];
                }
                score *= attention_scale;
                scores[static_cast<std::size_t>(key_position - first_position)] = score;
                maximum = std::max(maximum, score);
            }
            double denominator = 0.0;
            for (double& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            if (!(denominator > 0.0) || !std::isfinite(denominator)) {
                throw std::runtime_error("DFlash attention probabilities are invalid");
            }
            for (std::int64_t dim = 0; dim < model_->config_.head_dim; ++dim) {
                double sum = 0.0;
                for (std::int64_t key_position = first_position; key_position <= last_position;
                     ++key_position) {
                    const float* value_row = key_position < context_length_
                        ? cache.values.data() + key_position * kv_width +
                            kv_head * model_->config_.head_dim
                        : block_value.data() +
                            ((key_position - context_length_) * model_->config_.num_kv_heads + kv_head) *
                                model_->config_.head_dim;
                    sum += scores[static_cast<std::size_t>(key_position - first_position)] /
                        denominator * value_row[dim];
                }
                attention[static_cast<std::size_t>(
                    (query_row * model_->config_.num_heads + query_head) *
                        model_->config_.head_dim + dim)] = static_cast<float>(sum);
            }
        });
        linear_rows(attention, block, layer.output, &projected, cancelled);
        hidden.resize(residual.size());
        for (std::size_t index = 0; index < hidden.size(); ++index) {
            hidden[index] = residual[index] + projected[index];
        }
        const std::vector<float> mlp_residual = hidden;
        rms_rows(hidden, block, d, &layer.post_attention_norm, model_->config_.norm_eps,
                 &normalized, cancelled);
        linear_rows(normalized, block, layer.mlp_gate, &mlp_gate, cancelled);
        linear_rows(normalized, block, layer.mlp_up, &mlp_up, cancelled);
        activated.resize(mlp_gate.size());
        for (std::size_t index = 0; index < activated.size(); ++index) {
            activated[index] = silu(mlp_gate[index]) * mlp_up[index];
        }
        linear_rows(activated, block, layer.mlp_down, &down, cancelled);
        for (std::size_t index = 0; index < hidden.size(); ++index) {
            hidden[index] = mlp_residual[index] + down[index];
        }
    }
    rms_rows(hidden, block, d, model_->final_norm_.get(), model_->config_.norm_eps,
             &normalized, cancelled);

    Proposal proposal;
    proposal.token_ids.reserve(static_cast<std::size_t>(proposal_tokens));
    if (require_logits) {
        proposal.logits.reserve(checked_size(checked_mul(
            proposal_tokens, model_->target_->vocab_size(), "proposal logits"),
            "proposal logits"));
    }
    for (std::int64_t row = 0; row < proposal_tokens; ++row) {
        const float* hidden_row = normalized.data() + (row + 1) * d;
        std::vector<float> logits = model_->target_->raw_logits_from_hidden(hidden_row);
        const auto best = std::max_element(logits.begin(), logits.end());
        proposal.token_ids.push_back(static_cast<std::int64_t>(
            std::distance(logits.begin(), best)));
        if (require_logits) {
            proposal.logits.insert(proposal.logits.end(), logits.begin(), logits.end());
        }
    }
    return proposal;
}

std::int64_t Session::cache_bytes() const noexcept {
    if (cuda_cache_) {
        return cuda_cache_->allocated_bytes() +
            static_cast<std::int64_t>(pending_taps_.capacity()) *
                static_cast<std::int64_t>(sizeof(float));
    }
    std::int64_t values = static_cast<std::int64_t>(pending_taps_.capacity());
    for (const LayerCache& layer : layers_) {
        values += static_cast<std::int64_t>(layer.keys.capacity() + layer.values.capacity());
    }
    return values * static_cast<std::int64_t>(sizeof(float));
}

}  // namespace neuralfn::resident_glimmer_assistant
