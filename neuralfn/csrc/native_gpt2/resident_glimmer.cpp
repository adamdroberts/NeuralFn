#include "resident_glimmer.h"
#include "resident_glimmer_assistant.h"
#include "resident_glimmer_cuda.h"
#include "resident_glimmer_vision.h"
#include "resident_sha256.h"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <numeric>
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

namespace neuralfn::resident_glimmer {
namespace {

using neuralfn::resident_dense::ResidentCancellationError;

constexpr std::int64_t kProductionTextParameters = 27'854'780'928LL;
constexpr std::int64_t kProductionTextBytes = 55'709'561'856LL;
constexpr std::int64_t kProductionVisionBytes = 3'843'691'520LL;

std::int64_t checked_add(std::int64_t left, std::int64_t right, const char* label) {
    if (left < 0 || right < 0 || left > std::numeric_limits<std::int64_t>::max() - right) {
        throw std::runtime_error(std::string("native Glimmer size overflow at ") + label);
    }
    return left + right;
}

std::int64_t checked_mul(std::int64_t left, std::int64_t right, const char* label) {
    if (left < 0 || right < 0 ||
        (left != 0 && right > std::numeric_limits<std::int64_t>::max() / left)) {
        throw std::runtime_error(std::string("native Glimmer size overflow at ") + label);
    }
    return left * right;
}

std::size_t checked_size(std::int64_t value, const char* label) {
    if (value < 0 || static_cast<std::uint64_t>(value) >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(std::string("native Glimmer host-size overflow at ") + label);
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

std::int64_t text_parameter_count(const GlimmerInferenceConfig& config) {
    const std::int64_t d = config.model_dim;
    const std::int64_t q = checked_mul(config.num_heads, config.head_dim, "query width");
    const std::int64_t kv = checked_mul(config.num_kv_heads, config.head_dim, "KV width");
    const std::int64_t f = config.intermediate_dim;
    std::int64_t layer = checked_mul(4, d, "four layer norms");
    layer = checked_add(layer, checked_mul(q, d, "query projection"), "query projection");
    layer = checked_add(layer, checked_mul(kv, d, "key projection"), "key projection");
    layer = checked_add(layer, checked_mul(kv, d, "value projection"), "value projection");
    layer = checked_add(layer, checked_mul(q, d, "attention gate"), "attention gate");
    layer = checked_add(layer, checked_mul(d, q, "attention output"), "attention output");
    layer = checked_add(
        layer,
        checked_mul(3, checked_mul(f, d, "MLP projection"), "three MLP projections"),
        "MLP projections");
    std::int64_t total = checked_mul(
        2, checked_mul(config.vocab_size, d, "embedding/head"), "embedding and head");
    total = checked_add(total, d, "final norm");
    return checked_add(
        total,
        checked_mul(config.num_layers, layer, "decoder layers"),
        "decoder parameters");
}

bool production_geometry(const GlimmerInferenceConfig& config) {
    return config.max_seq_len == 131072 && config.vocab_size == 202048 &&
        config.num_layers == 52 && config.model_dim == 6656 &&
        config.intermediate_dim == 19968 && config.num_heads == 32 &&
        config.num_kv_heads == 2 && config.head_dim == 128 &&
        config.sliding_window == 2048 && config.rope_theta == 500000.0 &&
        config.norm_eps == 1.0e-5 && config.post_norm_eps == 1.0e-8 &&
        config.q_scale_factor == 3.87 &&
        config.output_multiplier == 0.19611613513818404 &&
        config.logit_softcap == 20.0;
}

void validate_config(const GlimmerInferenceConfig& config) {
    if (config.max_seq_len <= 0 || config.vocab_size <= 0 || config.num_layers <= 0 ||
        config.num_layers % 4 != 0 || config.model_dim <= 0 ||
        config.intermediate_dim <= 0 || config.num_heads <= 0 ||
        config.num_kv_heads <= 0 || config.num_heads % config.num_kv_heads != 0 ||
        config.head_dim <= 0 || config.head_dim % 2 != 0 || config.sliding_window <= 0 ||
        config.sliding_window > config.max_seq_len) {
        throw std::runtime_error("native Glimmer checkpoint has invalid decoder geometry");
    }
    for (const auto& [value, label] : std::initializer_list<std::pair<double, const char*>>{
             {config.rope_theta, "RoPE theta"},
             {config.norm_eps, "RMSNorm epsilon"},
             {config.post_norm_eps, "post RMSNorm epsilon"},
             {config.q_scale_factor, "query scale"},
             {config.output_multiplier, "output multiplier"},
             {config.logit_softcap, "logit softcap"},
         }) {
        if (!std::isfinite(value) || !(value > 0.0)) {
            throw std::runtime_error(std::string("native Glimmer ") + label + " must be positive");
        }
    }
    if (!valid_sha256(config.checkpoint_sha256)) {
        throw std::runtime_error("native Glimmer checkpoint requires a lowercase SHA-256 fingerprint");
    }
    if (config.cuda_device < 0 ||
        (config.whole_model_cuda && config.tile_ops_lib.empty()) ||
        (!config.whole_model_cuda &&
         (!config.tile_ops_lib.empty() || !config.cuda_runtime_lib.empty() ||
          config.cuda_device != 0))) {
        throw std::runtime_error("native Glimmer CUDA load options are inconsistent");
    }
    if (config.container == WeightContainer::GgufKQuant && !production_geometry(config)) {
        throw std::runtime_error("native Glimmer GGUF execution only accepts the pinned production geometry");
    }
    const std::int64_t parameters = text_parameter_count(config);
    if (production_geometry(config) && parameters != kProductionTextParameters) {
        throw std::runtime_error("native Glimmer production parameter contract is inconsistent");
    }
}

float bf16_to_float(const std::uint8_t* source) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, source, sizeof(bits));
    const std::uint32_t fp32 = static_cast<std::uint32_t>(bits) << 16u;
    return std::bit_cast<float>(fp32);
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
        value = std::ldexp(1.0 + static_cast<double>(mantissa) / 1024.0, exponent - 15);
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
void parallel_for(std::int64_t count, std::int64_t minimum_per_thread, Function&& function) {
    if (count <= 0) {
        return;
    }
    const unsigned hardware = std::max(1u, std::thread::hardware_concurrency());
    const std::int64_t wanted = (count + minimum_per_thread - 1) / minimum_per_thread;
    const std::int64_t workers = std::max<std::int64_t>(
        1, std::min<std::int64_t>({count, wanted, static_cast<std::int64_t>(hardware), 32}));
    if (workers == 1) {
        for (std::int64_t index = 0; index < count; ++index) {
            function(index);
        }
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
                for (std::int64_t index = begin; index < end; ++index) {
                    {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (error) {
                            return;
                        }
                    }
                    function(index);
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

float silu(float value) {
    const double source = value;
    if (source >= 0.0) {
        return static_cast<float>(source / (1.0 + std::exp(-source)));
    }
    const double exponential = std::exp(source);
    return static_cast<float>(source * exponential / (1.0 + exponential));
}

bool contains_token(const std::vector<std::int64_t>& values, std::int64_t token) {
    return std::find(values.begin(), values.end(), token) != values.end();
}

std::int64_t argmax_token(const std::vector<float>& logits) {
    if (logits.empty()) throw std::runtime_error("speculative logits are empty");
    return static_cast<std::int64_t>(std::distance(
        logits.begin(), std::max_element(logits.begin(), logits.end())));
}

std::int64_t sample_probabilities(
    const std::vector<double>& probabilities,
    std::mt19937_64& rng) {
    if (probabilities.empty()) {
        throw std::runtime_error("speculative sampling distribution is empty");
    }
    std::discrete_distribution<std::size_t> distribution(
        probabilities.begin(), probabilities.end());
    return static_cast<std::int64_t>(distribution(rng));
}

std::vector<double> processed_probabilities(
    const std::vector<float>& logits,
    const GenerationConfig& config) {
    if (logits.empty() || !std::isfinite(config.temperature) || !(config.temperature > 0.0) ||
        config.top_k < 0 || !std::isfinite(config.top_p) ||
        !(config.top_p > 0.0) || config.top_p > 1.0) {
        throw std::runtime_error("speculative sampling configuration is invalid");
    }
    std::vector<std::int64_t> candidates(logits.size());
    std::iota(candidates.begin(), candidates.end(), 0);
    std::sort(candidates.begin(), candidates.end(), [&](std::int64_t left, std::int64_t right) {
        const float lhs = logits[static_cast<std::size_t>(left)];
        const float rhs = logits[static_cast<std::size_t>(right)];
        return lhs == rhs ? left < right : lhs > rhs;
    });
    if (config.top_k > 0 && config.top_k < static_cast<std::int64_t>(candidates.size())) {
        candidates.resize(static_cast<std::size_t>(config.top_k));
    }
    const double maximum = logits[static_cast<std::size_t>(candidates.front())];
    std::vector<double> candidate_weights;
    candidate_weights.reserve(candidates.size());
    double total = 0.0;
    for (std::int64_t token : candidates) {
        const double weight = std::exp(std::max(
            -745.0,
            (static_cast<double>(logits[static_cast<std::size_t>(token)]) - maximum) /
                config.temperature));
        candidate_weights.push_back(weight);
        total += weight;
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        throw std::runtime_error("speculative processed probabilities are invalid");
    }
    double cumulative = 0.0;
    std::size_t retained = candidate_weights.size();
    for (std::size_t index = 0; index < candidate_weights.size(); ++index) {
        cumulative += candidate_weights[index] / total;
        if (cumulative >= config.top_p) {
            retained = index + 1;
            break;
        }
    }
    candidates.resize(retained);
    candidate_weights.resize(retained);
    total = std::accumulate(candidate_weights.begin(), candidate_weights.end(), 0.0);
    std::vector<double> probabilities(logits.size(), 0.0);
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        probabilities[static_cast<std::size_t>(candidates[index])] =
            candidate_weights[index] / total;
    }
    return probabilities;
}

class Cursor final {
public:
    Cursor(const std::uint8_t* data, std::size_t size) : data_(data), size_(size) {}

    std::size_t offset() const noexcept { return offset_; }
    const std::uint8_t* pointer() const noexcept { return data_ + offset_; }

    void require(std::size_t count, const char* label) const {
        if (count > size_ - std::min(size_, offset_)) {
            throw std::runtime_error(std::string("Muse Glimmer GGUF is truncated at ") + label);
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
        if (length > 32u * 1024u * 1024u || length > std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("Muse Glimmer GGUF string exceeds the 32 MiB bound");
        }
        require(static_cast<std::size_t>(length), label);
        std::string result(
            reinterpret_cast<const char*>(data_ + offset_), static_cast<std::size_t>(length));
        offset_ += static_cast<std::size_t>(length);
        if (result.empty() || result.find('\0') != std::string::npos) {
            throw std::runtime_error("Muse Glimmer GGUF key/name is empty or contains NUL");
        }
        return result;
    }

private:
    const std::uint8_t* data_;
    std::size_t size_;
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

void skip_gguf_value(Cursor* cursor, std::uint32_t type, int depth = 0) {
    if (depth > 1) {
        throw std::runtime_error("nested GGUF arrays are unsupported");
    }
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
    if (type != GgufArray) {
        throw std::runtime_error("Muse Glimmer GGUF uses an unsupported metadata type");
    }
    const std::uint32_t element_type = cursor->read<std::uint32_t>("array element type");
    if (element_type == GgufArray) {
        throw std::runtime_error("nested GGUF arrays are unsupported");
    }
    const std::uint64_t count = cursor->read<std::uint64_t>("array length");
    if (count > 1'000'000) {
        throw std::runtime_error("Muse Glimmer GGUF array exceeds 1,000,000 elements");
    }
    for (std::uint64_t index = 0; index < count; ++index) {
        skip_gguf_value(cursor, element_type, depth + 1);
    }
}

std::uint64_t read_integer_metadata(Cursor* cursor, std::uint32_t type, const char* label) {
    if (type == GgufUint32) {
        return cursor->read<std::uint32_t>(label);
    }
    if (type == GgufUint64) {
        return cursor->read<std::uint64_t>(label);
    }
    throw std::runtime_error(std::string("Muse Glimmer GGUF ") + label + " has the wrong type");
}

double read_float_metadata(Cursor* cursor, std::uint32_t type, const char* label) {
    double value = 0.0;
    if (type == GgufFloat32) {
        value = cursor->read<float>(label);
    } else if (type == GgufFloat64) {
        value = cursor->read<double>(label);
    } else {
        throw std::runtime_error(std::string("Muse Glimmer GGUF ") + label + " has the wrong type");
    }
    if (!std::isfinite(value)) {
        throw std::runtime_error(std::string("Muse Glimmer GGUF ") + label + " is non-finite");
    }
    return value;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0 || alignment > 4096 ||
        value > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
        throw std::runtime_error("Muse Glimmer GGUF alignment is invalid");
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

}  // namespace

class GlimmerModel::MappedFile final {
public:
    explicit MappedFile(const std::string& path) {
#if defined(_WIN32)
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input) {
            throw std::runtime_error("failed to open native Glimmer checkpoint: " + path);
        }
        const auto end = input.tellg();
        if (end <= 0 || static_cast<std::uint64_t>(end) >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error("native Glimmer checkpoint has an invalid byte length");
        }
        owned_.resize(static_cast<std::size_t>(end));
        input.seekg(0);
        input.read(reinterpret_cast<char*>(owned_.data()), end);
        if (!input) {
            throw std::runtime_error("native Glimmer checkpoint is truncated");
        }
        data_ = owned_.data();
        size_ = owned_.size();
#else
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) {
            throw std::runtime_error(
                "failed to open native Glimmer checkpoint: " + std::string(std::strerror(errno)));
        }
        struct stat status {};
        if (::fstat(fd_, &status) != 0 || !S_ISREG(status.st_mode) || status.st_size <= 0 ||
            static_cast<std::uint64_t>(status.st_size) >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            const std::string message = std::strerror(errno);
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("native Glimmer checkpoint has an invalid byte length: " + message);
        }
        size_ = static_cast<std::size_t>(status.st_size);
        void* mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapped == MAP_FAILED) {
            const std::string message = std::strerror(errno);
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("failed to mmap native Glimmer checkpoint: " + message);
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

struct GlimmerModel::WeightView {
    enum class Encoding {
        F32,
        BF16,
        Q4K,
        Q5K,
        Q6K,
    };

    const std::uint8_t* data = nullptr;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::int64_t row_stride_bytes = 0;
    std::int64_t nbytes = 0;
    Encoding encoding = Encoding::BF16;
    bool centered = false;

    float value(std::int64_t row, std::int64_t col) const {
        if (row < 0 || row >= rows || col < 0 || col >= cols || data == nullptr) {
            throw std::runtime_error("native Glimmer weight index is outside its tensor extent");
        }
        const std::uint8_t* row_data = data + row * row_stride_bytes;
        if (encoding == Encoding::F32) {
            float result = 0.0f;
            std::memcpy(&result, row_data + col * 4, sizeof(result));
            return result;
        }
        if (encoding == Encoding::BF16) {
            return bf16_to_float(row_data + col * 2);
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
            const int low = high_half ? (low_bits[group * 32 + lane] >> 4)
                                      : (low_bits[group * 32 + lane] & 0x0f);
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

struct GlimmerModel::LayerLayout {
    WeightView input_norm;
    WeightView post_attention_norm;
    WeightView pre_feedforward_norm;
    WeightView post_feedforward_norm;
    WeightView q;
    WeightView k;
    WeightView v;
    WeightView gate;
    WeightView output;
    WeightView mlp_gate;
    WeightView mlp_up;
    WeightView mlp_down;
    std::optional<WeightView> q_norm;
    std::optional<WeightView> k_norm;
};

struct GlimmerModel::LoraWeight {
    WeightView a;
    WeightView b;
    double scaling = 1.0;
};

struct GlimmerModel::LoraLayerLayout {
    std::optional<LoraWeight> q;
    std::optional<LoraWeight> k;
    std::optional<LoraWeight> v;
    std::optional<LoraWeight> gate;
    std::optional<LoraWeight> output;
    std::optional<LoraWeight> mlp_gate;
    std::optional<LoraWeight> mlp_up;
    std::optional<LoraWeight> mlp_down;
};

namespace {

void linear(
    const std::vector<float>& input,
    const GlimmerModel::WeightView& weight,
    std::vector<float>* output,
    const std::atomic<bool>& cancelled) {
    if (weight.cols != static_cast<std::int64_t>(input.size())) {
        throw std::runtime_error("native Glimmer linear input width does not match its weight");
    }
    output->assign(checked_size(weight.rows, "linear output"), 0.0f);
    parallel_for(weight.rows, 128, [&](std::int64_t row) {
        if ((row & 31) == 0) {
            throw_if_cancelled(cancelled);
        }
        double sum = 0.0;
        for (std::int64_t col = 0; col < weight.cols; ++col) {
            sum += static_cast<double>(input[static_cast<std::size_t>(col)]) *
                weight.value(row, col);
        }
        const float result = static_cast<float>(sum);
        if (!std::isfinite(result)) {
            throw std::runtime_error("native Glimmer linear produced a non-finite value");
        }
        (*output)[static_cast<std::size_t>(row)] = result;
    });
}

void linear_with_lora(
    const std::vector<float>& input,
    const GlimmerModel::WeightView& weight,
    const std::optional<GlimmerModel::LoraWeight>& adapter,
    std::vector<float>* output,
    const std::atomic<bool>& cancelled) {
    linear(input, weight, output, cancelled);
    if (!adapter.has_value()) return;
    const auto& lora = *adapter;
    if (lora.a.cols != weight.cols || lora.b.rows != weight.rows ||
        lora.b.cols != lora.a.rows || !std::isfinite(lora.scaling) ||
        !(lora.scaling > 0.0)) {
        throw std::runtime_error("native Glimmer LoRA projection geometry is invalid");
    }
    std::vector<float> rank;
    linear(input, lora.a, &rank, cancelled);
    std::vector<float> delta;
    linear(rank, lora.b, &delta, cancelled);
    if (delta.size() != output->size()) {
        throw std::runtime_error("native Glimmer LoRA projection output is invalid");
    }
    parallel_for(static_cast<std::int64_t>(output->size()), 256, [&](std::int64_t index) {
        const double value = static_cast<double>((*output)[static_cast<std::size_t>(index)]) +
            lora.scaling * static_cast<double>(delta[static_cast<std::size_t>(index)]);
        const float result = static_cast<float>(value);
        if (!std::isfinite(result)) {
            throw std::runtime_error("native Glimmer LoRA projection produced a non-finite value");
        }
        (*output)[static_cast<std::size_t>(index)] = result;
    });
}

void rms_norm(
    const std::vector<float>& input,
    const GlimmerModel::WeightView* weight,
    double epsilon,
    std::vector<float>* output,
    const std::atomic<bool>& cancelled) {
    throw_if_cancelled(cancelled);
    if (input.empty() || (weight != nullptr && (weight->rows != 1 ||
            weight->cols != static_cast<std::int64_t>(input.size())))) {
        throw std::runtime_error("native Glimmer RMSNorm has invalid geometry");
    }
    double squares = 0.0;
    for (float value : input) {
        squares += static_cast<double>(value) * value;
    }
    const double inverse = 1.0 / std::sqrt(squares / static_cast<double>(input.size()) + epsilon);
    output->resize(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        double scale = 1.0;
        if (weight != nullptr) {
            scale = weight->value(0, static_cast<std::int64_t>(index));
            if (weight->centered) {
                scale += 1.0;
            }
        }
        const float result = static_cast<float>(static_cast<double>(input[index]) * inverse * scale);
        if (!std::isfinite(result)) {
            throw std::runtime_error("native Glimmer RMSNorm produced a non-finite value");
        }
        (*output)[index] = result;
    }
}

void qk_norm_in_place(
    std::vector<float>* values,
    std::int64_t heads,
    std::int64_t head_dim,
    const GlimmerModel::WeightView* weight,
    double epsilon,
    double scalar,
    const std::atomic<bool>& cancelled) {
    if (static_cast<std::int64_t>(values->size()) != heads * head_dim ||
        (weight != nullptr && (weight->rows != 1 || weight->cols != head_dim))) {
        throw std::runtime_error("native Glimmer Q/K norm has invalid geometry");
    }
    parallel_for(heads, 4, [&](std::int64_t head) {
        throw_if_cancelled(cancelled);
        float* row = values->data() + head * head_dim;
        double squares = 0.0;
        for (std::int64_t dim = 0; dim < head_dim; ++dim) {
            squares += static_cast<double>(row[dim]) * row[dim];
        }
        const double inverse = 1.0 / std::sqrt(squares / static_cast<double>(head_dim) + epsilon);
        for (std::int64_t dim = 0; dim < head_dim; ++dim) {
            const double learned = weight == nullptr ? 1.0 : weight->value(0, dim);
            row[dim] = static_cast<float>(static_cast<double>(row[dim]) * inverse * learned * scalar);
        }
    });
}

void apply_rope(
    std::vector<float>* values,
    std::int64_t heads,
    std::int64_t head_dim,
    std::int64_t position,
    double theta,
    bool interleaved,
    const std::atomic<bool>& cancelled) {
    parallel_for(heads, 4, [&](std::int64_t head) {
        throw_if_cancelled(cancelled);
        float* row = values->data() + head * head_dim;
        const std::int64_t half = head_dim / 2;
        for (std::int64_t index = 0; index < half; ++index) {
            const double inverse_frequency = 1.0 / std::pow(
                theta, static_cast<double>(2 * index) / static_cast<double>(head_dim));
            const double angle = static_cast<double>(position) * inverse_frequency;
            const double cosine = std::cos(angle);
            const double sine = std::sin(angle);
            const std::int64_t first_index = interleaved ? 2 * index : index;
            const std::int64_t second_index = interleaved ? 2 * index + 1 : index + half;
            const float first = row[first_index];
            const float second = row[second_index];
            // Meta/Transformers rotate_half: [x1*cos-x2*sin, x2*cos+x1*sin].
            row[first_index] = static_cast<float>(first * cosine - second * sine);
            row[second_index] = static_cast<float>(second * cosine + first * sine);
        }
    });
}

GlimmerModel::WeightView::Encoding encoding_from_ggml(std::uint32_t type) {
    switch (type) {
        case 0: return GlimmerModel::WeightView::Encoding::F32;
        case 12: return GlimmerModel::WeightView::Encoding::Q4K;
        case 13: return GlimmerModel::WeightView::Encoding::Q5K;
        case 14: return GlimmerModel::WeightView::Encoding::Q6K;
        case 30: return GlimmerModel::WeightView::Encoding::BF16;
        default: throw std::runtime_error("Muse Glimmer GGUF uses an unsupported tensor encoding");
    }
}

std::pair<std::int64_t, std::int64_t> expected_gguf_shape(const std::string& name) {
    if (name == "token_embd.weight" || name == "output.weight") {
        return {202048, 6656};
    }
    if (name == "output_norm.weight") {
        return {1, 6656};
    }
    if (!name.starts_with("blk.") || !name.ends_with(".weight")) {
        throw std::runtime_error("Muse Glimmer GGUF contains an unexpected tensor name: " + name);
    }
    const std::size_t first_dot = name.find('.', 4);
    if (first_dot == std::string::npos) {
        throw std::runtime_error("Muse Glimmer GGUF layer tensor name is malformed");
    }
    const std::string layer_text = name.substr(4, first_dot - 4);
    if (layer_text.empty() || !std::all_of(layer_text.begin(), layer_text.end(), ::isdigit)) {
        throw std::runtime_error("Muse Glimmer GGUF layer tensor index is malformed");
    }
    const int layer = std::stoi(layer_text);
    if (layer < 0 || layer >= 52) {
        throw std::runtime_error("Muse Glimmer GGUF layer tensor index is out of range");
    }
    const std::string suffix = name.substr(first_dot + 1, name.size() - first_dot - 8);
    const std::unordered_map<std::string, std::pair<std::int64_t, std::int64_t>> shapes = {
        {"attn_norm", {1, 6656}},
        {"post_attention_norm", {1, 6656}},
        {"ffn_norm", {1, 6656}},
        {"post_ffw_norm", {1, 6656}},
        {"ffn_down", {6656, 19968}},
        {"ffn_gate", {19968, 6656}},
        {"ffn_up", {19968, 6656}},
        {"attn_gate", {4096, 6656}},
        {"attn_k", {256, 6656}},
        {"attn_output", {6656, 4096}},
        {"attn_q", {4096, 6656}},
        {"attn_v", {256, 6656}},
        {"attn_q_norm", {1, 128}},
        {"attn_k_norm", {1, 128}},
    };
    const auto found = shapes.find(suffix);
    if (found == shapes.end()) {
        throw std::runtime_error("Muse Glimmer GGUF contains an unexpected layer tensor: " + name);
    }
    return found->second;
}

std::unordered_set<std::string> expected_gguf_names() {
    std::unordered_set<std::string> names = {
        "token_embd.weight", "output.weight", "output_norm.weight"};
    const std::vector<std::string> suffixes = {
        "attn_norm", "ffn_down", "ffn_gate", "ffn_up", "post_attention_norm",
        "post_ffw_norm", "ffn_norm", "attn_gate", "attn_k", "attn_output",
        "attn_q_norm", "attn_k_norm", "attn_q", "attn_v"};
    for (int layer = 0; layer < 52; ++layer) {
        for (const std::string& suffix : suffixes) {
            names.insert("blk." + std::to_string(layer) + "." + suffix + ".weight");
        }
    }
    return names;
}

}  // namespace

GlimmerModel::GlimmerModel(
    std::string checkpoint_path,
    GlimmerInferenceConfig config,
    std::unique_ptr<MappedFile> mapped)
    : checkpoint_path_(std::move(checkpoint_path)),
      config_(std::move(config)),
      mapped_(std::move(mapped)) {
    if (config_.container == WeightContainer::NativeBf16) {
        build_native_bf16_layout();
    } else {
        build_gguf_layout();
    }
    if (config_.whole_model_cuda) {
        initialize_cuda_backend();
    }
}

GlimmerModel::~GlimmerModel() {
    close();
}

std::shared_ptr<GlimmerModel> GlimmerModel::load(
    const std::string& checkpoint_path,
    GlimmerInferenceConfig config) {
    validate_config(config);
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error("native Glimmer checkpoints require a little-endian host");
    }
    auto mapped = std::make_unique<MappedFile>(checkpoint_path);
    if (!config.checkpoint_sha256_preverified) {
        const std::string actual = neuralfn::resident_support::sha256_hex(
            mapped->data(), mapped->size());
        if (actual != config.checkpoint_sha256) {
            throw std::runtime_error(
                "native Glimmer loaded bytes do not match the manifest SHA-256 fingerprint");
        }
    }
    return std::shared_ptr<GlimmerModel>(new GlimmerModel(
        checkpoint_path, std::move(config), std::move(mapped)));
}

void GlimmerModel::build_native_bf16_layout() {
    const std::int64_t expected_parameters = text_parameter_count(config_);
    const std::int64_t expected_bytes = checked_mul(expected_parameters, 2, "BF16 checkpoint bytes");
    const std::int64_t actual_bytes = static_cast<std::int64_t>(mapped_->size());
    const bool embedded_production_vision = production_geometry(config_) &&
        actual_bytes == checked_add(kProductionTextBytes, kProductionVisionBytes, "full BF16 bytes");
    if (actual_bytes != expected_bytes && !embedded_production_vision) {
        throw std::runtime_error(
            "native Glimmer BF16 checkpoint length does not match its canonical text/full layout");
    }
    std::int64_t offset = 0;
    auto take = [&](std::int64_t rows, std::int64_t cols, bool centered = false) {
        WeightView view;
        view.data = mapped_->data() + checked_size(offset, "BF16 tensor offset");
        view.rows = rows;
        view.cols = cols;
        view.row_stride_bytes = checked_mul(cols, 2, "BF16 row stride");
        view.nbytes = checked_mul(rows, view.row_stride_bytes, "BF16 tensor bytes");
        view.encoding = WeightView::Encoding::BF16;
        view.centered = centered;
        offset = checked_add(offset, view.nbytes, "BF16 tensor layout");
        if (offset > expected_bytes) {
            throw std::runtime_error("native Glimmer BF16 tensor layout exceeds its text payload");
        }
        return view;
    };
    token_embedding_ = std::make_unique<WeightView>(take(config_.vocab_size, config_.model_dim));
    const std::int64_t query_width = config_.num_heads * config_.head_dim;
    const std::int64_t key_value_width = config_.num_kv_heads * config_.head_dim;
    layers_.reserve(static_cast<std::size_t>(config_.num_layers));
    for (std::int64_t index = 0; index < config_.num_layers; ++index) {
        LayerLayout layer;
        layer.input_norm = take(1, config_.model_dim, true);
        layer.post_attention_norm = take(1, config_.model_dim, true);
        layer.pre_feedforward_norm = take(1, config_.model_dim, true);
        layer.post_feedforward_norm = take(1, config_.model_dim, true);
        layer.q = take(query_width, config_.model_dim);
        layer.k = take(key_value_width, config_.model_dim);
        layer.v = take(key_value_width, config_.model_dim);
        layer.gate = take(query_width, config_.model_dim);
        layer.output = take(config_.model_dim, query_width);
        layer.mlp_gate = take(config_.intermediate_dim, config_.model_dim);
        layer.mlp_up = take(config_.intermediate_dim, config_.model_dim);
        layer.mlp_down = take(config_.model_dim, config_.intermediate_dim);
        layers_.push_back(std::move(layer));
    }
    final_norm_ = std::make_unique<WeightView>(take(1, config_.model_dim));
    lm_head_ = std::make_unique<WeightView>(take(config_.vocab_size, config_.model_dim));
    if (offset != expected_bytes) {
        throw std::runtime_error("native Glimmer BF16 tensor layout has the wrong final extent");
    }
    if (embedded_production_vision) {
        vision_model_ = neuralfn::resident_glimmer_vision::Model::load_bf16(
            mapped_->data() + checked_size(expected_bytes, "embedded vision offset"),
            kProductionVisionBytes);
    }
    parameter_count_ = expected_parameters;
}

void GlimmerModel::build_gguf_layout() {
    Cursor cursor(mapped_->data(), mapped_->size());
    cursor.require(4, "magic");
    if (std::memcmp(cursor.pointer(), "GGUF", 4) != 0) {
        throw std::runtime_error("Muse Glimmer GGUF magic is invalid");
    }
    cursor.skip(4, "magic");
    if (cursor.read<std::uint32_t>("version") != 3) {
        throw std::runtime_error("native Glimmer requires GGUF version 3");
    }
    const std::uint64_t tensor_count = cursor.read<std::uint64_t>("tensor count");
    const std::uint64_t metadata_count = cursor.read<std::uint64_t>("metadata count");
    if (tensor_count != 731 || metadata_count != 32) {
        throw std::runtime_error("Muse Glimmer GGUF must contain 731 tensors and 32 metadata entries");
    }
    const std::unordered_set<std::string> expected_metadata = {
        "general.architecture", "general.type", "general.name", "general.size_label",
        "muse-glimmer.block_count", "muse-glimmer.context_length",
        "muse-glimmer.embedding_length", "muse-glimmer.feed_forward_length",
        "muse-glimmer.attention.head_count", "muse-glimmer.attention.head_count_kv",
        "muse-glimmer.rope.freq_base", "muse-glimmer.attention.layer_norm_rms_epsilon",
        "muse-glimmer.attention.key_length", "muse-glimmer.attention.value_length",
        "muse-glimmer.final_logit_softcapping", "muse-glimmer.logit_scale",
        "muse-glimmer.attention.sliding_window",
        "muse-glimmer.attention.sliding_window_pattern", "general.quantization_version",
        "tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.tokens",
        "tokenizer.ggml.token_type", "tokenizer.ggml.merges", "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id",
        "tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_sep_token",
        "tokenizer.ggml.eot_token_id", "general.file_type", "tokenizer.chat_template",
    };
    std::unordered_set<std::string> seen_metadata;
    for (std::uint64_t entry = 0; entry < metadata_count; ++entry) {
        const std::string key = cursor.string("metadata key");
        if (!seen_metadata.insert(key).second || !expected_metadata.contains(key)) {
            throw std::runtime_error("Muse Glimmer GGUF metadata allowlist mismatch at " + key);
        }
        const std::uint32_t type = cursor.read<std::uint32_t>("metadata type");
        if (key == "general.architecture") {
            if (type != GgufString || cursor.string("architecture") != "muse-glimmer") {
                throw std::runtime_error("Muse Glimmer GGUF has the wrong architecture");
            }
        } else if (key == "muse-glimmer.block_count" ||
                   key == "muse-glimmer.context_length" ||
                   key == "muse-glimmer.embedding_length" ||
                   key == "muse-glimmer.feed_forward_length" ||
                   key == "muse-glimmer.attention.head_count" ||
                   key == "muse-glimmer.attention.head_count_kv" ||
                   key == "muse-glimmer.attention.key_length" ||
                   key == "muse-glimmer.attention.value_length" ||
                   key == "muse-glimmer.attention.sliding_window" ||
                   key == "general.quantization_version" || key == "general.file_type" ||
                   key == "tokenizer.ggml.bos_token_id" ||
                   key == "tokenizer.ggml.eos_token_id" ||
                   key == "tokenizer.ggml.padding_token_id" ||
                   key == "tokenizer.ggml.eot_token_id") {
            const std::uint64_t value = read_integer_metadata(&cursor, type, key.c_str());
            const std::unordered_map<std::string, std::uint64_t> expected = {
                {"muse-glimmer.block_count", 52}, {"muse-glimmer.context_length", 131072},
                {"muse-glimmer.embedding_length", 6656},
                {"muse-glimmer.feed_forward_length", 19968},
                {"muse-glimmer.attention.head_count", 32},
                {"muse-glimmer.attention.head_count_kv", 2},
                {"muse-glimmer.attention.key_length", 128},
                {"muse-glimmer.attention.value_length", 128},
                {"muse-glimmer.attention.sliding_window", 2048},
                {"general.quantization_version", 2}, {"general.file_type", 15},
                {"tokenizer.ggml.bos_token_id", 200000},
                {"tokenizer.ggml.eos_token_id", 200001},
                {"tokenizer.ggml.padding_token_id", 200018},
                {"tokenizer.ggml.eot_token_id", 200008},
            };
            if (value != expected.at(key)) {
                throw std::runtime_error("Muse Glimmer GGUF scalar metadata mismatch at " + key);
            }
        } else if (key == "muse-glimmer.rope.freq_base" ||
                   key == "muse-glimmer.attention.layer_norm_rms_epsilon" ||
                   key == "muse-glimmer.final_logit_softcapping" ||
                   key == "muse-glimmer.logit_scale") {
            const double value = read_float_metadata(&cursor, type, key.c_str());
            const std::unordered_map<std::string, double> expected = {
                {"muse-glimmer.rope.freq_base", 500000.0},
                {"muse-glimmer.attention.layer_norm_rms_epsilon", 1.0e-5},
                {"muse-glimmer.final_logit_softcapping", 20.0},
                {"muse-glimmer.logit_scale", 0.19611613513818404},
            };
            const double target = expected.at(key);
            if (std::abs(value - target) > std::max(1.0e-9, std::abs(target) * 1.0e-6)) {
                throw std::runtime_error("Muse Glimmer GGUF float metadata mismatch at " + key);
            }
        } else if (key == "muse-glimmer.attention.sliding_window_pattern") {
            if (type != GgufArray || cursor.read<std::uint32_t>("pattern type") != GgufBool ||
                cursor.read<std::uint64_t>("pattern length") != 52) {
                throw std::runtime_error("Muse Glimmer GGUF local/global schedule is malformed");
            }
            for (int layer = 0; layer < 52; ++layer) {
                const bool local = cursor.read<std::uint8_t>("pattern value") != 0;
                if (local != (layer % 4 != 3)) {
                    throw std::runtime_error("Muse Glimmer GGUF local/global schedule is incorrect");
                }
            }
        } else {
            skip_gguf_value(&cursor, type);
        }
    }
    if (seen_metadata != expected_metadata) {
        throw std::runtime_error("Muse Glimmer GGUF metadata allowlist is incomplete");
    }

    struct RawTensor {
        std::string name;
        std::vector<std::uint64_t> dimensions;
        std::uint32_t type = 0;
        std::uint64_t offset = 0;
    };
    std::vector<RawTensor> raw;
    raw.reserve(731);
    std::unordered_set<std::string> names;
    for (std::uint64_t index = 0; index < tensor_count; ++index) {
        RawTensor tensor;
        tensor.name = cursor.string("tensor name");
        if (!names.insert(tensor.name).second) {
            throw std::runtime_error("Muse Glimmer GGUF contains a duplicate tensor name");
        }
        const std::uint32_t rank = cursor.read<std::uint32_t>("tensor rank");
        if (rank < 1 || rank > 2) {
            throw std::runtime_error("Muse Glimmer GGUF tensor rank is unsupported");
        }
        for (std::uint32_t dimension = 0; dimension < rank; ++dimension) {
            const std::uint64_t value = cursor.read<std::uint64_t>("tensor dimension");
            if (value == 0 || value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
                throw std::runtime_error("Muse Glimmer GGUF tensor has an invalid dimension");
            }
            tensor.dimensions.push_back(value);
        }
        tensor.type = cursor.read<std::uint32_t>("tensor encoding");
        tensor.offset = cursor.read<std::uint64_t>("tensor offset");
        raw.push_back(std::move(tensor));
    }
    if (names != expected_gguf_names()) {
        throw std::runtime_error("Muse Glimmer GGUF tensor allowlist mismatch");
    }
    const std::size_t data_offset = align_up(cursor.offset(), 32);
    cursor.require(data_offset - cursor.offset(), "header padding");
    for (std::size_t index = cursor.offset(); index < data_offset; ++index) {
        if (mapped_->data()[index] != 0) {
            throw std::runtime_error("Muse Glimmer GGUF header padding is nonzero");
        }
    }

    std::unordered_map<std::string, WeightView> tensors;
    struct Extent { std::uint64_t offset; std::uint64_t nbytes; std::string name; };
    std::vector<Extent> extents;
    std::unordered_map<std::uint32_t, std::int64_t> inventory;
    for (const RawTensor& tensor : raw) {
        const auto [rows, cols] = expected_gguf_shape(tensor.name);
        if (tensor.dimensions[0] != static_cast<std::uint64_t>(cols) ||
            (tensor.dimensions.size() == 1 ? rows != 1
                                          : tensor.dimensions[1] != static_cast<std::uint64_t>(rows))) {
            throw std::runtime_error("Muse Glimmer GGUF tensor shape mismatch at " + tensor.name);
        }
        const WeightView::Encoding encoding = encoding_from_ggml(tensor.type);
        std::int64_t block_elements = 1;
        std::int64_t block_bytes = tensor.type == 0 ? 4 : 2;
        if (tensor.type == 12) { block_elements = 256; block_bytes = 144; }
        if (tensor.type == 13) { block_elements = 256; block_bytes = 176; }
        if (tensor.type == 14) { block_elements = 256; block_bytes = 210; }
        if (cols % block_elements != 0 || tensor.offset % 32 != 0) {
            throw std::runtime_error("Muse Glimmer GGUF tensor block/alignment contract failed");
        }
        const std::int64_t row_stride = checked_mul(cols / block_elements, block_bytes, "GGUF row stride");
        const std::int64_t nbytes = checked_mul(rows, row_stride, "GGUF tensor bytes");
        if (tensor.offset > mapped_->size() - std::min(mapped_->size(), data_offset) ||
            static_cast<std::uint64_t>(nbytes) >
                mapped_->size() - data_offset - static_cast<std::size_t>(tensor.offset)) {
            throw std::runtime_error("Muse Glimmer GGUF tensor exceeds the file extent");
        }
        WeightView view;
        view.data = mapped_->data() + data_offset + static_cast<std::size_t>(tensor.offset);
        view.rows = rows;
        view.cols = cols;
        view.row_stride_bytes = row_stride;
        view.nbytes = nbytes;
        view.encoding = encoding;
        view.centered = false;  // The official converter folds (1 + delta).
        tensors.emplace(tensor.name, view);
        extents.push_back({tensor.offset, static_cast<std::uint64_t>(nbytes), tensor.name});
        ++inventory[tensor.type];
    }
    std::sort(extents.begin(), extents.end(), [](const Extent& left, const Extent& right) {
        return left.offset < right.offset;
    });
    std::uint64_t expected_offset = 0;
    for (const Extent& extent : extents) {
        if (extent.offset != expected_offset) {
            throw std::runtime_error("Muse Glimmer GGUF tensor data is not canonical/contiguous");
        }
        expected_offset = static_cast<std::uint64_t>(align_up(
            checked_size(checked_add(
                static_cast<std::int64_t>(extent.offset),
                static_cast<std::int64_t>(extent.nbytes), "GGUF extent"), "GGUF extent"), 32));
    }
    const Extent& last = extents.back();
    if (data_offset + last.offset + last.nbytes != mapped_->size()) {
        throw std::runtime_error("Muse Glimmer GGUF has trailing or missing tensor bytes");
    }
    const bool profile17 = inventory[0] == 313 && inventory[12] == 365 &&
        inventory[13] == 1 && inventory[14] == 52 && inventory[30] == 0;
    const bool profileDynamic = inventory[0] == 313 && inventory[12] == 51 &&
        inventory[13] == 130 && inventory[14] == 237 && inventory[30] == 0;
    if (!profile17 && !profileDynamic) {
        throw std::runtime_error("Muse Glimmer GGUF tensor encoding inventory is not a canonical K-Quant profile");
    }

    token_embedding_ = std::make_unique<WeightView>(tensors.at("token_embd.weight"));
    final_norm_ = std::make_unique<WeightView>(tensors.at("output_norm.weight"));
    lm_head_ = std::make_unique<WeightView>(tensors.at("output.weight"));
    layers_.reserve(52);
    for (int index = 0; index < 52; ++index) {
        const std::string prefix = "blk." + std::to_string(index) + ".";
        LayerLayout layer;
        layer.input_norm = tensors.at(prefix + "attn_norm.weight");
        layer.post_attention_norm = tensors.at(prefix + "post_attention_norm.weight");
        layer.pre_feedforward_norm = tensors.at(prefix + "ffn_norm.weight");
        layer.post_feedforward_norm = tensors.at(prefix + "post_ffw_norm.weight");
        layer.q = tensors.at(prefix + "attn_q.weight");
        layer.k = tensors.at(prefix + "attn_k.weight");
        layer.v = tensors.at(prefix + "attn_v.weight");
        layer.gate = tensors.at(prefix + "attn_gate.weight");
        layer.output = tensors.at(prefix + "attn_output.weight");
        layer.mlp_gate = tensors.at(prefix + "ffn_gate.weight");
        layer.mlp_up = tensors.at(prefix + "ffn_up.weight");
        layer.mlp_down = tensors.at(prefix + "ffn_down.weight");
        layer.q_norm = tensors.at(prefix + "attn_q_norm.weight");
        layer.k_norm = tensors.at(prefix + "attn_k_norm.weight");
        layers_.push_back(std::move(layer));
    }
    parameter_count_ = text_parameter_count(config_);
}

void GlimmerModel::initialize_cuda_backend() {
    using neuralfn::resident_glimmer_cuda::Config;
    using neuralfn::resident_glimmer_cuda::HostLayerWeights;
    using neuralfn::resident_glimmer_cuda::HostWeightPlan;
    using neuralfn::resident_glimmer_cuda::HostWeightView;
    auto convert = [](const WeightView& source) {
        HostWeightView target;
        target.data = source.data;
        target.rows = source.rows;
        target.cols = source.cols;
        target.row_stride_bytes = source.row_stride_bytes;
        target.nbytes = source.nbytes;
        target.centered = source.centered;
        switch (source.encoding) {
            case WeightView::Encoding::F32: target.encoding = 0; break;
            case WeightView::Encoding::Q4K: target.encoding = 12; break;
            case WeightView::Encoding::Q5K: target.encoding = 13; break;
            case WeightView::Encoding::Q6K: target.encoding = 14; break;
            case WeightView::Encoding::BF16: target.encoding = 30; break;
        }
        return target;
    };
    HostWeightPlan plan;
    plan.token_embedding = convert(*token_embedding_);
    plan.final_norm = convert(*final_norm_);
    plan.lm_head = convert(*lm_head_);
    plan.layers.reserve(layers_.size());
    for (const LayerLayout& source : layers_) {
        HostLayerWeights layer;
        layer.input_norm = convert(source.input_norm);
        layer.post_attention_norm = convert(source.post_attention_norm);
        layer.pre_feedforward_norm = convert(source.pre_feedforward_norm);
        layer.post_feedforward_norm = convert(source.post_feedforward_norm);
        layer.q = convert(source.q);
        layer.k = convert(source.k);
        layer.v = convert(source.v);
        layer.gate = convert(source.gate);
        layer.output = convert(source.output);
        layer.mlp_gate = convert(source.mlp_gate);
        layer.mlp_up = convert(source.mlp_up);
        layer.mlp_down = convert(source.mlp_down);
        if (source.q_norm) layer.q_norm = convert(*source.q_norm);
        if (source.k_norm) layer.k_norm = convert(*source.k_norm);
        plan.layers.push_back(std::move(layer));
    }
    Config cuda;
    cuda.max_seq_len = config_.max_seq_len;
    cuda.vocab_size = config_.vocab_size;
    cuda.num_layers = config_.num_layers;
    cuda.model_dim = config_.model_dim;
    cuda.intermediate_dim = config_.intermediate_dim;
    cuda.num_heads = config_.num_heads;
    cuda.num_kv_heads = config_.num_kv_heads;
    cuda.head_dim = config_.head_dim;
    cuda.sliding_window = config_.sliding_window;
    cuda.rope_theta = static_cast<float>(config_.rope_theta);
    cuda.norm_eps = static_cast<float>(config_.norm_eps);
    cuda.post_norm_eps = static_cast<float>(config_.post_norm_eps);
    cuda.q_scale_factor = static_cast<float>(config_.q_scale_factor);
    cuda.output_multiplier = static_cast<float>(config_.output_multiplier);
    cuda.logit_softcap = static_cast<float>(config_.logit_softcap);
    cuda.gguf_interleaved = config_.container == WeightContainer::GgufKQuant;
    cuda.cuda_device = config_.cuda_device;
    cuda.tile_ops_lib = config_.tile_ops_lib;
    cuda.cuda_runtime_lib = config_.cuda_runtime_lib;
    cuda_model_ = neuralfn::resident_glimmer_cuda::Model::load(cuda, plan);
    if (vision_model_) {
        cuda_model_->load_vision(
            vision_model_->cuda_config(), vision_model_->cuda_weight_plan());
    }
}

void GlimmerModel::require_open() const {
    if (closed()) {
        throw std::runtime_error("resident inference model is closed");
    }
}

std::shared_ptr<GlimmerSession> GlimmerModel::create_session(
    std::int64_t seed,
    KVCacheMode cache_mode) {
    std::shared_lock<std::shared_mutex> lock(lifecycle_mutex_);
    require_open();
    if (cache_mode == KVCacheMode::TurboQuant) {
        throw std::runtime_error("Muse Glimmer uses its exact hybrid lossless cache; TurboQuant is unsupported");
    }
    return std::make_shared<GlimmerSession>(shared_from_this(), seed, cache_mode);
}

std::shared_ptr<GlimmerSession> GlimmerModel::create_speculative_session(
    std::int64_t seed,
    KVCacheMode cache_mode,
    std::shared_ptr<neuralfn::resident_glimmer_assistant::Model> assistant) {
    std::shared_lock<std::shared_mutex> lock(lifecycle_mutex_);
    require_open();
    if (!assistant || assistant->target().get() != this) {
        throw std::runtime_error("DFlash assistant is not bound to this target model");
    }
    if (adapter_mapped_) {
        throw std::runtime_error(
            "stock Muse Glimmer DFlash is disabled for an adapted target until an "
            "adapter-bound assistant passes compatibility validation");
    }
    if (cache_mode != KVCacheMode::Full) {
        throw std::runtime_error("DFlash speculative decoding requires the lossless full cache mode");
    }
    return std::make_shared<GlimmerSession>(
        shared_from_this(), seed, cache_mode, std::move(assistant));
}

void GlimmerModel::close() noexcept {
    std::unique_lock<std::shared_mutex> lock(lifecycle_mutex_);
    if (cuda_model_) {
        cuda_model_->close();
    }
    closed_.store(true);
}

ModelStats GlimmerModel::stats() const {
    ModelStats result;
    result.checkpoint_path = checkpoint_path_;
    result.max_seq_len = config_.max_seq_len;
    result.vocab_size = config_.vocab_size;
    result.padded_vocab_size = config_.vocab_size;
    result.num_layers = config_.num_layers;
    result.num_heads = config_.num_heads;
    result.channels = config_.model_dim;
    result.parameter_count = parameter_count_ + lora_parameter_count_;
    result.weight_bytes = static_cast<std::int64_t>(mapped_->size()) +
        (adapter_mapped_ ? static_cast<std::int64_t>(adapter_mapped_->size()) : 0);
    result.weights_load_count = adapter_mapped_ ? 2 : 1;
    result.open_sessions = open_sessions_.load();
    result.forward_calls = forward_calls_.load();
    result.use_qk_norm = true;
    result.qk_norm_eps = config_.norm_eps;
    result.logit_softcap = config_.logit_softcap;
    result.mlp_activation = "swiglu";
    return result;
}

std::int64_t GlimmerModel::cuda_resident_weight_bytes() const noexcept {
    return cuda_model_ ? cuda_model_->resident_weight_bytes() : 0;
}

std::int64_t GlimmerModel::cuda_workspace_bytes() const noexcept {
    return cuda_model_ ? cuda_model_->workspace_bytes() : 0;
}

std::int64_t GlimmerModel::cuda_kernel_launches() const noexcept {
    return cuda_model_ ? cuda_model_->kernel_launches() : 0;
}

std::int64_t GlimmerModel::cuda_k_quant_mmq_linears() const noexcept {
    return cuda_model_ ? cuda_model_->k_quant_mmq_linears() : 0;
}

std::int64_t GlimmerModel::cuda_q8_activation_quantizations() const noexcept {
    return cuda_model_ ? cuda_model_->q8_activation_quantizations() : 0;
}

std::int64_t GlimmerModel::cuda_q8_packed_linears() const noexcept {
    return cuda_model_ ? cuda_model_->q8_packed_linears() : 0;
}

std::int64_t GlimmerModel::cuda_device_argmax_calls() const noexcept {
    return cuda_model_ ? cuda_model_->device_argmax_calls() : 0;
}

std::int64_t GlimmerModel::cuda_device_argmax_rows() const noexcept {
    return cuda_model_ ? cuda_model_->device_argmax_rows() : 0;
}

int GlimmerModel::cuda_device() const noexcept {
    return cuda_model_ ? cuda_model_->cuda_device() : -1;
}

std::string GlimmerModel::cuda_tile_ops_library() const {
    return cuda_model_ ? cuda_model_->tile_ops_library() : std::string{};
}

std::string GlimmerModel::cuda_runtime_library() const {
    return cuda_model_ ? cuda_model_->cuda_runtime_library() : std::string{};
}

bool GlimmerModel::vision_whole_model_cuda() const noexcept {
    return cuda_model_ && cuda_model_->has_vision();
}

void GlimmerModel::load_vision_companion(const std::string& checkpoint_path) {
    std::unique_lock<std::shared_mutex> lock(lifecycle_mutex_);
    require_open();
    if (vision_model_) {
        throw std::runtime_error("Muse Glimmer vision weights are already loaded");
    }
    if (open_sessions_.load() != 0) {
        throw std::runtime_error(
            "Muse Glimmer mmproj must be loaded before creating target sessions");
    }
    auto loaded = neuralfn::resident_glimmer_vision::Model::load_gguf(
        checkpoint_path);
    if (cuda_model_) {
        cuda_model_->load_vision(
            loaded->cuda_config(), loaded->cuda_weight_plan());
    }
    vision_model_ = std::move(loaded);
}

void GlimmerModel::load_lora_adapter(
    const std::string& checkpoint_path,
    const std::string& checkpoint_sha256,
    std::int64_t rank,
    double alpha,
    std::uint32_t target_mask) {
    std::unique_lock<std::shared_mutex> lock(lifecycle_mutex_);
    require_open();
    if (adapter_mapped_) {
        throw std::runtime_error("Muse Glimmer LoRA adapter is already loaded");
    }
    if (open_sessions_.load() != 0) {
        throw std::runtime_error(
            "Muse Glimmer LoRA adapter must be loaded before creating target sessions");
    }
    if (!valid_sha256(checkpoint_sha256) || rank <= 0 || rank > config_.model_dim ||
        !std::isfinite(alpha) || !(alpha > 0.0) || target_mask == 0 ||
        (target_mask & ~0xffU) != 0) {
        throw std::runtime_error("Muse Glimmer LoRA adapter metadata is invalid");
    }
    auto mapped = std::make_unique<MappedFile>(checkpoint_path);
    if (neuralfn::resident_support::sha256_hex(mapped->data(), mapped->size()) !=
        checkpoint_sha256) {
        throw std::runtime_error("Muse Glimmer LoRA adapter SHA-256 mismatch");
    }
    std::int64_t offset = 0;
    std::int64_t parameter_count = 0;
    const double scaling = alpha / static_cast<double>(rank);
    auto take = [&](std::int64_t rows, std::int64_t cols) {
        LoraWeight result;
        result.a.data = mapped->data() + checked_size(offset, "LoRA A offset");
        result.a.rows = rank;
        result.a.cols = cols;
        result.a.row_stride_bytes = checked_mul(cols, 2, "LoRA A row stride");
        result.a.nbytes = checked_mul(rank, result.a.row_stride_bytes, "LoRA A bytes");
        result.a.encoding = WeightView::Encoding::BF16;
        offset = checked_add(offset, result.a.nbytes, "LoRA A extent");
        if (offset > static_cast<std::int64_t>(mapped->size())) {
            throw std::runtime_error("Muse Glimmer LoRA A tensor exceeds the file extent");
        }
        result.b.data = mapped->data() + checked_size(offset, "LoRA B offset");
        result.b.rows = rows;
        result.b.cols = rank;
        result.b.row_stride_bytes = checked_mul(rank, 2, "LoRA B row stride");
        result.b.nbytes = checked_mul(rows, result.b.row_stride_bytes, "LoRA B bytes");
        result.b.encoding = WeightView::Encoding::BF16;
        offset = checked_add(offset, result.b.nbytes, "LoRA B extent");
        parameter_count = checked_add(
            parameter_count,
            checked_add(checked_mul(rank, cols, "LoRA A parameters"),
                        checked_mul(rows, rank, "LoRA B parameters"),
                        "LoRA site parameters"),
            "LoRA parameters");
        if (offset > static_cast<std::int64_t>(mapped->size())) {
            throw std::runtime_error("Muse Glimmer LoRA adapter tensor exceeds the file extent");
        }
        result.scaling = scaling;
        return result;
    };
    const std::int64_t query_width = checked_mul(
        config_.num_heads, config_.head_dim, "LoRA query width");
    const std::int64_t kv_width = checked_mul(
        config_.num_kv_heads, config_.head_dim, "LoRA KV width");
    std::vector<LoraLayerLayout> layouts;
    layouts.resize(static_cast<std::size_t>(config_.num_layers));
    for (std::int64_t layer = 0; layer < config_.num_layers; ++layer) {
        LoraLayerLayout& layout = layouts[static_cast<std::size_t>(layer)];
        // The byte order follows the canonical base checkpoint table, not the
        // public target-mask bit order: q,k,v,attention-gate,o,gate,up,down.
        if (target_mask & (1U << 0U)) layout.q = take(query_width, config_.model_dim);
        if (target_mask & (1U << 1U)) layout.k = take(kv_width, config_.model_dim);
        if (target_mask & (1U << 2U)) layout.v = take(kv_width, config_.model_dim);
        if (target_mask & (1U << 4U)) layout.gate = take(query_width, config_.model_dim);
        if (target_mask & (1U << 3U)) layout.output = take(config_.model_dim, query_width);
        if (target_mask & (1U << 5U)) layout.mlp_gate = take(config_.intermediate_dim, config_.model_dim);
        if (target_mask & (1U << 6U)) layout.mlp_up = take(config_.intermediate_dim, config_.model_dim);
        if (target_mask & (1U << 7U)) layout.mlp_down = take(config_.model_dim, config_.intermediate_dim);
    }
    if (offset != static_cast<std::int64_t>(mapped->size())) {
        throw std::runtime_error("Muse Glimmer LoRA adapter has trailing or missing tensor bytes");
    }
    if (cuda_model_) {
        using neuralfn::resident_glimmer_cuda::HostLoraLayer;
        using neuralfn::resident_glimmer_cuda::HostLoraPlan;
        using neuralfn::resident_glimmer_cuda::HostLoraWeight;
        using neuralfn::resident_glimmer_cuda::HostWeightView;
        auto convert_weight = [](const WeightView& source) {
            if (source.encoding != WeightView::Encoding::BF16) {
                throw std::runtime_error("Muse Glimmer native LoRA tensors must be BF16");
            }
            HostWeightView target;
            target.data = source.data;
            target.rows = source.rows;
            target.cols = source.cols;
            target.row_stride_bytes = source.row_stride_bytes;
            target.nbytes = source.nbytes;
            target.encoding = 30;
            return target;
        };
        auto convert_lora = [&](const LoraWeight& source) {
            HostLoraWeight target;
            target.a = convert_weight(source.a);
            target.b = convert_weight(source.b);
            target.scaling = static_cast<float>(source.scaling);
            return target;
        };
        HostLoraPlan plan;
        plan.layers.reserve(layouts.size());
        for (const LoraLayerLayout& source : layouts) {
            HostLoraLayer layer;
            if (source.q) layer.q = convert_lora(*source.q);
            if (source.k) layer.k = convert_lora(*source.k);
            if (source.v) layer.v = convert_lora(*source.v);
            if (source.gate) layer.gate = convert_lora(*source.gate);
            if (source.output) layer.output = convert_lora(*source.output);
            if (source.mlp_gate) layer.mlp_gate = convert_lora(*source.mlp_gate);
            if (source.mlp_up) layer.mlp_up = convert_lora(*source.mlp_up);
            if (source.mlp_down) layer.mlp_down = convert_lora(*source.mlp_down);
            plan.layers.push_back(std::move(layer));
        }
        cuda_model_->load_lora_adapter(plan);
    }
    lora_layers_ = std::move(layouts);
    lora_parameter_count_ = parameter_count;
    adapter_mapped_ = std::move(mapped);
}

std::int64_t GlimmerModel::vision_output_size() const noexcept {
    return vision_model_ ? vision_model_->output_size() : 0;
}

std::int64_t GlimmerModel::vision_weight_bytes() const noexcept {
    return vision_model_ ? vision_model_->weight_bytes() : 0;
}

std::vector<float> GlimmerModel::encode_media(
    const std::vector<float>& packed_patches,
    const std::vector<std::int64_t>& grid_thw,
    const std::atomic<bool>& cancelled) const {
    std::shared_lock<std::shared_mutex> lock(lifecycle_mutex_);
    require_open();
    if (!vision_model_) {
        throw std::runtime_error(
            "Muse Glimmer media encoding requires a full BF16 checkpoint or a compatible mmproj companion");
    }
    if (cuda_model_) {
        if (!cuda_model_->has_vision()) {
            throw std::runtime_error(
                "whole-model Glimmer CUDA has no resident vision weights");
        }
        return cuda_model_->encode_vision(
            packed_patches, grid_thw, cancelled);
    }
    return vision_model_->encode(packed_patches, grid_thw, cancelled);
}

std::unique_ptr<GlimmerCacheStorage> GlimmerModel::create_cache() const {
    auto cache = std::make_unique<GlimmerCacheStorage>();
    if (cuda_model_) {
        cache->cuda = cuda_model_->create_cache();
        return cache;
    }
    cache->layers.resize(static_cast<std::size_t>(config_.num_layers));
    const std::int64_t local_values = checked_mul(config_.sliding_window, kv_dim(), "local cache values");
    for (std::int64_t layer = 0; layer < config_.num_layers; ++layer) {
        GlimmerLayerCache& storage = cache->layers[static_cast<std::size_t>(layer)];
        storage.local = is_local_layer(layer);
        if (storage.local) {
            storage.keys.assign(checked_size(local_values, "local key cache"), 0.0f);
            storage.values.assign(checked_size(local_values, "local value cache"), 0.0f);
        }
    }
    cache->final_hidden.assign(static_cast<std::size_t>(config_.model_dim), 0.0f);
    return cache;
}

void GlimmerModel::forward_append_token(
    std::int64_t token,
    std::int64_t position,
    GlimmerCacheStorage* cache,
    const std::atomic<bool>& cancelled,
    const std::vector<std::int64_t>* tap_layers,
    std::vector<float>* target_taps,
    bool fast_k_quant) const {
    forward_append_input(
        token, nullptr, position, cache, cancelled, tap_layers, target_taps,
        fast_k_quant);
}

void GlimmerModel::forward_append_embedding(
    const std::vector<float>& embedding,
    std::int64_t position,
    GlimmerCacheStorage* cache,
    const std::atomic<bool>& cancelled,
    const std::vector<std::int64_t>* tap_layers,
    std::vector<float>* target_taps) const {
    forward_append_input(
        -1, &embedding, position, cache, cancelled, tap_layers, target_taps,
        false);
}

void GlimmerModel::forward_append_input(
    std::int64_t token,
    const std::vector<float>* embedding,
    std::int64_t position,
    GlimmerCacheStorage* cache,
    const std::atomic<bool>& cancelled,
    const std::vector<std::int64_t>* tap_layers,
    std::vector<float>* target_taps,
    bool fast_k_quant) const {
    require_open();
    if (embedding == nullptr && (token < 0 || token >= config_.vocab_size)) {
        throw std::runtime_error("resident Glimmer token id is outside the checkpoint vocabulary");
    }
    if (embedding != nullptr && embedding->size() != static_cast<std::size_t>(config_.model_dim)) {
        throw std::runtime_error("resident Glimmer replacement embedding has the wrong width");
    }
    if (position < 0 || position >= config_.max_seq_len) {
        throw std::runtime_error("resident Glimmer position exceeds the checkpoint context window");
    }
    if (cuda_model_) {
        if (cache == nullptr || !cache->cuda) {
            throw std::runtime_error("resident Glimmer CUDA cache is unavailable");
        }
        forward_calls_.fetch_add(1);
        if (embedding == nullptr) {
            cuda_model_->append_token(
                token, position, cache->cuda, cancelled, tap_layers, target_taps,
                fast_k_quant);
        } else {
            cuda_model_->append_embedding(
                *embedding, position, cache->cuda, cancelled, tap_layers, target_taps,
                false);
        }
        return;
    }
    if ((tap_layers == nullptr) != (target_taps == nullptr)) {
        throw std::runtime_error("resident Glimmer tap layers/output must be supplied together");
    }
    if (tap_layers != nullptr) {
        if (tap_layers->empty() || !std::is_sorted(tap_layers->begin(), tap_layers->end()) ||
            std::adjacent_find(tap_layers->begin(), tap_layers->end()) != tap_layers->end() ||
            tap_layers->front() < 0 || tap_layers->back() >= config_.num_layers) {
            throw std::runtime_error("resident Glimmer target tap layer list is invalid");
        }
        target_taps->clear();
        target_taps->reserve(checked_size(
            checked_mul(static_cast<std::int64_t>(tap_layers->size()), config_.model_dim,
                        "target tap extent"),
            "target tap extent"));
    }
    if (cache == nullptr || cache->layers.size() != static_cast<std::size_t>(config_.num_layers)) {
        throw std::runtime_error("resident Glimmer hybrid cache has invalid layer storage");
    }
    for (std::int64_t layer = 0; layer < config_.num_layers; ++layer) {
        GlimmerLayerCache& storage = cache->layers[static_cast<std::size_t>(layer)];
        if (storage.local != is_local_layer(layer) || storage.logical_length != position) {
            throw std::runtime_error("resident Glimmer hybrid cache logical state is inconsistent");
        }
        if (!storage.local) {
            const std::int64_t required = checked_mul(position + 1, kv_dim(), "global cache extent");
            storage.keys.resize(checked_size(required, "global key cache"));
            storage.values.resize(checked_size(required, "global value cache"));
        }
    }

    throw_if_cancelled(cancelled);
    forward_calls_.fetch_add(1);
    std::vector<float> hidden(static_cast<std::size_t>(config_.model_dim));
    if (embedding != nullptr) {
        hidden = *embedding;
    } else {
        for (std::int64_t dim = 0; dim < config_.model_dim; ++dim) {
            hidden[static_cast<std::size_t>(dim)] = token_embedding_->value(token, dim);
        }
    }
    std::vector<float> normalized;
    rms_norm(hidden, nullptr, config_.norm_eps, &normalized, cancelled);
    hidden.swap(normalized);

    std::vector<std::vector<float>> staged_keys(static_cast<std::size_t>(config_.num_layers));
    std::vector<std::vector<float>> staged_values(static_cast<std::size_t>(config_.num_layers));
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    std::vector<float> gate;
    std::vector<float> attention(static_cast<std::size_t>(config_.num_heads * config_.head_dim));
    std::vector<float> projected;
    std::vector<float> mlp_gate;
    std::vector<float> mlp_up;
    std::vector<float> activated;
    std::vector<float> down;
    const bool gguf = config_.container == WeightContainer::GgufKQuant;
    const double attention_scale = 1.0 / std::sqrt(static_cast<double>(config_.head_dim));
    const std::optional<LoraWeight> no_adapter;

    for (std::int64_t layer_index = 0; layer_index < config_.num_layers; ++layer_index) {
        throw_if_cancelled(cancelled);
        const LayerLayout& layout = layers_[static_cast<std::size_t>(layer_index)];
        const LoraLayerLayout* adapter = lora_layers_.empty()
            ? nullptr : &lora_layers_[static_cast<std::size_t>(layer_index)];
        const std::vector<float> attention_residual = hidden;
        rms_norm(hidden, &layout.input_norm, config_.norm_eps, &normalized, cancelled);
        linear_with_lora(normalized, layout.q, adapter ? adapter->q : no_adapter, &query, cancelled);
        linear_with_lora(normalized, layout.k, adapter ? adapter->k : no_adapter, &key, cancelled);
        linear_with_lora(normalized, layout.v, adapter ? adapter->v : no_adapter, &value, cancelled);
        linear_with_lora(normalized, layout.gate, adapter ? adapter->gate : no_adapter, &gate, cancelled);
        qk_norm_in_place(
            &query, config_.num_heads, config_.head_dim,
            layout.q_norm ? &*layout.q_norm : nullptr,
            config_.norm_eps, gguf ? 1.0 : config_.q_scale_factor, cancelled);
        qk_norm_in_place(
            &key, config_.num_kv_heads, config_.head_dim,
            layout.k_norm ? &*layout.k_norm : nullptr,
            config_.norm_eps, 1.0, cancelled);
        if (is_local_layer(layer_index)) {
            apply_rope(
                &query, config_.num_heads, config_.head_dim, position,
                config_.rope_theta, gguf, cancelled);
            apply_rope(
                &key, config_.num_kv_heads, config_.head_dim, position,
                config_.rope_theta, gguf, cancelled);
        }
        staged_keys[static_cast<std::size_t>(layer_index)] = key;
        staged_values[static_cast<std::size_t>(layer_index)] = value;
        std::fill(attention.begin(), attention.end(), 0.0f);
        const GlimmerLayerCache& storage = cache->layers[static_cast<std::size_t>(layer_index)];
        const std::int64_t first_position = storage.local
            ? std::max<std::int64_t>(0, position - config_.sliding_window + 1)
            : 0;
        parallel_for(config_.num_heads, 4, [&](std::int64_t query_head) {
            throw_if_cancelled(cancelled);
            const std::int64_t kv_head = query_head * config_.num_kv_heads / config_.num_heads;
            const float* query_row = query.data() + query_head * config_.head_dim;
            const std::int64_t count = position - first_position + 1;
            std::vector<double> scores(static_cast<std::size_t>(count));
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::int64_t key_position = first_position; key_position <= position; ++key_position) {
                const float* key_row = nullptr;
                if (key_position == position) {
                    key_row = key.data() + kv_head * config_.head_dim;
                } else {
                    const std::int64_t slot = storage.local
                        ? key_position % config_.sliding_window : key_position;
                    key_row = storage.keys.data() + slot * kv_dim() + kv_head * config_.head_dim;
                }
                double score = 0.0;
                for (std::int64_t dim = 0; dim < config_.head_dim; ++dim) {
                    score += static_cast<double>(query_row[dim]) * key_row[dim];
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
                throw std::runtime_error("resident Glimmer attention probabilities are invalid");
            }
            for (std::int64_t dim = 0; dim < config_.head_dim; ++dim) {
                double accumulated = 0.0;
                for (std::int64_t key_position = first_position; key_position <= position; ++key_position) {
                    const float* value_row = nullptr;
                    if (key_position == position) {
                        value_row = value.data() + kv_head * config_.head_dim;
                    } else {
                        const std::int64_t slot = storage.local
                            ? key_position % config_.sliding_window : key_position;
                        value_row = storage.values.data() + slot * kv_dim() + kv_head * config_.head_dim;
                    }
                    accumulated += scores[static_cast<std::size_t>(key_position - first_position)] /
                        denominator * value_row[dim];
                }
                attention[static_cast<std::size_t>(query_head * config_.head_dim + dim)] =
                    static_cast<float>(accumulated);
            }
        });
        if (gate.size() != attention.size()) {
            throw std::runtime_error("resident Glimmer attention gate width is inconsistent");
        }
        for (std::size_t index = 0; index < attention.size(); ++index) {
            const double gate_value = gate[index];
            const double sigmoid = gate_value >= 0.0
                ? 1.0 / (1.0 + std::exp(-gate_value))
                : std::exp(gate_value) / (1.0 + std::exp(gate_value));
            attention[index] = static_cast<float>(attention[index] * sigmoid);
        }
        linear_with_lora(
            attention, layout.output, adapter ? adapter->output : no_adapter,
            &projected, cancelled);
        rms_norm(projected, &layout.post_attention_norm, config_.post_norm_eps, &normalized, cancelled);
        hidden.resize(static_cast<std::size_t>(config_.model_dim));
        for (std::int64_t dim = 0; dim < config_.model_dim; ++dim) {
            hidden[static_cast<std::size_t>(dim)] = attention_residual[static_cast<std::size_t>(dim)] +
                normalized[static_cast<std::size_t>(dim)];
        }

        const std::vector<float> mlp_residual = hidden;
        rms_norm(hidden, &layout.pre_feedforward_norm, config_.norm_eps, &normalized, cancelled);
        linear_with_lora(
            normalized, layout.mlp_gate, adapter ? adapter->mlp_gate : no_adapter,
            &mlp_gate, cancelled);
        linear_with_lora(
            normalized, layout.mlp_up, adapter ? adapter->mlp_up : no_adapter,
            &mlp_up, cancelled);
        activated.resize(static_cast<std::size_t>(config_.intermediate_dim));
        for (std::int64_t dim = 0; dim < config_.intermediate_dim; ++dim) {
            activated[static_cast<std::size_t>(dim)] = silu(mlp_gate[static_cast<std::size_t>(dim)]) *
                mlp_up[static_cast<std::size_t>(dim)];
        }
        linear_with_lora(
            activated, layout.mlp_down, adapter ? adapter->mlp_down : no_adapter,
            &down, cancelled);
        rms_norm(down, &layout.post_feedforward_norm, config_.post_norm_eps, &normalized, cancelled);
        for (std::int64_t dim = 0; dim < config_.model_dim; ++dim) {
            hidden[static_cast<std::size_t>(dim)] = mlp_residual[static_cast<std::size_t>(dim)] +
                normalized[static_cast<std::size_t>(dim)];
        }
        if (tap_layers != nullptr && std::binary_search(
                tap_layers->begin(), tap_layers->end(), layer_index)) {
            target_taps->insert(target_taps->end(), hidden.begin(), hidden.end());
        }
    }

    rms_norm(hidden, final_norm_.get(), config_.norm_eps, &normalized, cancelled);
    throw_if_cancelled(cancelled);
    for (std::int64_t layer = 0; layer < config_.num_layers; ++layer) {
        GlimmerLayerCache& storage = cache->layers[static_cast<std::size_t>(layer)];
        const std::int64_t slot = storage.local ? position % config_.sliding_window : position;
        const std::size_t offset = checked_size(checked_mul(slot, kv_dim(), "cache row offset"), "cache row offset");
        std::copy(
            staged_keys[static_cast<std::size_t>(layer)].begin(),
            staged_keys[static_cast<std::size_t>(layer)].end(),
            storage.keys.begin() + static_cast<std::ptrdiff_t>(offset));
        std::copy(
            staged_values[static_cast<std::size_t>(layer)].begin(),
            staged_values[static_cast<std::size_t>(layer)].end(),
            storage.values.begin() + static_cast<std::ptrdiff_t>(offset));
        storage.logical_length = position + 1;
    }
    cache->final_hidden = std::move(normalized);
    throw_if_cancelled(cancelled);
}

std::vector<float> GlimmerModel::forward_last_logits(
    const std::vector<std::int64_t>& tokens,
    const std::atomic<bool>& cancelled) const {
    require_open();
    if (tokens.empty()) {
        throw std::runtime_error("resident Glimmer decode requires at least one prompt token");
    }
    if (tokens.size() > static_cast<std::size_t>(config_.max_seq_len)) {
        throw std::runtime_error("resident Glimmer history exceeds the checkpoint context window");
    }
    auto cache = create_cache();
    for (std::int64_t position = 0; position < static_cast<std::int64_t>(tokens.size()); ++position) {
        forward_append_token(tokens[static_cast<std::size_t>(position)], position, cache.get(), cancelled);
    }
    return logits_from_cache(*cache);
}

std::vector<float> GlimmerModel::logits_from_cache(const GlimmerCacheStorage& cache) const {
    require_open();
    if (cuda_model_) {
        if (!cache.cuda) {
            throw std::runtime_error("resident Glimmer CUDA cache is unavailable for logits");
        }
        return cuda_model_->logits(cache.cuda);
    }
    return logits_from_hidden(cache.final_hidden.data());
}

std::vector<float> GlimmerModel::logits_from_hidden(const float* hidden) const {
    std::vector<float> logits = raw_logits_from_hidden(hidden);
    for (float& value : logits) {
        value = static_cast<float>(config_.logit_softcap * std::tanh(
            (config_.output_multiplier * static_cast<double>(value)) / config_.logit_softcap));
    }
    return logits;
}

std::vector<float> GlimmerModel::raw_logits_from_hidden(const float* hidden) const {
    require_open();
    if (hidden == nullptr) {
        throw std::runtime_error("resident Glimmer final hidden state is null");
    }
    if (cuda_model_) {
        return cuda_model_->raw_logits(hidden);
    }
    std::vector<float> input(hidden, hidden + config_.model_dim);
    std::vector<float> logits;
    static const std::atomic<bool> never_cancelled{false};
    linear(input, *lm_head_, &logits, never_cancelled);
    return logits;
}

std::vector<float> GlimmerModel::raw_logits_rows_from_hidden(
    const float* hidden,
    std::int64_t rows,
    bool fast_k_quant) const {
    require_open();
    if (hidden == nullptr || rows <= 0 || rows > 16) {
        throw std::runtime_error(
            "resident Glimmer raw batched logits require 1..16 hidden rows");
    }
    if (cuda_model_) {
        return cuda_model_->raw_logits_rows(hidden, rows, fast_k_quant);
    }
    std::vector<float> result;
    result.reserve(checked_size(checked_mul(
        rows, config_.vocab_size, "raw batched logits"), "raw batched logits"));
    for (std::int64_t row = 0; row < rows; ++row) {
        std::vector<float> logits = raw_logits_from_hidden(
            hidden + row * config_.model_dim);
        result.insert(result.end(), logits.begin(), logits.end());
    }
    return result;
}

std::vector<std::int64_t> GlimmerModel::raw_argmax_rows_from_hidden(
    const float* hidden,
    std::int64_t rows,
    bool fast_k_quant) const {
    require_open();
    if (hidden == nullptr || rows <= 0 || rows > 16) {
        throw std::runtime_error(
            "resident Glimmer raw batched argmax requires 1..16 hidden rows");
    }
    if (cuda_model_ && cuda_model_->has_device_argmax()) {
        return cuda_model_->raw_argmax_rows(hidden, rows, fast_k_quant).indices;
    }
    const std::vector<float> logits = raw_logits_rows_from_hidden(
        hidden, rows, fast_k_quant);
    std::vector<std::int64_t> result;
    result.reserve(checked_size(rows, "raw argmax rows"));
    for (std::int64_t row = 0; row < rows; ++row) {
        const float* first = logits.data() + row * config_.vocab_size;
        result.push_back(static_cast<std::int64_t>(
            std::distance(first, std::max_element(first, first + config_.vocab_size))));
    }
    return result;
}

std::vector<std::int64_t> GlimmerModel::raw_argmax_rows_from_device_hidden(
    const float* device_hidden,
    int source_cuda_device,
    std::int64_t rows,
    bool fast_k_quant) const {
    require_open();
    if (!cuda_model_ || device_hidden == nullptr || rows <= 0 || rows > 16 ||
        !cuda_model_->has_device_argmax()) {
        throw std::runtime_error(
            "resident Glimmer device-hidden argmax requires CUDA and 1..16 rows");
    }
    return cuda_model_->raw_argmax_rows_device(
        device_hidden, source_cuda_device, rows, fast_k_quant).indices;
}

bool GlimmerModel::has_cuda_device_argmax() const noexcept {
    return cuda_model_ && cuda_model_->has_device_argmax();
}

std::vector<float> GlimmerModel::raw_token_embedding(std::int64_t token) const {
    require_open();
    if (token < 0 || token >= config_.vocab_size) {
        throw std::runtime_error("resident Glimmer token id is outside the checkpoint vocabulary");
    }
    if (cuda_model_) {
        return cuda_model_->raw_embedding(token);
    }
    std::vector<float> result(static_cast<std::size_t>(config_.model_dim));
    for (std::int64_t dim = 0; dim < config_.model_dim; ++dim) {
        result[static_cast<std::size_t>(dim)] = token_embedding_->value(token, dim);
    }
    return result;
}

std::vector<float> GlimmerModel::raw_token_embeddings(
    const std::vector<std::int64_t>& tokens) const {
    require_open();
    if (tokens.empty() || tokens.size() > 16 ||
        std::any_of(tokens.begin(), tokens.end(), [&](std::int64_t token) {
            return token < 0 || token >= config_.vocab_size;
        })) {
        throw std::runtime_error(
            "resident Glimmer raw batched embeddings require 1..16 valid tokens");
    }
    if (cuda_model_) {
        return cuda_model_->raw_embeddings(tokens);
    }
    std::vector<float> result;
    result.reserve(checked_size(checked_mul(
        static_cast<std::int64_t>(tokens.size()), config_.model_dim,
        "raw batched embeddings"), "raw batched embeddings"));
    for (std::int64_t token : tokens) {
        std::vector<float> row = raw_token_embedding(token);
        result.insert(result.end(), row.begin(), row.end());
    }
    return result;
}

const float* GlimmerModel::raw_token_embeddings_device(
    const std::vector<std::int64_t>& tokens) const {
    require_open();
    if (!cuda_model_) {
        throw std::runtime_error(
            "resident Glimmer device embeddings require a CUDA resident");
    }
    return cuda_model_->raw_embeddings_device(tokens);
}

DecodeResult GlimmerModel::select_token(
    const std::vector<float>& logits,
    const GenerationConfig& config,
    std::mt19937_64& rng) const {
    if (!std::isfinite(config.temperature) || config.temperature < 0.0 || config.top_k < 0 ||
        !std::isfinite(config.top_p) || !(config.top_p > 0.0) || config.top_p > 1.0) {
        throw std::runtime_error("resident Glimmer sampling configuration is invalid");
    }
    if (logits.size() != static_cast<std::size_t>(config_.vocab_size)) {
        throw std::runtime_error("resident Glimmer logits have the wrong vocabulary width");
    }
    std::int64_t selected = static_cast<std::int64_t>(std::distance(
        logits.begin(), std::max_element(logits.begin(), logits.end())));
    if (config.temperature > 0.0 && config.top_k != 1) {
        std::vector<std::int64_t> candidates(static_cast<std::size_t>(config_.vocab_size));
        std::iota(candidates.begin(), candidates.end(), 0);
        std::sort(candidates.begin(), candidates.end(), [&](std::int64_t left, std::int64_t right) {
            const float lhs = logits[static_cast<std::size_t>(left)];
            const float rhs = logits[static_cast<std::size_t>(right)];
            return lhs == rhs ? left < right : lhs > rhs;
        });
        if (config.top_k > 0 && config.top_k < static_cast<std::int64_t>(candidates.size())) {
            candidates.resize(static_cast<std::size_t>(config.top_k));
        }
        const double maximum = logits[static_cast<std::size_t>(candidates.front())];
        std::vector<double> weights;
        weights.reserve(candidates.size());
        double total = 0.0;
        for (std::int64_t token : candidates) {
            const double shifted =
                (static_cast<double>(logits[static_cast<std::size_t>(token)]) - maximum) /
                config.temperature;
            const double weight = std::exp(std::max(-745.0, shifted));
            weights.push_back(weight);
            total += weight;
        }
        if (!(total > 0.0) || !std::isfinite(total)) {
            throw std::runtime_error("resident Glimmer sampling probabilities are invalid");
        }
        double cumulative = 0.0;
        std::size_t retained = weights.size();
        for (std::size_t index = 0; index < weights.size(); ++index) {
            cumulative += weights[index] / total;
            if (cumulative >= config.top_p) {
                retained = index + 1;
                break;
            }
        }
        candidates.resize(retained);
        weights.resize(retained);
        std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
        selected = candidates[distribution(rng)];
    }
    DecodeResult result;
    result.token_id = selected;
    result.selected_logit = logits[static_cast<std::size_t>(selected)];
    if (contains_token(config.stop_token_ids, selected)) {
        result.finish_reason = "stop";
    }
    return result;
}

GlimmerSession::GlimmerSession(
    std::shared_ptr<GlimmerModel> model,
    std::int64_t seed,
    KVCacheMode cache_mode,
    std::shared_ptr<neuralfn::resident_glimmer_assistant::Model> assistant)
    : model_(std::move(model)),
      cache_mode_(cache_mode),
      assistant_model_(std::move(assistant)),
      seed_(seed),
      rng_(static_cast<std::mt19937_64::result_type>(seed)) {
    if (cache_mode_ == KVCacheMode::TurboQuant) {
        throw std::runtime_error("Muse Glimmer does not support TurboQuant cache mode");
    }
    tokens_.reserve(static_cast<std::size_t>(model_->max_seq_len()));
    if (cache_mode_ == KVCacheMode::Full) {
        cache_ = model_->create_cache();
    }
    if (assistant_model_) {
        if (cache_mode_ != KVCacheMode::Full || assistant_model_->target().get() != model_.get()) {
            throw std::runtime_error("DFlash session requires its bound target and full cache");
        }
        assistant_session_ = assistant_model_->create_session();
    }
    model_->session_opened();
}

GlimmerSession::~GlimmerSession() {
    close();
}

void GlimmerSession::require_open() const {
    if (closed_) {
        throw std::runtime_error("resident inference session is closed");
    }
    if (model_->closed()) {
        throw std::runtime_error("resident inference model is closed");
    }
}

void GlimmerSession::rebuild_cache() {
    if (cache_mode_ != KVCacheMode::Full) {
        return;
    }
    auto rebuilt = model_->create_cache();
    auto rebuilt_assistant = assistant_model_ ? assistant_model_->create_session() : nullptr;
    std::atomic<bool> restore_cancelled{false};
    for (std::int64_t position = 0; position < static_cast<std::int64_t>(tokens_.size()); ++position) {
        std::vector<float> taps;
        const auto replacement = embedding_overrides_.find(position);
        if (replacement == embedding_overrides_.end()) {
            model_->forward_append_token(
                tokens_[static_cast<std::size_t>(position)], position, rebuilt.get(), restore_cancelled,
                rebuilt_assistant ? &assistant_model_->target_layer_ids() : nullptr,
                rebuilt_assistant ? &taps : nullptr);
        } else {
            model_->forward_append_embedding(
                replacement->second, position, rebuilt.get(), restore_cancelled,
                rebuilt_assistant ? &assistant_model_->target_layer_ids() : nullptr,
                rebuilt_assistant ? &taps : nullptr);
        }
        if (rebuilt_assistant) {
            rebuilt_assistant->record_target_taps(position, taps, restore_cancelled);
        }
    }
    cache_ = std::move(rebuilt);
    assistant_session_ = std::move(rebuilt_assistant);
    cache_length_ = static_cast<std::int64_t>(tokens_.size());
    speculative_pending_token_ = false;
}

void GlimmerSession::append_cached_token(
    std::int64_t token,
    std::int64_t position,
    bool fast_k_quant) {
    std::vector<float> taps;
    model_->forward_append_token(
        token, position, cache_.get(), cancelled_,
        assistant_session_ ? &assistant_model_->target_layer_ids() : nullptr,
        assistant_session_ ? &taps : nullptr,
        fast_k_quant);
    if (assistant_session_) {
        assistant_session_->record_target_taps(position, taps, cancelled_);
    }
}

void GlimmerSession::append_cached_embedding(
    std::int64_t token,
    std::int64_t position,
    const std::vector<float>& embedding) {
    if (token != 200091 && token != 200092) {
        throw std::runtime_error("replacement embeddings require a Muse Glimmer image/video token");
    }
    std::vector<float> taps;
    model_->forward_append_embedding(
        embedding, position, cache_.get(), cancelled_,
        assistant_session_ ? &assistant_model_->target_layer_ids() : nullptr,
        assistant_session_ ? &taps : nullptr);
    if (assistant_session_) {
        assistant_session_->record_target_taps(position, taps, cancelled_);
    }
}

void GlimmerSession::materialize_speculative_pending_token(bool fast_k_quant) {
    if (!speculative_pending_token_) return;
    if (cache_mode_ != KVCacheMode::Full || !cache_ || !assistant_session_ ||
        tokens_.empty() || cache_length_ + 1 != static_cast<std::int64_t>(tokens_.size())) {
        throw std::runtime_error("Glimmer lagged speculative cache state is inconsistent");
    }
    append_cached_token(tokens_.back(), cache_length_, fast_k_quant);
    ++cache_length_;
    speculative_pending_token_ = false;
}

std::int64_t GlimmerSession::cache_bytes() const {
    if (!cache_) {
        return 0;
    }
    if (cache_->cuda) {
        const std::int64_t length = cache_->cuda->logical_length();
        const std::int64_t global_layers = model_->num_layers() / 4;
        const std::int64_t local_layers = model_->num_layers() - global_layers;
        const std::int64_t retained_rows =
            local_layers * std::min(length, model_->sliding_window()) +
            global_layers * length;
        const std::int64_t target_bytes = checked_add(
            checked_mul(model_->model_dim(), 4, "CUDA final hidden bytes"),
            checked_mul(
                2,
                checked_mul(retained_rows,
                    checked_mul(model_->kv_dim(), 2, "CUDA BF16 KV row bytes"),
                    "CUDA hybrid KV rows"),
                "CUDA K/V bytes"),
            "CUDA cache bytes");
        return checked_add(
            target_bytes,
            assistant_session_ ? assistant_session_->cache_bytes() : 0,
            "target and DFlash cache bytes");
    }
    std::int64_t bytes = checked_mul(model_->model_dim(), sizeof(float), "final hidden bytes");
    for (const GlimmerLayerCache& layer : cache_->layers) {
        const std::int64_t rows = layer.local
            ? std::min(layer.logical_length, model_->sliding_window())
            : layer.logical_length;
        bytes = checked_add(
            bytes,
            checked_mul(2, checked_mul(rows, checked_mul(model_->kv_dim(), sizeof(float), "KV row bytes"),
                "KV layer bytes"), "K/V bytes"),
            "hybrid cache bytes");
    }
    return checked_add(
        bytes,
        assistant_session_ ? assistant_session_->cache_bytes() : 0,
        "target and DFlash cache bytes");
}

void GlimmerSession::prefill(
    const std::vector<std::int64_t>& token_ids,
    std::int64_t start_position) {
    prefill_with_embeddings(token_ids, start_position, {}, {});
}

void GlimmerSession::prefill_with_embeddings(
    const std::vector<std::int64_t>& token_ids,
    std::int64_t start_position,
    const std::vector<std::int64_t>& replacement_positions,
    const std::vector<float>& replacement_embeddings) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::shared_lock<std::shared_mutex> model_lock(model_->lifecycle_mutex_);
    require_open();
    throw_if_cancelled(cancelled_);
    if (start_position != static_cast<std::int64_t>(tokens_.size())) {
        throw std::runtime_error("prefill start_position does not match native session history");
    }
    if (checked_add(start_position, static_cast<std::int64_t>(token_ids.size()), "prefill history") >
        model_->max_seq_len()) {
        throw std::runtime_error("prefill would exceed the checkpoint context window");
    }
    for (std::int64_t token : token_ids) {
        if (token < 0 || token >= model_->vocab_size()) {
            throw std::runtime_error("prefill token id is outside the checkpoint vocabulary");
        }
    }
    if (replacement_embeddings.size() !=
        replacement_positions.size() * static_cast<std::size_t>(model_->model_dim())) {
        throw std::runtime_error("prefill replacement embedding extent is invalid");
    }
    if (!replacement_positions.empty() && cache_mode_ != KVCacheMode::Full) {
        throw std::runtime_error("multimodal prefill requires the lossless hybrid cache");
    }
    std::unordered_map<std::int64_t, std::vector<float>> replacements;
    for (std::size_t index = 0; index < replacement_positions.size(); ++index) {
        const std::int64_t absolute = replacement_positions[index];
        if (absolute < start_position ||
            absolute >= start_position + static_cast<std::int64_t>(token_ids.size()) ||
            !replacements.emplace(absolute, std::vector<float>{}).second) {
            throw std::runtime_error("prefill replacement position is duplicate or outside the appended suffix");
        }
        const std::int64_t token = token_ids[static_cast<std::size_t>(absolute - start_position)];
        if (token != 200091 && token != 200092) {
            throw std::runtime_error("prefill replacement position does not name an image/video token");
        }
        const float* begin = replacement_embeddings.data() + index * model_->model_dim();
        replacements[absolute].assign(begin, begin + model_->model_dim());
        if (!std::all_of(replacements[absolute].begin(), replacements[absolute].end(), [](float value) {
                return std::isfinite(value);
            })) {
            throw std::runtime_error("prefill replacement embedding contains NaN or infinity");
        }
    }
    const std::int64_t initial_length = static_cast<std::int64_t>(tokens_.size());
    try {
        materialize_speculative_pending_token();
        const bool batched_cuda_text_prefill = cache_mode_ == KVCacheMode::Full &&
            cache_ && cache_->cuda && replacements.empty();
        if (batched_cuda_text_prefill) {
            std::size_t offset = 0;
            while (offset < token_ids.size()) {
                const std::int64_t rows = std::min<std::int64_t>(
                    16, static_cast<std::int64_t>(token_ids.size() - offset));
                if (rows == 1) {
                    append_cached_token(token_ids[offset], cache_length_);
                    ++cache_length_;
                    tokens_.push_back(token_ids[offset]);
                    ++offset;
                    continue;
                }
                const std::vector<std::int64_t> chunk(
                    token_ids.begin() + static_cast<std::ptrdiff_t>(offset),
                    token_ids.begin() + static_cast<std::ptrdiff_t>(offset + rows));
                const std::int64_t chunk_start = cache_length_;
                const bool device_target_taps = assistant_session_ &&
                    assistant_model_->cuda_device_tap_pack();
                auto verification = model_->cuda_model_->verify_tokens(
                    chunk, chunk_start, cache_->cuda, cancelled_,
                    assistant_session_ ? &assistant_model_->target_layer_ids() : nullptr,
                    false, false, false, !device_target_taps);
                model_->cuda_model_->commit_verification(
                    cache_->cuda, verification, rows);
                if (assistant_session_) {
                    if (device_target_taps) {
                        if (verification->device_target_taps() == nullptr) {
                            throw std::runtime_error(
                                "Glimmer CUDA batched prefill returned no device target taps");
                        }
                        assistant_session_->record_target_taps_batch_device(
                            chunk_start, verification->device_target_taps(),
                            verification->cuda_device(), verification->rows(), rows,
                            cancelled_);
                    } else {
                        const std::vector<float>& taps = verification->target_taps();
                        const std::int64_t expected = checked_mul(
                            rows, assistant_model_->target_tap_width(),
                            "batched prefill target taps");
                        if (static_cast<std::int64_t>(taps.size()) != expected) {
                            throw std::runtime_error(
                                "Glimmer CUDA batched prefill returned malformed target taps");
                        }
                        assistant_session_->record_target_taps_batch(
                            chunk_start, taps.data(), rows, cancelled_);
                    }
                }
                tokens_.insert(tokens_.end(), chunk.begin(), chunk.end());
                cache_length_ += rows;
                model_->forward_calls_.fetch_add(rows);
                offset += static_cast<std::size_t>(rows);
            }
        } else {
            for (std::int64_t token : token_ids) {
                const std::int64_t position = static_cast<std::int64_t>(tokens_.size());
                const auto replacement = replacements.find(position);
                if (cache_mode_ == KVCacheMode::Full) {
                    if (replacement == replacements.end()) {
                        append_cached_token(token, cache_length_);
                    } else {
                        append_cached_embedding(token, cache_length_, replacement->second);
                    }
                    ++cache_length_;
                }
                tokens_.push_back(token);
                if (replacement != replacements.end()) {
                    embedding_overrides_.emplace(position, replacement->second);
                }
            }
        }
    } catch (...) {
        tokens_.resize(static_cast<std::size_t>(initial_length));
        for (auto iterator = embedding_overrides_.begin(); iterator != embedding_overrides_.end();) {
            if (iterator->first >= initial_length) iterator = embedding_overrides_.erase(iterator);
            else ++iterator;
        }
        cache_length_ = initial_length;
        if (cache_mode_ == KVCacheMode::Full) {
            try {
                rebuild_cache();
            } catch (...) {
                closed_ = true;
                model_->session_closed();
            }
        }
        throw;
    }
    ++prefill_calls_;
    prefill_tokens_ += static_cast<std::int64_t>(token_ids.size());
}

std::vector<float> GlimmerSession::current_logits() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::shared_lock<std::shared_mutex> model_lock(model_->lifecycle_mutex_);
    require_open();
    throw_if_cancelled(cancelled_);
    if (tokens_.empty()) {
        throw std::runtime_error("current_logits requires a non-empty prefilled token history");
    }
    materialize_speculative_pending_token();
    if (cache_mode_ == KVCacheMode::Full) {
        if (!cache_ || cache_length_ != static_cast<std::int64_t>(tokens_.size())) {
            throw std::runtime_error("resident Glimmer cache length does not match session history");
        }
        return model_->logits_from_cache(*cache_);
    }
    return model_->forward_last_logits(tokens_, cancelled_);
}

DecodeResult GlimmerSession::decode_one(const GenerationConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::shared_lock<std::shared_mutex> model_lock(model_->lifecycle_mutex_);
    require_open();
    throw_if_cancelled(cancelled_);
    if (tokens_.empty()) {
        throw std::runtime_error("decode_one requires a non-empty prefilled token history");
    }
    if (static_cast<std::int64_t>(tokens_.size()) >= model_->max_seq_len()) {
        throw std::runtime_error("decode_one would exceed the checkpoint context window");
    }
    for (std::int64_t token : config.stop_token_ids) {
        if (token < 0 || token >= model_->vocab_size()) {
            throw std::runtime_error("stop_token_ids contains a token outside the checkpoint vocabulary");
        }
    }
    if (!std::isfinite(config.temperature) || config.temperature < 0.0 ||
        config.top_k < 0 || !std::isfinite(config.top_p) ||
        !(config.top_p > 0.0) || config.top_p > 1.0) {
        throw std::runtime_error("resident Glimmer sampling configuration is invalid");
    }
    const std::mt19937_64 rng_before = rng_;
    const std::optional<std::int64_t> generation_seed_before = active_generation_seed_;
    const bool fast_k_quant = config.temperature > 0.0;
    try {
        if (config.seed.has_value() && config.seed != active_generation_seed_) {
            rng_.seed(static_cast<std::mt19937_64::result_type>(*config.seed));
            active_generation_seed_ = config.seed;
        }
        materialize_speculative_pending_token(fast_k_quant);
        DecodeResult result;
        const bool cuda_greedy = cache_mode_ == KVCacheMode::Full &&
            model_->cuda_model_ && model_->cuda_model_->has_device_argmax() &&
            (config.temperature == 0.0 || config.top_k == 1);
        const bool cuda_graph_greedy = cuda_greedy && !assistant_session_ &&
            model_->cuda_model_->has_decode_graphs();
        if (cuda_greedy) {
            auto selected = cuda_graph_greedy
                ? model_->cuda_model_->decode_argmax_and_append(
                      cache_->cuda, cancelled_)
                : model_->cuda_model_->argmax_logits(cache_->cuda);
            if (selected.indices.size() != 1 || selected.values.size() != 1) {
                throw std::runtime_error("Glimmer CUDA current argmax is malformed");
            }
            result.token_id = selected.indices.front();
            result.selected_logit = selected.values.front();
            if (contains_token(config.stop_token_ids, result.token_id)) {
                result.finish_reason = "stop";
            }
        } else {
            std::vector<float> logits = cache_mode_ == KVCacheMode::Full
                ? model_->logits_from_cache(*cache_)
                : model_->forward_last_logits(tokens_, cancelled_);
            result = model_->select_token(logits, config, rng_);
        }
        // A captured graph commits one token atomically. Cancellation is
        // observed before its launch and again at the next token boundary;
        // throwing here would leave the already-advanced device cache ahead
        // of the session transcript.
        if (!cuda_graph_greedy) throw_if_cancelled(cancelled_);
        if (cache_mode_ == KVCacheMode::Full) {
            if (!cuda_graph_greedy) {
                append_cached_token(result.token_id, cache_length_, fast_k_quant);
            }
            ++cache_length_;
            ++decode_rows_processed_;
        } else {
            decode_rows_processed_ += static_cast<std::int64_t>(tokens_.size());
        }
        tokens_.push_back(result.token_id);
        if (assistant_session_) speculative_ready_ = true;
        strict_model_compute_ = config.temperature == 0.0;
        ++decode_calls_;
        return result;
    } catch (...) {
        rng_ = rng_before;
        active_generation_seed_ = generation_seed_before;
        throw;
    }
}

SpeculativeStepResult GlimmerSession::decode_speculative_block(
    const GenerationConfig& config,
    std::int64_t max_tokens_remaining) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::shared_lock<std::shared_mutex> model_lock(model_->lifecycle_mutex_);
    require_open();
    throw_if_cancelled(cancelled_);
    if (!assistant_session_ || !assistant_model_ || cache_mode_ != KVCacheMode::Full) {
        throw std::runtime_error("DFlash speculative decoding requires a loaded assistant and full cache");
    }
    if (tokens_.empty()) {
        throw std::runtime_error("DFlash speculative decoding requires a non-empty prompt");
    }
    const std::int64_t context_remaining =
        model_->max_seq_len() - static_cast<std::int64_t>(tokens_.size());
    if (max_tokens_remaining <= 0 || max_tokens_remaining > context_remaining) {
        throw std::runtime_error("DFlash max_tokens_remaining exceeds the current request/context boundary");
    }
    for (std::int64_t token : config.stop_token_ids) {
        if (token < 0 || token >= model_->vocab_size()) {
            throw std::runtime_error("stop_token_ids contains a token outside the checkpoint vocabulary");
        }
    }
    const bool greedy = config.temperature == 0.0 || config.top_k == 1;
    const bool fast_k_quant = config.temperature > 0.0;
    if (!greedy && (!std::isfinite(config.temperature) || !(config.temperature > 0.0) ||
                    config.top_k < 0 || !std::isfinite(config.top_p) ||
                    !(config.top_p > 0.0) || config.top_p > 1.0)) {
        throw std::runtime_error("DFlash lossless sampling configuration is invalid");
    }

    const std::mt19937_64 rng_before = rng_;
    const std::optional<std::int64_t> generation_seed_before = active_generation_seed_;
    const std::int64_t original_length = static_cast<std::int64_t>(tokens_.size());
    const bool original_speculative_ready = speculative_ready_;
    try {
        if (config.seed.has_value() && config.seed != active_generation_seed_) {
            rng_.seed(static_cast<std::mt19937_64::result_type>(*config.seed));
            active_generation_seed_ = config.seed;
        }
        SpeculativeStepResult result;
        // Match the pinned candidate generator: its first assisted iteration
        // has no target hidden outputs yet, so the target emits one anchor.
        if (!speculative_ready_) {
            const std::vector<float> logits = model_->logits_from_cache(*cache_);
            DecodeResult decoded = model_->select_token(logits, config, rng_);
            tokens_.push_back(decoded.token_id);
            if (model_->cuda_model_) {
                assistant_session_->prepare_lagged_anchor(cache_length_, cancelled_);
                speculative_pending_token_ = true;
            } else {
                append_cached_token(decoded.token_id, cache_length_, fast_k_quant);
                ++cache_length_;
            }
            result.tokens.push_back(std::move(decoded));
            result.target_rows = model_->cuda_model_ ? 0 : 1;
            result.target_only_warmup = true;
            speculative_ready_ = true;
            if (!model_->cuda_model_) ++decode_rows_processed_;
            ++decode_calls_;
            strict_model_compute_ = config.temperature == 0.0;
            return result;
        }

        const bool lagged_cuda = model_->cuda_model_ && speculative_pending_token_;
        if (lagged_cuda &&
            cache_length_ + 1 != static_cast<std::int64_t>(tokens_.size())) {
            throw std::runtime_error("Glimmer lagged CUDA verification history is inconsistent");
        }

        // A one-token request tail does not benefit from constructing a full
        // 16-row diffusion window and verifying two rows.  Advance the single
        // lagged target row, sample directly from its logits, and leave the
        // selected token as the next lagged anchor.  This preserves the same
        // cache invariant as a normal speculative correction while avoiding
        // an otherwise mandatory assistant block at every generation limit.
        if (lagged_cuda && max_tokens_remaining == 1) {
            materialize_speculative_pending_token(fast_k_quant);
            DecodeResult decoded;
            const bool cuda_greedy = model_->cuda_model_->has_device_argmax() && greedy;
            if (cuda_greedy) {
                auto selected = model_->cuda_model_->argmax_logits(cache_->cuda);
                if (selected.indices.size() != 1 || selected.values.size() != 1) {
                    throw std::runtime_error("Glimmer CUDA tail argmax is malformed");
                }
                decoded.token_id = selected.indices.front();
                decoded.selected_logit = selected.values.front();
                if (contains_token(config.stop_token_ids, decoded.token_id)) {
                    decoded.finish_reason = "stop";
                }
            } else {
                decoded = model_->select_token(
                    model_->logits_from_cache(*cache_), config, rng_);
            }
            tokens_.push_back(decoded.token_id);
            assistant_session_->prepare_lagged_anchor(cache_length_, cancelled_);
            speculative_pending_token_ = true;
            result.tokens.push_back(std::move(decoded));
            result.target_rows = 1;
            ++decode_rows_processed_;
            ++decode_calls_;
            strict_model_compute_ = config.temperature == 0.0;
            return result;
        }

        const std::int64_t proposal_count = std::min<std::int64_t>(
            {assistant_model_->proposal_tokens(), max_tokens_remaining,
             model_->max_seq_len() - original_length});
        auto proposal = assistant_session_->propose(
            tokens_.back(), proposal_count, cancelled_, !greedy, fast_k_quant);
        if (static_cast<std::int64_t>(proposal.token_ids.size()) != proposal_count ||
            static_cast<std::int64_t>(proposal.logits.size()) !=
                (greedy ? 0 : proposal_count * model_->vocab_size())) {
            throw std::runtime_error("DFlash assistant returned a malformed proposal block");
        }
        std::vector<std::vector<double>> q_probabilities;
        if (!greedy) {
            q_probabilities.reserve(static_cast<std::size_t>(proposal_count));
            for (std::int64_t row = 0; row < proposal_count; ++row) {
                const float* first = proposal.logits.data() + row * model_->vocab_size();
                std::vector<float> row_logits(first, first + model_->vocab_size());
                q_probabilities.push_back(processed_probabilities(row_logits, config));
                proposal.token_ids[static_cast<std::size_t>(row)] =
                    sample_probabilities(q_probabilities.back(), rng_);
            }
        }

        const bool cuda_greedy_argmax = greedy && model_->cuda_model_ &&
            model_->cuda_model_->has_device_argmax();
        std::vector<std::vector<float>> target_logits;
        std::vector<std::int64_t> target_argmax_indices;
        std::vector<float> target_argmax_values;
        if (cuda_greedy_argmax) {
            target_argmax_indices.reserve(static_cast<std::size_t>(proposal_count + 1));
            target_argmax_values.reserve(static_cast<std::size_t>(proposal_count + 1));
            if (!lagged_cuda) {
                auto current = model_->cuda_model_->argmax_logits(cache_->cuda);
                if (current.indices.size() != 1 || current.values.size() != 1) {
                    throw std::runtime_error("Glimmer CUDA current argmax is malformed");
                }
                target_argmax_indices = std::move(current.indices);
                target_argmax_values = std::move(current.values);
            }
        } else {
            target_logits.reserve(static_cast<std::size_t>(proposal_count + 1));
            if (!lagged_cuda) target_logits.push_back(model_->logits_from_cache(*cache_));
        }
        std::int64_t actual_target_rows = 0;
        std::shared_ptr<neuralfn::resident_glimmer_cuda::Verification> cuda_verification;
        if (model_->cuda_model_) {
            const bool device_target_taps = assistant_model_->cuda_device_tap_pack();
            std::vector<std::int64_t> verification_tokens;
            verification_tokens.reserve(static_cast<std::size_t>(
                proposal_count + (lagged_cuda ? 1 : 0)));
            if (lagged_cuda) verification_tokens.push_back(tokens_.back());
            verification_tokens.insert(
                verification_tokens.end(), proposal.token_ids.begin(), proposal.token_ids.end());
            const std::int64_t verification_rows =
                static_cast<std::int64_t>(verification_tokens.size());
            cuda_verification = model_->cuda_model_->verify_tokens(
                verification_tokens, cache_length_, cache_->cuda, cancelled_,
                &assistant_model_->target_layer_ids(), !cuda_greedy_argmax,
                cuda_greedy_argmax, fast_k_quant,
                !device_target_taps, true);
            if (cuda_verification->rows() != verification_rows) {
                throw std::runtime_error("Glimmer CUDA verifier returned malformed rows");
            }
            if (cuda_greedy_argmax) {
                const auto& block_indices = cuda_verification->argmax_indices();
                const auto& block_values = cuda_verification->argmax_values();
                if (static_cast<std::int64_t>(block_indices.size()) != verification_rows ||
                    block_values.size() != block_indices.size()) {
                    throw std::runtime_error(
                        "Glimmer CUDA verifier returned malformed argmax rows");
                }
                target_argmax_indices.insert(
                    target_argmax_indices.end(), block_indices.begin(), block_indices.end());
                target_argmax_values.insert(
                    target_argmax_values.end(), block_values.begin(), block_values.end());
            } else {
                const std::vector<float>& block_logits = cuda_verification->logits();
                const std::int64_t vocab = model_->vocab_size();
                if (static_cast<std::int64_t>(block_logits.size()) !=
                    verification_rows * vocab) {
                    throw std::runtime_error(
                        "Glimmer CUDA verifier returned malformed logit rows");
                }
                for (std::int64_t row = 0; row < verification_rows; ++row) {
                    const float* first = block_logits.data() + row * vocab;
                    target_logits.emplace_back(first, first + vocab);
                }
            }
            actual_target_rows = verification_rows;
        } else {
            for (std::int64_t row = 0; row < proposal_count; ++row) {
                append_cached_token(
                    proposal.token_ids[static_cast<std::size_t>(row)], cache_length_,
                    fast_k_quant);
                ++cache_length_;
                ++actual_target_rows;
                tokens_.push_back(proposal.token_ids[static_cast<std::size_t>(row)]);
                target_logits.push_back(model_->logits_from_cache(*cache_));
            }
        }
        std::int64_t accepted = 0;
        bool accepted_stop = false;
        std::vector<std::vector<double>> p_probabilities;
        if (!greedy) p_probabilities.reserve(static_cast<std::size_t>(proposal_count + 1));
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        for (; accepted < proposal_count; ++accepted) {
            const std::int64_t candidate = proposal.token_ids[static_cast<std::size_t>(accepted)];
            bool keep = false;
            if (greedy) {
                const std::int64_t target_token = cuda_greedy_argmax
                    ? target_argmax_indices[static_cast<std::size_t>(accepted)]
                    : argmax_token(target_logits[static_cast<std::size_t>(accepted)]);
                keep = target_token == candidate;
            } else {
                p_probabilities.push_back(processed_probabilities(
                    target_logits[static_cast<std::size_t>(accepted)], config));
                const double q = q_probabilities[static_cast<std::size_t>(accepted)]
                    [static_cast<std::size_t>(candidate)];
                const double p = p_probabilities.back()[static_cast<std::size_t>(candidate)];
                if (!(q > 0.0) || !std::isfinite(p) || !std::isfinite(q)) {
                    throw std::runtime_error("DFlash p/q probability is invalid");
                }
                keep = uniform(rng_) <= std::min(1.0, p / q);
            }
            if (!keep) break;
            DecodeResult decoded;
            decoded.token_id = candidate;
            decoded.selected_logit = cuda_greedy_argmax
                ? target_argmax_values[static_cast<std::size_t>(accepted)]
                : target_logits[static_cast<std::size_t>(accepted)]
                    [static_cast<std::size_t>(candidate)];
            if (contains_token(config.stop_token_ids, candidate)) {
                decoded.finish_reason = "stop";
                accepted_stop = true;
            }
            result.tokens.push_back(std::move(decoded));
            if (accepted_stop) {
                ++accepted;
                break;
            }
        }
        if (!accepted_stop && static_cast<std::int64_t>(result.tokens.size()) < max_tokens_remaining) {
            DecodeResult correction;
            if (accepted < proposal_count) {
                if (greedy) {
                    if (cuda_greedy_argmax) {
                        correction.token_id =
                            target_argmax_indices[static_cast<std::size_t>(accepted)];
                        correction.selected_logit =
                            target_argmax_values[static_cast<std::size_t>(accepted)];
                    } else {
                        const std::vector<float>& p_logits =
                            target_logits[static_cast<std::size_t>(accepted)];
                        correction.token_id = argmax_token(p_logits);
                        correction.selected_logit =
                            p_logits[static_cast<std::size_t>(correction.token_id)];
                    }
                } else {
                    const std::vector<float>& p_logits =
                        target_logits[static_cast<std::size_t>(accepted)];
                    if (static_cast<std::int64_t>(p_probabilities.size()) <= accepted) {
                        p_probabilities.push_back(processed_probabilities(p_logits, config));
                    }
                    std::vector<double> residual(static_cast<std::size_t>(model_->vocab_size()));
                    double total = 0.0;
                    for (std::int64_t token = 0; token < model_->vocab_size(); ++token) {
                        const double value = std::max(
                            0.0,
                            p_probabilities[static_cast<std::size_t>(accepted)]
                                [static_cast<std::size_t>(token)] -
                            q_probabilities[static_cast<std::size_t>(accepted)]
                                [static_cast<std::size_t>(token)]);
                        residual[static_cast<std::size_t>(token)] = value;
                        total += value;
                    }
                    if (!(total > 0.0) || !std::isfinite(total)) {
                        throw std::runtime_error("DFlash rejection residual distribution is empty");
                    }
                    for (double& value : residual) value /= total;
                    correction.token_id = sample_probabilities(residual, rng_);
                    correction.selected_logit =
                        p_logits[static_cast<std::size_t>(correction.token_id)];
                }
            } else {
                if (cuda_greedy_argmax) {
                    correction.token_id = target_argmax_indices.back();
                    correction.selected_logit = target_argmax_values.back();
                } else {
                    const std::vector<float>& bonus_logits = target_logits.back();
                    correction.token_id = greedy
                        ? argmax_token(bonus_logits)
                        : sample_probabilities(
                              processed_probabilities(bonus_logits, config), rng_);
                    correction.selected_logit =
                        bonus_logits[static_cast<std::size_t>(correction.token_id)];
                }
            }
            if (contains_token(config.stop_token_ids, correction.token_id)) {
                correction.finish_reason = "stop";
            }
            result.tokens.push_back(std::move(correction));
        }
        if (cuda_verification) {
            const bool device_target_taps = assistant_model_->cuda_device_tap_pack();
            const std::int64_t committed_anchor_rows = lagged_cuda ? 1 : 0;
            const std::int64_t committed_rows = committed_anchor_rows + accepted;
            const bool fused_lagged_context = device_target_taps && lagged_cuda &&
                static_cast<std::int64_t>(result.tokens.size()) > accepted;
            model_->cuda_model_->commit_verification(
                cache_->cuda, cuda_verification, committed_rows, false);
            const std::int64_t accepted_start = cache_length_;
            for (std::int64_t row = 0; row < accepted; ++row) {
                const std::int64_t token = proposal.token_ids[static_cast<std::size_t>(row)];
                tokens_.push_back(token);
            }
            if (committed_rows > 0) {
                if (device_target_taps) {
                    if (cuda_verification->device_target_taps() == nullptr) {
                        throw std::runtime_error(
                            "Glimmer CUDA verifier returned no device target taps");
                    }
                    if (fused_lagged_context) {
                        assistant_session_->
                            record_target_taps_batch_device_and_prepare_lagged_anchor(
                                accepted_start,
                                cuda_verification->device_target_taps(),
                                cuda_verification->cuda_device(),
                                cuda_verification->rows(), committed_rows,
                                cancelled_);
                    } else {
                        assistant_session_->record_target_taps_batch_device(
                            accepted_start, cuda_verification->device_target_taps(),
                            cuda_verification->cuda_device(), cuda_verification->rows(),
                            committed_rows, cancelled_);
                    }
                } else {
                    const std::vector<float>& taps = cuda_verification->target_taps();
                    const std::int64_t tap_width = assistant_model_->target_tap_width();
                    if (static_cast<std::int64_t>(taps.size()) !=
                        cuda_verification->rows() * tap_width) {
                        throw std::runtime_error(
                            "Glimmer CUDA verifier returned malformed target taps");
                    }
                    assistant_session_->record_target_taps_batch(
                        accepted_start, taps.data(), committed_rows, cancelled_);
                }
            }
            cache_length_ += committed_rows;
            if (static_cast<std::int64_t>(result.tokens.size()) > accepted) {
                const DecodeResult& decoded = result.tokens.back();
                if (lagged_cuda) {
                    tokens_.push_back(decoded.token_id);
                    if (!fused_lagged_context) {
                        assistant_session_->prepare_lagged_anchor(
                            cache_length_, cancelled_);
                    }
                    speculative_pending_token_ = true;
                } else {
                    append_cached_token(decoded.token_id, cache_length_, fast_k_quant);
                    ++cache_length_;
                    ++actual_target_rows;
                    tokens_.push_back(decoded.token_id);
                }
            } else if (lagged_cuda) {
                speculative_pending_token_ = false;
            }
        } else {
            const bool retained_full_proposal = accepted == proposal_count && !accepted_stop &&
                (static_cast<std::int64_t>(result.tokens.size()) == proposal_count ||
                 static_cast<std::int64_t>(result.tokens.size()) == proposal_count + 1);
            if (retained_full_proposal) {
                if (static_cast<std::int64_t>(result.tokens.size()) == proposal_count + 1) {
                    const std::int64_t bonus = result.tokens.back().token_id;
                    append_cached_token(bonus, cache_length_, fast_k_quant);
                    ++cache_length_;
                    ++actual_target_rows;
                    tokens_.push_back(bonus);
                }
            } else {
                tokens_.resize(static_cast<std::size_t>(original_length));
                cache_length_ = original_length;
                rebuild_cache();
                for (const DecodeResult& decoded : result.tokens) {
                    append_cached_token(decoded.token_id, cache_length_, fast_k_quant);
                    ++cache_length_;
                    ++actual_target_rows;
                    tokens_.push_back(decoded.token_id);
                }
            }
        }
        if (cuda_verification) {
            // Target and assistant use distinct CUDA streams.  The target
            // cache commit was intentionally deferred so it could overlap the
            // assistant context update above; complete it before exposing the
            // atomic speculative block to the caller.
            model_->cuda_model_->synchronize();
        }
        result.proposed_tokens = proposal_count;
        result.accepted_tokens = accepted;
        result.rejected_tokens = proposal_count - accepted;
        result.target_rows = actual_target_rows;
        result.assistant_blocks = 1;
        ++speculative_blocks_;
        speculative_proposed_ += result.proposed_tokens;
        speculative_accepted_ += result.accepted_tokens;
        speculative_rejected_ += result.rejected_tokens;
        decode_rows_processed_ += actual_target_rows;
        decode_calls_ += static_cast<std::int64_t>(result.tokens.size());
        strict_model_compute_ = config.temperature == 0.0;
        return result;
    } catch (...) {
        rng_ = rng_before;
        active_generation_seed_ = generation_seed_before;
        tokens_.resize(static_cast<std::size_t>(original_length));
        cache_length_ = original_length;
        try {
            rebuild_cache();
            speculative_ready_ = original_speculative_ready;
        } catch (...) {
            closed_ = true;
            model_->session_closed();
        }
        throw;
    }
}

void GlimmerSession::truncate(std::int64_t token_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::shared_lock<std::shared_mutex> model_lock(model_->lifecycle_mutex_);
    require_open();
    if (token_count < 0 || token_count > static_cast<std::int64_t>(tokens_.size())) {
        throw std::runtime_error("truncate token_count is outside the native session history");
    }
    if (token_count != static_cast<std::int64_t>(tokens_.size())) {
        const std::vector<std::int64_t> original = tokens_;
        const auto original_embeddings = embedding_overrides_;
        tokens_.resize(static_cast<std::size_t>(token_count));
        for (auto iterator = embedding_overrides_.begin(); iterator != embedding_overrides_.end();) {
            if (iterator->first >= token_count) iterator = embedding_overrides_.erase(iterator);
            else ++iterator;
        }
        try {
            rebuild_cache();
        } catch (...) {
            tokens_ = original;
            embedding_overrides_ = original_embeddings;
            rebuild_cache();
            throw;
        }
        speculative_pending_token_ = false;
        speculative_ready_ = false;
    }
    ++truncate_calls_;
}

void GlimmerSession::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::shared_lock<std::shared_mutex> model_lock(model_->lifecycle_mutex_);
    require_open();
    tokens_.clear();
    embedding_overrides_.clear();
    cache_length_ = 0;
    if (cache_mode_ == KVCacheMode::Full) {
        cache_ = model_->create_cache();
    }
    if (assistant_model_) assistant_session_ = assistant_model_->create_session();
    active_generation_seed_.reset();
    rng_.seed(static_cast<std::mt19937_64::result_type>(seed_));
    cancelled_.store(false);
    strict_model_compute_ = false;
    speculative_pending_token_ = false;
    speculative_ready_ = false;
    ++reset_calls_;
}

void GlimmerSession::cancel() noexcept {
    cancelled_.store(true);
}

void GlimmerSession::close() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!closed_) {
        closed_ = true;
        cancelled_.store(true);
        tokens_.clear();
        embedding_overrides_.clear();
        cache_length_ = 0;
        cache_.reset();
        model_->session_closed();
    }
}

SessionStats GlimmerSession::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    SessionStats result;
    result.token_count = static_cast<std::int64_t>(tokens_.size());
    result.prefill_calls = prefill_calls_;
    result.prefill_tokens = prefill_tokens_;
    result.decode_calls = decode_calls_;
    result.truncate_calls = truncate_calls_;
    result.reset_calls = reset_calls_;
    result.cached_tokens = cache_mode_ == KVCacheMode::Full ? cache_length_ : 0;
    result.cache_bytes = cache_mode_ == KVCacheMode::Full ? cache_bytes() : 0;
    result.uncompressed_cache_bytes = result.cache_bytes;
    if (cache_) {
        std::int64_t capacity = cache_->cuda ? cache_->cuda->allocated_bytes() : checked_mul(
            static_cast<std::int64_t>(cache_->final_hidden.capacity()), sizeof(float),
            "final hidden capacity");
        if (!cache_->cuda) {
            for (const GlimmerLayerCache& layer : cache_->layers) {
                capacity = checked_add(capacity, checked_mul(
                    static_cast<std::int64_t>(layer.keys.capacity() + layer.values.capacity()),
                    sizeof(float), "KV capacity"), "cache capacity");
            }
        }
        if (assistant_session_) {
            capacity = checked_add(
                capacity, assistant_session_->cache_bytes(), "DFlash cache capacity");
        }
        result.cache_capacity_bytes = capacity;
    }
    result.decode_rows_processed = decode_rows_processed_;
    result.speculative_blocks = speculative_blocks_;
    result.speculative_proposed_tokens = speculative_proposed_;
    result.speculative_accepted_tokens = speculative_accepted_;
    result.speculative_rejected_tokens = speculative_rejected_;
    result.assistant_cache_bytes = assistant_session_ ? assistant_session_->cache_bytes() : 0;
    result.speculative_decoding = static_cast<bool>(assistant_session_);
    result.cache_mode = cache_mode_;
    result.strict_model_compute = strict_model_compute_;
    result.lossy_cache = false;
    result.cancelled = cancelled_.load();
    result.closed = closed_;
    return result;
}

std::shared_ptr<GlimmerSession> GlimmerSession::fork_prefix(
    std::int64_t token_count,
    std::int64_t seed) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::shared_lock<std::shared_mutex> model_lock(model_->lifecycle_mutex_);
    require_open();
    if (cache_mode_ != KVCacheMode::Full || token_count <= 0 ||
        token_count > static_cast<std::int64_t>(tokens_.size())) {
        throw std::runtime_error("resident Glimmer prefix fork requires a non-empty lossless prefix");
    }
    auto child = std::make_shared<GlimmerSession>(
        model_, seed, KVCacheMode::Full, assistant_model_);
    std::atomic<bool> not_cancelled{false};
    for (std::int64_t position = 0; position < token_count; ++position) {
        const std::int64_t token = tokens_[static_cast<std::size_t>(position)];
        std::vector<float> taps;
        const auto replacement = embedding_overrides_.find(position);
        if (replacement == embedding_overrides_.end()) {
            model_->forward_append_token(
                token, position, child->cache_.get(), not_cancelled,
                child->assistant_session_ ? &assistant_model_->target_layer_ids() : nullptr,
                child->assistant_session_ ? &taps : nullptr);
        } else {
            model_->forward_append_embedding(
                replacement->second, position, child->cache_.get(), not_cancelled,
                child->assistant_session_ ? &assistant_model_->target_layer_ids() : nullptr,
                child->assistant_session_ ? &taps : nullptr);
            child->embedding_overrides_.emplace(position, replacement->second);
        }
        if (child->assistant_session_) {
            child->assistant_session_->record_target_taps(position, taps, not_cancelled);
        }
        child->tokens_.push_back(token);
        ++child->cache_length_;
    }
    return child;
}

}  // namespace neuralfn::resident_glimmer
