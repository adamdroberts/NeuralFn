#include "resident_turboquant.h"

#include "resident_tile_turboquant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace neuralfn::resident_dense {
namespace {

std::size_t packed_bytes(const std::vector<std::int64_t>& widths) {
    std::int64_t bits = 0;
    for (std::int64_t width : widths) {
        if (width < 1 || width > 8 || bits > std::numeric_limits<std::int64_t>::max() - width) {
            throw std::runtime_error("TurboQuant bit widths must be integers in 1..8");
        }
        bits += width;
    }
    return static_cast<std::size_t>((bits + 7) / 8);
}

std::int64_t checked_mul(std::int64_t left, std::int64_t right, const char* label) {
    if (left < 0 || right < 0 ||
        (left != 0 && right > std::numeric_limits<std::int64_t>::max() / left)) {
        throw std::runtime_error(std::string("TurboQuant size overflow at ") + label);
    }
    return left * right;
}

void require_finite(const std::vector<double>& values, const char* label) {
    if (!std::all_of(values.begin(), values.end(), [](double value) {
            return std::isfinite(value);
        })) {
        throw std::runtime_error(std::string("TurboQuant ") + label + " must be finite");
    }
}

void require_orthonormal(
    const std::vector<double>& matrix,
    std::int64_t dimension,
    const char* label) {
    constexpr double kTolerance = 1.0e-8;
    const std::size_t size = static_cast<std::size_t>(dimension);
    for (std::size_t left = 0; left < size; ++left) {
        for (std::size_t right = 0; right < size; ++right) {
            double row_dot = 0.0;
            double column_dot = 0.0;
            for (std::size_t index = 0; index < size; ++index) {
                row_dot += matrix[left * size + index] * matrix[right * size + index];
                column_dot += matrix[index * size + left] * matrix[index * size + right];
            }
            const double expected = left == right ? 1.0 : 0.0;
            if (std::abs(row_dot - expected) > kTolerance ||
                std::abs(column_dot - expected) > kTolerance) {
                throw std::runtime_error(
                    std::string("TurboQuant ") + label + " must be orthonormal");
            }
        }
    }
}

void require_nondegenerate_projection(
    const std::vector<double>& matrix,
    std::int64_t dimension) {
    const std::size_t size = static_cast<std::size_t>(dimension);
    for (std::size_t row = 0; row < size; ++row) {
        double squared_norm = 0.0;
        for (std::size_t column = 0; column < size; ++column) {
            const double value = matrix[row * size + column];
            squared_norm += value * value;
        }
        if (!(squared_norm > 1.0e-20)) {
            throw std::runtime_error("TurboQuant QJL projection rows must be nondegenerate");
        }
    }
}

std::int64_t validated_head_dimension(std::int64_t num_heads, std::int64_t channels) {
    if (num_heads <= 0 || channels <= 0 || channels % num_heads != 0) {
        throw std::runtime_error("TurboQuant cache geometry does not match the resident model");
    }
    return channels / num_heads;
}

std::vector<std::uint8_t> pack_indices(
    const std::vector<std::int64_t>& indices,
    const std::vector<std::int64_t>& widths) {
    if (indices.size() != widths.size()) {
        throw std::runtime_error("TurboQuant index and width counts do not match");
    }
    std::vector<std::uint8_t> output(packed_bytes(widths), 0);
    std::size_t bit_offset = 0;
    for (std::size_t index = 0; index < indices.size(); ++index) {
        const std::int64_t width = widths[index];
        const std::int64_t value = indices[index];
        if (value < 0 || value >= (std::int64_t{1} << width)) {
            throw std::runtime_error("TurboQuant index does not fit its packed bit width");
        }
        for (std::int64_t bit = 0; bit < width; ++bit) {
            if ((value & (std::int64_t{1} << bit)) != 0) {
                const std::size_t absolute = bit_offset + static_cast<std::size_t>(bit);
                output[absolute / 8] |= static_cast<std::uint8_t>(1u << (absolute % 8));
            }
        }
        bit_offset += static_cast<std::size_t>(width);
    }
    return output;
}

std::vector<std::int64_t> unpack_indices(
    const std::vector<std::uint8_t>& packed,
    const std::vector<std::int64_t>& widths) {
    if (packed.size() != packed_bytes(widths)) {
        throw std::runtime_error("TurboQuant packed index payload has the wrong byte count");
    }
    std::vector<std::int64_t> output;
    output.reserve(widths.size());
    std::size_t bit_offset = 0;
    for (std::int64_t width : widths) {
        std::int64_t value = 0;
        for (std::int64_t bit = 0; bit < width; ++bit) {
            const std::size_t absolute = bit_offset + static_cast<std::size_t>(bit);
            if ((packed[absolute / 8] & static_cast<std::uint8_t>(1u << (absolute % 8))) != 0) {
                value |= std::int64_t{1} << bit;
            }
        }
        output.push_back(value);
        bit_offset += static_cast<std::size_t>(width);
    }
    const std::size_t total_bits = bit_offset;
    for (std::size_t bit = total_bits; bit < packed.size() * 8; ++bit) {
        if ((packed[bit / 8] & static_cast<std::uint8_t>(1u << (bit % 8))) != 0) {
            throw std::runtime_error("TurboQuant packed index payload has non-zero padding bits");
        }
    }
    return output;
}

float read_float(const std::uint8_t* source) {
    float value = 0.0f;
    std::memcpy(&value, source, sizeof(float));
    return value;
}

void write_float(std::uint8_t* target, float value) {
    std::memcpy(target, &value, sizeof(float));
}

void throw_if_cancelled(const std::atomic<bool>& cancelled) {
    if (cancelled.load()) {
        throw ResidentCancellationError("resident inference generation was cancelled");
    }
}

}  // namespace

TurboQuantCodec::TurboQuantCodec(TurboQuantTables tables)
    : tables_(std::move(tables)) {
    const std::int64_t dimension = tables_.dimension;
    if (dimension < 2 || dimension % 2 != 0) {
        throw std::runtime_error("TurboQuant 3.5-bit head dimension must be an even integer >= 2");
    }
    const std::size_t matrix_size = static_cast<std::size_t>(
        checked_mul(dimension, dimension, "codec matrix geometry"));
    if (tables_.rotation.size() != matrix_size) {
        throw std::runtime_error("TurboQuant rotation matrix has the wrong geometry");
    }
    require_finite(tables_.rotation, "rotation matrix");
    require_orthonormal(tables_.rotation, dimension, "rotation matrix");
    if (tables_.value_bit_widths.size() != static_cast<std::size_t>(dimension) ||
        tables_.key_bit_widths.size() != static_cast<std::size_t>(dimension)) {
        throw std::runtime_error("TurboQuant bit-width tables have the wrong dimension");
    }
    for (std::size_t index = 0; index < static_cast<std::size_t>(dimension); ++index) {
        const std::int64_t value_width = tables_.value_bit_widths[index];
        const std::int64_t key_width = tables_.key_bit_widths[index];
        const std::int64_t expected_value = index % 2 == 0 ? 4 : 3;
        if (value_width != expected_value) {
            throw std::runtime_error(
                "TurboQuant value widths must use the canonical even-channel outlier pattern");
        }
        const std::int64_t expected_key = tables_.profile == TurboQuantProfile::Qjl35
            ? value_width - 1
            : value_width;
        if (key_width != expected_key) {
            throw std::runtime_error("TurboQuant key widths do not match the selected profile");
        }
    }
    if (tables_.profile == TurboQuantProfile::Qjl35) {
        if (tables_.qjl_projection.size() != matrix_size) {
            throw std::runtime_error("TurboQuant QJL projection has the wrong geometry");
        }
        require_finite(tables_.qjl_projection, "QJL projection");
        require_nondegenerate_projection(tables_.qjl_projection, dimension);
    } else if (!tables_.qjl_projection.empty()) {
        throw std::runtime_error("TurboQuant MSE profile must not carry a QJL projection");
    }
    if (tables_.centroids.size() != 5) {
        throw std::runtime_error("TurboQuant centroid table must be indexed through width 4");
    }
    for (std::int64_t width : {std::int64_t{2}, std::int64_t{3}, std::int64_t{4}}) {
        const bool used = std::find(tables_.value_bit_widths.begin(), tables_.value_bit_widths.end(), width) !=
                tables_.value_bit_widths.end() ||
            std::find(tables_.key_bit_widths.begin(), tables_.key_bit_widths.end(), width) !=
                tables_.key_bit_widths.end();
        if (!used) {
            continue;
        }
        const auto& codebook = tables_.centroids[static_cast<std::size_t>(width)];
        if (codebook.size() != static_cast<std::size_t>(std::int64_t{1} << width) ||
            std::adjacent_find(
                codebook.begin(), codebook.end(), [](double left, double right) {
                    return !(left < right);
                }) != codebook.end()) {
            throw std::runtime_error("TurboQuant Lloyd-Max codebook has invalid width or ordering");
        }
        require_finite(codebook, "Lloyd-Max codebook");
        for (std::size_t index = 0; index < codebook.size(); ++index) {
            if (codebook[index] < -1.0 || codebook[index] > 1.0 ||
                std::abs(codebook[index] + codebook[codebook.size() - 1 - index]) > 1.0e-12) {
                throw std::runtime_error(
                    "TurboQuant Lloyd-Max codebook must be bounded and odd-symmetric");
            }
        }
    }
    key_index_bytes_ = packed_bytes(tables_.key_bit_widths);
    value_index_bytes_ = packed_bytes(tables_.value_bit_widths);
    sign_bytes_ = tables_.profile == TurboQuantProfile::Qjl35
        ? static_cast<std::size_t>(dimension / 8 + (dimension % 8 != 0 ? 1 : 0))
        : 0;
    key_record_bytes_ = sizeof(float) + key_index_bytes_ +
        (tables_.profile == TurboQuantProfile::Qjl35 ? sizeof(float) + sign_bytes_ : 0);
    value_record_bytes_ = sizeof(float) + value_index_bytes_;
}

bool TurboQuantCodec::matches(const TurboQuantTables& tables) const noexcept {
    return tables_.dimension == tables.dimension &&
        tables_.profile == tables.profile &&
        tables_.rotation == tables.rotation &&
        tables_.qjl_projection == tables.qjl_projection &&
        tables_.value_bit_widths == tables.value_bit_widths &&
        tables_.key_bit_widths == tables.key_bit_widths &&
        tables_.centroids == tables.centroids;
}

std::vector<double> TurboQuantCodec::rotate(const float* vector) const {
    if (vector == nullptr) {
        throw std::runtime_error("TurboQuant input vector is null");
    }
    const std::size_t dimension = static_cast<std::size_t>(tables_.dimension);
    std::vector<double> output(dimension, 0.0);
    for (std::size_t row = 0; row < dimension; ++row) {
        double sum = 0.0;
        for (std::size_t column = 0; column < dimension; ++column) {
            const float coordinate = vector[column];
            if (!std::isfinite(coordinate)) {
                throw std::runtime_error("TurboQuant input vector must be finite");
            }
            sum += tables_.rotation[row * dimension + column] * static_cast<double>(coordinate);
        }
        output[row] = sum;
    }
    return output;
}

std::vector<double> TurboQuantCodec::project_qjl(const float* vector) const {
    if (tables_.profile != TurboQuantProfile::Qjl35 || vector == nullptr) {
        throw std::runtime_error("TurboQuant QJL projection is unavailable");
    }
    const std::size_t dimension = static_cast<std::size_t>(tables_.dimension);
    std::vector<double> output(dimension, 0.0);
    for (std::size_t row = 0; row < dimension; ++row) {
        double sum = 0.0;
        for (std::size_t column = 0; column < dimension; ++column) {
            sum += tables_.qjl_projection[row * dimension + column] *
                static_cast<double>(vector[column]);
        }
        output[row] = sum;
    }
    return output;
}

TurboQuantEncodedVector TurboQuantCodec::encode_mse(
    const float* vector,
    const std::vector<std::int64_t>& bit_widths,
    bool include_qjl) const {
    if (vector == nullptr) {
        throw std::runtime_error("TurboQuant input vector is null");
    }
    const std::size_t dimension = static_cast<std::size_t>(tables_.dimension);
    double squared_norm = 0.0;
    for (std::size_t index = 0; index < dimension; ++index) {
        const float coordinate = vector[index];
        if (!std::isfinite(coordinate)) {
            throw std::runtime_error("TurboQuant input vector must be finite");
        }
        squared_norm += static_cast<double>(coordinate) * coordinate;
    }
    const double source_norm = std::sqrt(squared_norm);
    const float stored_norm = static_cast<float>(source_norm);
    if (!std::isfinite(stored_norm)) {
        throw std::runtime_error("TurboQuant vector norm is not representable as finite float32");
    }
    std::vector<float> unit(dimension, 0.0f);
    if (source_norm != 0.0) {
        for (std::size_t index = 0; index < dimension; ++index) {
            unit[index] = static_cast<float>(static_cast<double>(vector[index]) / source_norm);
        }
    }
    const std::vector<double> rotated = rotate(unit.data());
    std::vector<std::int64_t> indices;
    indices.reserve(dimension);
    std::vector<double> quantized_rotated(dimension, 0.0);
    for (std::size_t index = 0; index < dimension; ++index) {
        const std::int64_t width = bit_widths[index];
        const auto& codebook = tables_.centroids[static_cast<std::size_t>(width)];
        std::size_t selected = 0;
        double best = std::abs(rotated[index] - codebook[0]);
        for (std::size_t candidate = 1; candidate < codebook.size(); ++candidate) {
            const double distance = std::abs(rotated[index] - codebook[candidate]);
            if (distance < best) {
                best = distance;
                selected = candidate;
            }
        }
        indices.push_back(static_cast<std::int64_t>(selected));
        quantized_rotated[index] = codebook[selected];
    }
    TurboQuantEncodedVector encoded;
    encoded.norm = stored_norm;
    encoded.packed_indices = pack_indices(indices, bit_widths);
    if (!include_qjl) {
        return encoded;
    }

    std::vector<float> residual(dimension, 0.0f);
    double residual_squared_norm = 0.0;
    for (std::size_t column = 0; column < dimension; ++column) {
        double approximation = 0.0;
        for (std::size_t row = 0; row < dimension; ++row) {
            approximation += tables_.rotation[row * dimension + column] *
                quantized_rotated[row];
        }
        residual[column] = static_cast<float>(static_cast<double>(unit[column]) - approximation);
        residual_squared_norm += static_cast<double>(residual[column]) * residual[column];
    }
    encoded.residual_norm = static_cast<float>(std::sqrt(residual_squared_norm));
    if (!std::isfinite(encoded.residual_norm)) {
        throw std::runtime_error("TurboQuant residual norm is not representable as finite float32");
    }
    const std::vector<double> projected = project_qjl(residual.data());
    encoded.qjl_signs.assign(sign_bytes_, 0);
    for (std::size_t index = 0; index < dimension; ++index) {
        if (projected[index] >= 0.0) {
            encoded.qjl_signs[index / 8] |= static_cast<std::uint8_t>(1u << (index % 8));
        }
    }
    return encoded;
}

TurboQuantEncodedVector TurboQuantCodec::encode_key(const float* vector) const {
    return encode_mse(
        vector,
        tables_.key_bit_widths,
        tables_.profile == TurboQuantProfile::Qjl35);
}

TurboQuantEncodedVector TurboQuantCodec::encode_value(const float* vector) const {
    return encode_mse(vector, tables_.value_bit_widths, false);
}

void TurboQuantCodec::validate_encoded(
    const TurboQuantEncodedVector& encoded,
    bool qjl_key) const {
    const std::size_t expected_indices = qjl_key ? key_index_bytes_ : value_index_bytes_;
    if (!std::isfinite(encoded.norm) || encoded.packed_indices.size() != expected_indices) {
        throw std::runtime_error("TurboQuant encoded vector has invalid MSE payload");
    }
    if (qjl_key) {
        if (tables_.profile != TurboQuantProfile::Qjl35 ||
            !std::isfinite(encoded.residual_norm) || encoded.qjl_signs.size() != sign_bytes_) {
            throw std::runtime_error("TurboQuant encoded QJL key has invalid residual payload");
        }
    } else if (!encoded.qjl_signs.empty()) {
        throw std::runtime_error("TurboQuant value payload must not contain QJL signs");
    }
}

std::vector<double> TurboQuantCodec::decode_unit(
    const TurboQuantEncodedVector& encoded,
    const std::vector<std::int64_t>& bit_widths) const {
    const bool qjl_key = &bit_widths == &tables_.key_bit_widths &&
        tables_.profile == TurboQuantProfile::Qjl35;
    validate_encoded(encoded, qjl_key);
    const std::vector<std::int64_t> indices = unpack_indices(encoded.packed_indices, bit_widths);
    const std::size_t dimension = static_cast<std::size_t>(tables_.dimension);
    std::vector<double> output(dimension, 0.0);
    for (std::size_t column = 0; column < dimension; ++column) {
        double sum = 0.0;
        for (std::size_t row = 0; row < dimension; ++row) {
            const std::int64_t width = bit_widths[row];
            const double quantized = tables_.centroids[static_cast<std::size_t>(width)]
                [static_cast<std::size_t>(indices[row])];
            sum += tables_.rotation[row * dimension + column] * quantized;
        }
        output[column] = sum;
    }
    return output;
}

double TurboQuantCodec::key_inner_product(
    const float* query,
    const TurboQuantEncodedVector& encoded) const {
    const bool qjl = tables_.profile == TurboQuantProfile::Qjl35;
    validate_encoded(encoded, qjl);
    const auto& widths = tables_.key_bit_widths;
    const std::vector<std::int64_t> indices = unpack_indices(encoded.packed_indices, widths);
    const std::vector<double> rotated_query = rotate(query);
    double base = 0.0;
    for (std::size_t index = 0; index < widths.size(); ++index) {
        base += rotated_query[index] * tables_.centroids[static_cast<std::size_t>(widths[index])]
            [static_cast<std::size_t>(indices[index])];
    }
    if (!qjl) {
        return static_cast<double>(encoded.norm) * base;
    }
    const std::vector<double> projected_query = project_qjl(query);
    double signed_sum = 0.0;
    for (std::size_t index = 0; index < projected_query.size(); ++index) {
        const bool positive = (encoded.qjl_signs[index / 8] &
            static_cast<std::uint8_t>(1u << (index % 8))) != 0;
        signed_sum += projected_query[index] * (positive ? 1.0 : -1.0);
    }
    const double correction = std::sqrt(std::acos(-1.0) / 2.0) /
        static_cast<double>(tables_.dimension) * static_cast<double>(encoded.residual_norm) *
        signed_sum;
    return static_cast<double>(encoded.norm) * (base + correction);
}

void TurboQuantCodec::accumulate_value(
    float* output,
    double weight,
    const TurboQuantEncodedVector& encoded) const {
    if (output == nullptr || !std::isfinite(weight)) {
        throw std::runtime_error("TurboQuant value accumulation requires finite inputs");
    }
    validate_encoded(encoded, false);
    const std::vector<double> unit = decode_unit(encoded, tables_.value_bit_widths);
    for (std::size_t index = 0; index < unit.size(); ++index) {
        output[index] = static_cast<float>(
            static_cast<double>(output[index]) + weight * static_cast<double>(encoded.norm) * unit[index]);
    }
}

TurboQuantCache::TurboQuantCache(
    std::int64_t num_layers,
    std::int64_t num_heads,
    std::int64_t max_seq_len,
    std::int64_t channels,
    std::shared_ptr<const TurboQuantCodec> codec,
    std::unique_ptr<TileTurboQuantSession> tile_session)
    : num_layers_(num_layers),
      num_heads_(num_heads),
      max_seq_len_(max_seq_len),
      channels_(channels),
      head_dim_(validated_head_dimension(num_heads, channels)),
      codec_(std::move(codec)),
      storage_(std::make_shared<Storage>()),
      tile_session_(std::move(tile_session)) {
    if (num_layers <= 0 || max_seq_len <= 0 || !codec_ || codec_->dimension() != head_dim_) {
        throw std::runtime_error("TurboQuant cache geometry does not match the resident model");
    }
    const std::int64_t records = checked_mul(
        checked_mul(num_layers_, max_seq_len_, "layer positions"), num_heads_, "attention heads");
    storage_->key_bytes.assign(
        static_cast<std::size_t>(checked_mul(
            records, static_cast<std::int64_t>(codec_->key_record_bytes()), "key storage")),
        0);
    storage_->value_bytes.assign(
        static_cast<std::size_t>(checked_mul(
            records, static_cast<std::int64_t>(codec_->value_record_bytes()), "value storage")),
        0);
}

TurboQuantCache::TurboQuantCache(
    std::int64_t num_layers,
    std::int64_t num_heads,
    std::int64_t max_seq_len,
    std::int64_t channels,
    std::shared_ptr<const TurboQuantCodec> codec,
    std::shared_ptr<Storage> storage)
    : num_layers_(num_layers),
      num_heads_(num_heads),
      max_seq_len_(max_seq_len),
      channels_(channels),
      head_dim_(validated_head_dimension(num_heads, channels)),
      codec_(std::move(codec)),
      storage_(std::move(storage)) {
    if (num_layers <= 0 || max_seq_len <= 0 || !codec_ || !storage_ ||
        codec_->dimension() != head_dim_) {
        throw std::runtime_error("TurboQuant shared cache geometry is invalid");
    }
    const std::int64_t records = checked_mul(
        checked_mul(num_layers_, max_seq_len_, "layer positions"), num_heads_, "attention heads");
    const std::size_t expected_key_bytes = static_cast<std::size_t>(checked_mul(
        records, static_cast<std::int64_t>(codec_->key_record_bytes()), "key storage"));
    const std::size_t expected_value_bytes = static_cast<std::size_t>(checked_mul(
        records, static_cast<std::int64_t>(codec_->value_record_bytes()), "value storage"));
    if (storage_->key_bytes.size() != expected_key_bytes ||
        storage_->value_bytes.size() != expected_value_bytes) {
        throw std::runtime_error("TurboQuant shared cache storage geometry is invalid");
    }
}

TurboQuantCache::~TurboQuantCache() = default;

std::size_t TurboQuantCache::record_offset(
    std::int64_t layer,
    std::int64_t position,
    std::int64_t head,
    std::size_t record_bytes) const {
    if (layer < 0 || layer >= num_layers_ || position < 0 || position >= max_seq_len_ ||
        head < 0 || head >= num_heads_) {
        throw std::runtime_error("TurboQuant cache index is outside its preallocated geometry");
    }
    const std::int64_t record = (layer * max_seq_len_ + position) * num_heads_ + head;
    return static_cast<std::size_t>(record) * record_bytes;
}

void TurboQuantCache::write_record(
    std::vector<std::uint8_t>* storage,
    std::size_t offset,
    std::size_t record_bytes,
    const TurboQuantEncodedVector& encoded,
    bool qjl_key) {
    if (storage == nullptr || offset + record_bytes > storage->size()) {
        throw std::runtime_error("TurboQuant record write exceeds preallocated storage");
    }
    std::uint8_t* target = storage->data() + offset;
    write_float(target, encoded.norm);
    std::size_t cursor = sizeof(float);
    if (qjl_key) {
        write_float(target + cursor, encoded.residual_norm);
        cursor += sizeof(float);
    }
    std::copy(encoded.packed_indices.begin(), encoded.packed_indices.end(), target + cursor);
    cursor += encoded.packed_indices.size();
    if (qjl_key) {
        std::copy(encoded.qjl_signs.begin(), encoded.qjl_signs.end(), target + cursor);
        cursor += encoded.qjl_signs.size();
    }
    if (cursor != record_bytes) {
        throw std::runtime_error("TurboQuant encoded record does not match its fixed layout");
    }
}

TurboQuantEncodedVector TurboQuantCache::read_record(
    const std::vector<std::uint8_t>& storage,
    std::size_t offset,
    std::size_t record_bytes,
    bool qjl_key) const {
    if (offset + record_bytes > storage.size()) {
        throw std::runtime_error("TurboQuant record read exceeds preallocated storage");
    }
    const std::uint8_t* source = storage.data() + offset;
    TurboQuantEncodedVector encoded;
    encoded.norm = read_float(source);
    std::size_t cursor = sizeof(float);
    if (qjl_key) {
        encoded.residual_norm = read_float(source + cursor);
        cursor += sizeof(float);
    }
    const std::size_t index_bytes = qjl_key
        ? (record_bytes - sizeof(float) - sizeof(float) -
            static_cast<std::size_t>(
                head_dim_ / 8 + (head_dim_ % 8 != 0 ? 1 : 0)))
        : (record_bytes - sizeof(float));
    encoded.packed_indices.assign(source + cursor, source + cursor + index_bytes);
    cursor += index_bytes;
    if (qjl_key) {
        encoded.qjl_signs.assign(source + cursor, source + record_bytes);
        cursor = record_bytes;
    }
    if (cursor != record_bytes) {
        throw std::runtime_error("TurboQuant fixed record layout is inconsistent");
    }
    return encoded;
}

void TurboQuantCache::encode_row(
    std::int64_t layer,
    std::int64_t position,
    const float* key,
    const float* value,
    const std::atomic<bool>& cancelled) {
    if (!storage_ || storage_.use_count() != 1) {
        throw std::runtime_error(
            "TurboQuant packed storage must detach before a shared-prefix write");
    }
    const bool qjl = codec_->profile() == TurboQuantProfile::Qjl35;
    for (std::int64_t head = 0; head < num_heads_; ++head) {
        throw_if_cancelled(cancelled);
        const TurboQuantEncodedVector encoded_key = codec_->encode_key(key + head * head_dim_);
        const TurboQuantEncodedVector encoded_value = codec_->encode_value(value + head * head_dim_);
        write_record(
            &storage_->key_bytes,
            record_offset(layer, position, head, codec_->key_record_bytes()),
            codec_->key_record_bytes(),
            encoded_key,
            qjl);
        write_record(
            &storage_->value_bytes,
            record_offset(layer, position, head, codec_->value_record_bytes()),
            codec_->value_record_bytes(),
            encoded_value,
            false);
    }
    if (tile_session_) {
        const std::size_t key_offset = record_offset(
            layer, position, 0, codec_->key_record_bytes());
        const std::size_t value_offset = record_offset(
            layer, position, 0, codec_->value_record_bytes());
        tile_session_->upload_row(
            layer,
            position,
            storage_->key_bytes.data() + key_offset,
            static_cast<std::size_t>(num_heads_) * codec_->key_record_bytes(),
            storage_->value_bytes.data() + value_offset,
            static_cast<std::size_t>(num_heads_) * codec_->value_record_bytes(),
            cancelled);
    }
}

double TurboQuantCache::key_inner_product(
    std::int64_t layer,
    std::int64_t position,
    std::int64_t head,
    const float* query) const {
    cpu_compressed_attention_calls_.fetch_add(1, std::memory_order_relaxed);
    const bool qjl = codec_->profile() == TurboQuantProfile::Qjl35;
    const TurboQuantEncodedVector encoded = read_record(
        storage_->key_bytes,
        record_offset(layer, position, head, codec_->key_record_bytes()),
        codec_->key_record_bytes(),
        qjl);
    return codec_->key_inner_product(query, encoded);
}

void TurboQuantCache::accumulate_value(
    std::int64_t layer,
    std::int64_t position,
    std::int64_t head,
    double weight,
    float* output) const {
    cpu_compressed_attention_calls_.fetch_add(1, std::memory_order_relaxed);
    const TurboQuantEncodedVector encoded = read_record(
        storage_->value_bytes,
        record_offset(layer, position, head, codec_->value_record_bytes()),
        codec_->value_record_bytes(),
        false);
    codec_->accumulate_value(output, weight, encoded);
}

bool TurboQuantCache::tile_attention_enabled() const noexcept {
    return static_cast<bool>(tile_session_);
}

void TurboQuantCache::tile_attention(
    std::int64_t layer,
    std::int64_t past_sequence_length,
    const float* query,
    const float* current_key,
    const float* current_value,
    float* output,
    float scale,
    const std::atomic<bool>& cancelled) {
    if (!tile_session_) {
        throw std::runtime_error("Tile-CUDA TurboQuant attention is not configured for this cache");
    }
    tile_session_->attention(
        layer,
        past_sequence_length,
        query,
        current_key,
        current_value,
        output,
        scale,
        cancelled);
}

std::int64_t TurboQuantCache::actual_bytes_per_token() const {
    return checked_mul(
        checked_mul(
            num_layers_, num_heads_, "compressed K/V heads"),
        static_cast<std::int64_t>(codec_->key_record_bytes() + codec_->value_record_bytes()),
        "compressed K/V bytes");
}

std::int64_t TurboQuantCache::uncompressed_bytes_per_token() const {
    return checked_mul(
        checked_mul(
            checked_mul(num_layers_, channels_, "lossless K/V channels"),
            2,
            "lossless K/V vectors"),
        static_cast<std::int64_t>(sizeof(float)),
        "lossless K/V bytes");
}

std::int64_t TurboQuantCache::capacity_bytes() const {
    if (!storage_) {
        throw std::runtime_error("TurboQuant packed storage is unavailable");
    }
    return static_cast<std::int64_t>(
        storage_->key_bytes.size() + storage_->value_bytes.size());
}

std::string TurboQuantCache::profile_name() const {
    return codec_->profile() == TurboQuantProfile::Qjl35 ? "qjl-3.5" : "mse-3.5";
}

std::int64_t TurboQuantCache::cpu_compressed_attention_calls() const noexcept {
    return cpu_compressed_attention_calls_.load(std::memory_order_relaxed);
}

TileTurboQuantSessionStats TurboQuantCache::tile_stats() const {
    return tile_session_ ? tile_session_->stats() : TileTurboQuantSessionStats{};
}

std::unique_ptr<TurboQuantCache> TurboQuantCache::fork_shared_cpu() const {
    if (tile_session_) {
        throw std::runtime_error(
            "TurboQuant prefix COW rejects Tile-CUDA session storage");
    }
    if (!storage_) {
        throw std::runtime_error("TurboQuant packed storage is unavailable");
    }
    return std::unique_ptr<TurboQuantCache>(new TurboQuantCache(
        num_layers_,
        num_heads_,
        max_seq_len_,
        channels_,
        codec_,
        storage_));
}

std::shared_ptr<TurboQuantCache::Storage> TurboQuantCache::clone_storage() const {
    if (!storage_) {
        throw std::runtime_error("TurboQuant packed storage is unavailable");
    }
    return std::make_shared<Storage>(*storage_);
}

std::int64_t TurboQuantCache::storage_use_count() const noexcept {
    return storage_ ? static_cast<std::int64_t>(storage_.use_count()) : 0;
}

}  // namespace neuralfn::resident_dense
