#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace neuralfn::resident_dense {

class TileTurboQuantSession;
struct TileTurboQuantSessionStats;

class ResidentCancellationError final : public std::runtime_error {
public:
    explicit ResidentCancellationError(const char* message)
        : std::runtime_error(message) {}
};

enum class TurboQuantProfile {
    Mse35,
    Qjl35,
};

struct TurboQuantTables {
    std::int64_t dimension = 0;
    TurboQuantProfile profile = TurboQuantProfile::Mse35;
    std::vector<double> rotation;
    std::vector<double> qjl_projection;
    std::vector<std::int64_t> value_bit_widths;
    std::vector<std::int64_t> key_bit_widths;
    // Index by bit width. Widths 2, 3, and 4 are used by the shipped
    // 3.5-bit profiles; unused entries remain empty.
    std::vector<std::vector<double>> centroids;
};

struct TurboQuantEncodedVector {
    float norm = 0.0f;
    float residual_norm = 0.0f;
    std::vector<std::uint8_t> packed_indices;
    std::vector<std::uint8_t> qjl_signs;
};

// Native correctness implementation shared by the resident cache and codec
// agreement tests. It operates on one attention head at a time and never
// constructs a dequantized cache matrix.
class TurboQuantCodec final {
public:
    explicit TurboQuantCodec(TurboQuantTables tables);

    TurboQuantEncodedVector encode_key(const float* vector) const;
    TurboQuantEncodedVector encode_value(const float* vector) const;
    double key_inner_product(
        const float* query,
        const TurboQuantEncodedVector& encoded) const;
    void accumulate_value(
        float* output,
        double weight,
        const TurboQuantEncodedVector& encoded) const;

    std::int64_t dimension() const noexcept { return tables_.dimension; }
    TurboQuantProfile profile() const noexcept { return tables_.profile; }
    std::size_t key_record_bytes() const noexcept { return key_record_bytes_; }
    std::size_t value_record_bytes() const noexcept { return value_record_bytes_; }
    bool matches(const TurboQuantTables& tables) const noexcept;

    const TurboQuantTables& tables() const noexcept { return tables_; }

private:
    TurboQuantEncodedVector encode_mse(
        const float* vector,
        const std::vector<std::int64_t>& bit_widths,
        bool include_qjl) const;
    std::vector<double> decode_unit(
        const TurboQuantEncodedVector& encoded,
        const std::vector<std::int64_t>& bit_widths) const;
    std::vector<double> rotate(const float* vector) const;
    std::vector<double> project_qjl(const float* vector) const;
    void validate_encoded(
        const TurboQuantEncodedVector& encoded,
        bool qjl_key) const;

    TurboQuantTables tables_;
    std::size_t key_index_bytes_ = 0;
    std::size_t value_index_bytes_ = 0;
    std::size_t sign_bytes_ = 0;
    std::size_t key_record_bytes_ = 0;
    std::size_t value_record_bytes_ = 0;
};

// Preallocated per-session storage. Records are packed contiguously by
// layer/position/head; only the final hidden row stays lossless because it is
// required to produce the next logits without replaying the prefix.
class TurboQuantCache final {
public:
    // Packed K/V bytes are immutable while this allocation has more than one
    // session owner.  Session-local wrappers retain their own counters and
    // codec reference; only this storage is shared by a prefix fork.
    struct Storage {
        std::vector<std::uint8_t> key_bytes;
        std::vector<std::uint8_t> value_bytes;
    };

    TurboQuantCache(
        std::int64_t num_layers,
        std::int64_t num_heads,
        std::int64_t max_seq_len,
        std::int64_t channels,
        std::shared_ptr<const TurboQuantCodec> codec,
        std::unique_ptr<TileTurboQuantSession> tile_session = nullptr);
    ~TurboQuantCache();

    void encode_row(
        std::int64_t layer,
        std::int64_t position,
        const float* key,
        const float* value,
        const std::atomic<bool>& cancelled);
    double key_inner_product(
        std::int64_t layer,
        std::int64_t position,
        std::int64_t head,
        const float* query) const;
    void accumulate_value(
        std::int64_t layer,
        std::int64_t position,
        std::int64_t head,
        double weight,
        float* output) const;
    bool tile_attention_enabled() const noexcept;
    void tile_attention(
        std::int64_t layer,
        std::int64_t past_sequence_length,
        const float* query,
        const float* current_key,
        const float* current_value,
        float* output,
        float scale,
        const std::atomic<bool>& cancelled);

    std::int64_t actual_bytes_per_token() const;
    std::int64_t uncompressed_bytes_per_token() const;
    std::int64_t capacity_bytes() const;
    std::string profile_name() const;
    std::int64_t cpu_compressed_attention_calls() const noexcept;
    TileTurboQuantSessionStats tile_stats() const;
    std::unique_ptr<TurboQuantCache> fork_shared_cpu() const;
    std::shared_ptr<Storage> storage_handle() const noexcept { return storage_; }
    std::shared_ptr<Storage> clone_storage() const;
    void replace_storage(std::shared_ptr<Storage> storage) noexcept {
        storage_ = std::move(storage);
    }
    std::int64_t storage_use_count() const noexcept;

private:
    TurboQuantCache(
        std::int64_t num_layers,
        std::int64_t num_heads,
        std::int64_t max_seq_len,
        std::int64_t channels,
        std::shared_ptr<const TurboQuantCodec> codec,
        std::shared_ptr<Storage> storage);
    std::size_t record_offset(
        std::int64_t layer,
        std::int64_t position,
        std::int64_t head,
        std::size_t record_bytes) const;
    void write_record(
        std::vector<std::uint8_t>* storage,
        std::size_t offset,
        std::size_t record_bytes,
        const TurboQuantEncodedVector& encoded,
        bool qjl_key);
    TurboQuantEncodedVector read_record(
        const std::vector<std::uint8_t>& storage,
        std::size_t offset,
        std::size_t record_bytes,
        bool qjl_key) const;

    const std::int64_t num_layers_;
    const std::int64_t num_heads_;
    const std::int64_t max_seq_len_;
    const std::int64_t channels_;
    const std::int64_t head_dim_;
    std::shared_ptr<const TurboQuantCodec> codec_;
    std::shared_ptr<Storage> storage_;
    std::unique_ptr<TileTurboQuantSession> tile_session_;
    mutable std::atomic<std::int64_t> cpu_compressed_attention_calls_{0};
};

}  // namespace neuralfn::resident_dense
