#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace neuralfn::resident_dense {

class TurboQuantCodec;

struct TileTurboQuantConfig {
    std::string backend;
    std::string tile_ops_lib;
    std::string cuda_runtime_lib;
    std::int64_t device = 0;
};

struct TileTurboQuantModelStats {
    bool configured = false;
    std::string backend = "cpu";
    std::string tile_ops_lib;
    std::string cuda_runtime_lib;
    std::int64_t device = -1;
};

struct TileTurboQuantSessionStats {
    std::string backend = "cpu";
    std::string tile_ops_lib;
    std::string cuda_runtime_lib;
    std::int64_t device = -1;
    std::int64_t gpu_launches = 0;
    std::int64_t row_uploads = 0;
    std::int64_t h2d_bytes = 0;
    std::int64_t d2h_bytes = 0;
};

class TileTurboQuantSession final {
public:
    ~TileTurboQuantSession();

    TileTurboQuantSession(const TileTurboQuantSession&) = delete;
    TileTurboQuantSession& operator=(const TileTurboQuantSession&) = delete;

    void attention(
        std::int64_t layer,
        std::int64_t past_sequence_length,
        const float* query,
        const float* current_key,
        const float* current_value,
        float* output,
        float scale,
        const std::atomic<bool>& cancelled);
    void upload_row(
        std::int64_t layer,
        std::int64_t position,
        const std::uint8_t* key_records,
        std::size_t key_bytes,
        const std::uint8_t* value_records,
        std::size_t value_bytes,
        const std::atomic<bool>& cancelled);
    TileTurboQuantSessionStats stats() const noexcept;

private:
    class Impl;
    explicit TileTurboQuantSession(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;

    friend class TileTurboQuantModel;
};

class TileTurboQuantModel final {
public:
    static std::shared_ptr<TileTurboQuantModel> configure(
        TileTurboQuantConfig config,
        std::int64_t num_layers,
        std::int64_t num_heads,
        std::int64_t channels,
        std::int64_t max_seq_len);
    ~TileTurboQuantModel();

    TileTurboQuantModel(const TileTurboQuantModel&) = delete;
    TileTurboQuantModel& operator=(const TileTurboQuantModel&) = delete;

    std::unique_ptr<TileTurboQuantSession> create_session(
        std::shared_ptr<const TurboQuantCodec> codec);
    TileTurboQuantModelStats stats() const noexcept;

private:
    class Impl;
    explicit TileTurboQuantModel(std::shared_ptr<Impl> impl);

    std::shared_ptr<Impl> impl_;

    friend class TileTurboQuantSession;
};

}  // namespace neuralfn::resident_dense
