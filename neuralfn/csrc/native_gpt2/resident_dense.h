#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <shared_mutex>
#include <string>
#include <vector>

#include "resident_tile_turboquant.h"
#include "resident_turboquant.h"

namespace neuralfn::resident_dense {

inline constexpr int kResidentInferenceAbiVersion = 1;

struct GenerationConfig {
    double temperature = 0.8;
    std::int64_t top_k = 0;
    double top_p = 1.0;
    std::optional<std::int64_t> seed;
    std::vector<std::int64_t> stop_token_ids;
};

struct DecodeResult {
    std::int64_t token_id = 0;
    float selected_logit = 0.0f;
    std::string finish_reason;
};

enum class KVCacheMode {
    Off,
    Full,
    TurboQuant,
};

enum class MlpActivation {
    GeluExact,
    GeluTanh,
    Relu,
    Silu,
    ReluSquared,
};

struct DenseInferenceConfig {
    bool use_qk_norm = false;
    double qk_norm_eps = 1.0e-6;
    double logit_softcap = 0.0;
    bool moa_mode = false;
    std::int64_t moa_interval = 0;
    MlpActivation mlp_activation = MlpActivation::GeluExact;
};

struct ModelStats {
    std::string checkpoint_path;
    std::int64_t max_seq_len = 0;
    std::int64_t vocab_size = 0;
    std::int64_t padded_vocab_size = 0;
    std::int64_t num_layers = 0;
    std::int64_t num_heads = 0;
    std::int64_t channels = 0;
    std::int64_t parameter_count = 0;
    std::int64_t weight_bytes = 0;
    std::int64_t weights_load_count = 0;
    std::int64_t open_sessions = 0;
    std::int64_t forward_calls = 0;
    std::int64_t turboquant_table_load_count = 0;
    bool turboquant_tile_attention_configured = false;
    std::string turboquant_attention_backend = "cpu";
    std::string turboquant_tile_ops_lib;
    std::string turboquant_cuda_runtime_lib;
    std::int64_t turboquant_cuda_device = -1;
    bool use_qk_norm = false;
    double qk_norm_eps = 1.0e-6;
    double logit_softcap = 0.0;
    bool moa_mode = false;
    std::int64_t moa_interval = 0;
    std::string mlp_activation;
};

struct SessionStats {
    std::int64_t token_count = 0;
    std::int64_t prefill_calls = 0;
    std::int64_t prefill_tokens = 0;
    std::int64_t decode_calls = 0;
    std::int64_t truncate_calls = 0;
    std::int64_t reset_calls = 0;
    std::int64_t cached_tokens = 0;
    std::int64_t cache_bytes = 0;
    std::int64_t cache_capacity_bytes = 0;
    std::int64_t uncompressed_cache_bytes = 0;
    std::int64_t decode_rows_processed = 0;
    std::int64_t prefix_cow_forks_created = 0;
    std::int64_t prefix_cow_forked_from_tokens = 0;
    std::int64_t prefix_cow_storage_use_count = 0;
    // This session's valid cached rows that currently live in a shared
    // allocation.  This is deliberately not the minimum common prefix across
    // every owner; owners may expose different logical prefix lengths.
    std::int64_t prefix_cow_shared_cached_tokens = 0;
    std::int64_t prefix_cow_shared_capacity_bytes = 0;
    std::int64_t prefix_cow_detach_count = 0;
    std::int64_t prefix_cow_detached_capacity_bytes = 0;
    KVCacheMode cache_mode = KVCacheMode::Off;
    std::string turboquant_profile;
    std::string turboquant_attention_backend = "cpu";
    std::string turboquant_tile_ops_lib;
    std::string turboquant_cuda_runtime_lib;
    std::int64_t turboquant_cuda_device = -1;
    std::int64_t turboquant_gpu_launches = 0;
    std::int64_t turboquant_row_uploads = 0;
    std::int64_t turboquant_h2d_bytes = 0;
    std::int64_t turboquant_d2h_bytes = 0;
    std::int64_t turboquant_cpu_compressed_attention_calls = 0;
    bool strict_model_compute = false;
    bool lossy_cache = false;
    bool cancelled = false;
    bool closed = false;
};

class DenseModel;

class DenseSession final {
public:
    DenseSession(
        std::shared_ptr<DenseModel> model,
        std::int64_t seed,
        KVCacheMode cache_mode,
        std::shared_ptr<const TurboQuantCodec> turboquant_codec = nullptr,
        std::unique_ptr<TileTurboQuantSession> tile_turboquant_session = nullptr);
    ~DenseSession();

    DenseSession(const DenseSession&) = delete;
    DenseSession& operator=(const DenseSession&) = delete;

    void prefill(const std::vector<std::int64_t>& token_ids, std::int64_t start_position);
    std::vector<float> current_logits();
    DecodeResult decode_one(const GenerationConfig& config);
    void truncate(std::int64_t token_count);
    void reset();
    void cancel() noexcept;
    void close() noexcept;
    SessionStats stats() const;
    std::shared_ptr<DenseSession> fork_prefix(
        std::int64_t token_count,
        std::int64_t seed);

    const std::shared_ptr<DenseModel>& model() const noexcept { return model_; }

private:
    DenseSession(
        std::shared_ptr<DenseModel> model,
        std::int64_t seed,
        std::vector<std::int64_t> tokens,
        std::int64_t cache_length,
        std::shared_ptr<std::vector<std::vector<float>>> key_cache,
        std::shared_ptr<std::vector<std::vector<float>>> value_cache,
        std::shared_ptr<std::vector<float>> final_hidden_cache);
    DenseSession(
        std::shared_ptr<DenseModel> model,
        std::int64_t seed,
        std::vector<std::int64_t> tokens,
        std::int64_t cache_length,
        std::unique_ptr<TurboQuantCache> turboquant_cache,
        std::shared_ptr<std::vector<float>> final_hidden_cache);
    struct CowDetachSnapshot {
        bool detached = false;
        std::shared_ptr<std::vector<std::vector<float>>> key_cache;
        std::shared_ptr<std::vector<std::vector<float>>> value_cache;
        std::shared_ptr<TurboQuantCache::Storage> turboquant_storage;
        std::shared_ptr<std::vector<float>> final_hidden_cache;
        std::int64_t detach_count = 0;
        std::int64_t detached_capacity_bytes = 0;
    };
    void require_open() const;
    CowDetachSnapshot detach_cache_before_write();
    void rollback_cache_detach(CowDetachSnapshot snapshot) noexcept;
    std::int64_t full_cache_capacity_bytes() const;
    std::int64_t turboquant_cache_capacity_bytes() const;

    std::shared_ptr<DenseModel> model_;
    mutable std::mutex mutex_;
    std::vector<std::int64_t> tokens_;
    const KVCacheMode cache_mode_;
    std::shared_ptr<std::vector<std::vector<float>>> key_cache_;
    std::shared_ptr<std::vector<std::vector<float>>> value_cache_;
    std::unique_ptr<TurboQuantCache> turboquant_cache_;
    std::shared_ptr<std::vector<float>> final_hidden_cache_;
    std::int64_t cache_length_ = 0;
    const std::int64_t seed_;
    std::mt19937_64 rng_;
    std::optional<std::int64_t> active_generation_seed_;
    std::atomic<bool> cancelled_{false};
    bool closed_ = false;
    std::int64_t prefill_calls_ = 0;
    std::int64_t prefill_tokens_ = 0;
    std::int64_t decode_calls_ = 0;
    std::int64_t truncate_calls_ = 0;
    std::int64_t reset_calls_ = 0;
    std::int64_t decode_rows_processed_ = 0;
    std::int64_t prefix_cow_forks_created_ = 0;
    std::int64_t prefix_cow_forked_from_tokens_ = 0;
    std::int64_t prefix_cow_detach_count_ = 0;
    std::int64_t prefix_cow_detached_capacity_bytes_ = 0;
    bool strict_model_compute_ = false;
};

class DenseModel final : public std::enable_shared_from_this<DenseModel> {
public:
    static std::shared_ptr<DenseModel> load(
        const std::string& checkpoint_path,
        DenseInferenceConfig inference_config = {});

    DenseModel(const DenseModel&) = delete;
    DenseModel& operator=(const DenseModel&) = delete;

    std::shared_ptr<DenseSession> create_session(
        std::int64_t seed,
        KVCacheMode cache_mode,
        std::optional<TurboQuantTables> turboquant_tables = std::nullopt,
        bool tile_turboquant_attention = false);
    TileTurboQuantModelStats configure_turboquant_attention(
        TileTurboQuantConfig config);
    void close() noexcept;
    bool closed() const noexcept { return closed_.load(); }
    ModelStats stats() const;

    std::vector<float> forward_last_logits(
        const std::vector<std::int64_t>& tokens,
        const std::atomic<bool>& cancelled) const;
    void forward_append_token(
        std::int64_t token,
        std::int64_t position,
        std::vector<std::vector<float>>* key_cache,
        std::vector<std::vector<float>>* value_cache,
        TurboQuantCache* turboquant_cache,
        std::vector<float>* final_hidden_cache,
        const std::atomic<bool>& cancelled) const;
    std::vector<float> logits_from_hidden(const float* hidden) const;
    DecodeResult select_token(
        const std::vector<float>& logits,
        const std::vector<std::int64_t>& history,
        const GenerationConfig& config,
        std::mt19937_64& rng) const;

    std::int64_t max_seq_len() const noexcept { return max_seq_len_; }
    std::int64_t vocab_size() const noexcept { return vocab_size_; }

    void session_opened() noexcept { open_sessions_.fetch_add(1); }
    void session_closed() noexcept { open_sessions_.fetch_sub(1); }

private:
    friend class DenseSession;

    struct BlockLayout {
        std::int64_t ln1_weight = 0;
        std::int64_t ln1_bias = 0;
        std::int64_t qkv_weight = 0;
        std::int64_t qkv_bias = 0;
        std::int64_t attn_proj_weight = 0;
        std::int64_t attn_proj_bias = 0;
        std::int64_t ln2_weight = 0;
        std::int64_t ln2_bias = 0;
        std::int64_t fc_weight = 0;
        std::int64_t fc_bias = 0;
        std::int64_t mlp_proj_weight = 0;
        std::int64_t mlp_proj_bias = 0;
    };

    DenseModel(
        std::string checkpoint_path,
        std::int64_t max_seq_len,
        std::int64_t vocab_size,
        std::int64_t num_layers,
        std::int64_t num_heads,
        std::int64_t channels,
        std::int64_t padded_vocab_size,
        DenseInferenceConfig inference_config,
        std::vector<float> weights);

    const float* at(std::int64_t offset) const;
    void require_open() const;
    std::shared_ptr<const TurboQuantCodec> resolve_turboquant_codec(
        TurboQuantTables tables);

    const std::string checkpoint_path_;
    const std::int64_t max_seq_len_;
    const std::int64_t vocab_size_;
    const std::int64_t num_layers_;
    const std::int64_t num_heads_;
    const std::int64_t channels_;
    const std::int64_t padded_vocab_size_;
    const DenseInferenceConfig inference_config_;
    const std::vector<float> weights_;
    std::int64_t wte_weight_ = 0;
    std::int64_t wpe_weight_ = 0;
    std::vector<BlockLayout> blocks_;
    std::int64_t final_ln_weight_ = 0;
    std::int64_t final_ln_bias_ = 0;
    std::atomic<bool> closed_{false};
    mutable std::shared_mutex lifecycle_mutex_;
    std::atomic<std::int64_t> open_sessions_{0};
    mutable std::atomic<std::int64_t> forward_calls_{0};
    mutable std::mutex turboquant_table_mutex_;
    std::shared_ptr<const TurboQuantCodec> mse_turboquant_codec_;
    std::shared_ptr<const TurboQuantCodec> qjl_turboquant_codec_;
    std::atomic<std::int64_t> turboquant_table_load_count_{0};
    mutable std::mutex tile_turboquant_mutex_;
    std::shared_ptr<TileTurboQuantModel> tile_turboquant_model_;
};

}  // namespace neuralfn::resident_dense
