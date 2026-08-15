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

#include "resident_dense.h"

namespace neuralfn::resident_llama {

// The LLaMA adapter deliberately shares the additive resident ABI value and
// generation/stat structures with the dense adapter.  Its checkpoint and
// cache contracts remain distinct and are validated before construction.
using neuralfn::resident_dense::DecodeResult;
using neuralfn::resident_dense::GenerationConfig;
using neuralfn::resident_dense::KVCacheMode;
using neuralfn::resident_dense::ModelStats;
using neuralfn::resident_dense::SessionStats;

struct LlamaInferenceConfig {
    std::int64_t max_seq_len = 0;
    std::int64_t vocab_size = 0;
    std::int64_t padded_vocab_size = 0;
    std::int64_t num_layers = 0;
    std::int64_t model_dim = 0;
    std::int64_t hidden_dim = 0;
    std::int64_t num_heads = 0;
    std::int64_t num_kv_heads = 0;
    std::int64_t head_dim = 0;
    double rope_theta = 10000.0;
    double rope_scaling_factor = 1.0;
    double rms_norm_eps = 1.0e-6;
    bool standard_moe = false;
    std::int64_t experts = 0;
    std::int64_t top_k = 0;
    double mlp_multiplier = 0.0;
    std::int64_t multiple_of = 0;
    double router_aux_loss_coef = 0.0;
    std::string checkpoint_sha256;
};

class LlamaModel;

class LlamaSession final {
public:
    LlamaSession(
        std::shared_ptr<LlamaModel> model,
        std::int64_t seed,
        KVCacheMode cache_mode);
    ~LlamaSession();

    LlamaSession(const LlamaSession&) = delete;
    LlamaSession& operator=(const LlamaSession&) = delete;

    void prefill(const std::vector<std::int64_t>& token_ids, std::int64_t start_position);
    std::vector<float> current_logits();
    DecodeResult decode_one(const GenerationConfig& config);
    void truncate(std::int64_t token_count);
    void reset();
    void cancel() noexcept;
    void close() noexcept;
    SessionStats stats() const;
    std::shared_ptr<LlamaSession> fork_prefix(
        std::int64_t token_count,
        std::int64_t seed);

    const std::shared_ptr<LlamaModel>& model() const noexcept { return model_; }

private:
    struct FullCacheStorage {
        std::vector<std::vector<float>> key_cache;
        std::vector<std::vector<float>> value_cache;
        std::vector<float> final_hidden_cache;
    };

    LlamaSession(
        std::shared_ptr<LlamaModel> model,
        std::int64_t seed,
        std::vector<std::int64_t> tokens,
        std::int64_t cache_length,
        std::shared_ptr<FullCacheStorage> full_cache);
    void require_open() const;
    void detach_full_cache_before_write();
    std::int64_t full_cache_capacity_bytes() const;

    std::shared_ptr<LlamaModel> model_;
    mutable std::mutex mutex_;
    std::vector<std::int64_t> tokens_;
    const KVCacheMode cache_mode_;
    std::shared_ptr<FullCacheStorage> full_cache_;
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

class LlamaModel final : public std::enable_shared_from_this<LlamaModel> {
public:
    static std::shared_ptr<LlamaModel> load(
        const std::string& checkpoint_path,
        LlamaInferenceConfig config);

    LlamaModel(const LlamaModel&) = delete;
    LlamaModel& operator=(const LlamaModel&) = delete;

    std::shared_ptr<LlamaSession> create_session(
        std::int64_t seed,
        KVCacheMode cache_mode);
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
        std::vector<float>* final_hidden_cache,
        const std::atomic<bool>& cancelled) const;
    std::vector<float> logits_from_hidden(const float* hidden) const;
    DecodeResult select_token(
        const std::vector<float>& logits,
        const GenerationConfig& config,
        std::mt19937_64& rng) const;

    std::int64_t max_seq_len() const noexcept { return config_.max_seq_len; }
    std::int64_t vocab_size() const noexcept { return config_.vocab_size; }
    std::int64_t model_dim() const noexcept { return config_.model_dim; }
    std::int64_t hidden_dim() const noexcept { return config_.hidden_dim; }
    std::int64_t num_kv_heads() const noexcept { return config_.num_kv_heads; }
    std::int64_t head_dim() const noexcept { return config_.head_dim; }
    double rope_theta() const noexcept { return config_.rope_theta; }
    double rms_norm_eps() const noexcept { return config_.rms_norm_eps; }
    bool standard_moe() const noexcept { return config_.standard_moe; }
    std::int64_t experts() const noexcept { return config_.experts; }
    std::int64_t top_k() const noexcept { return config_.top_k; }
    double mlp_multiplier() const noexcept { return config_.mlp_multiplier; }
    std::int64_t multiple_of() const noexcept { return config_.multiple_of; }
    double router_aux_loss_coef() const noexcept { return config_.router_aux_loss_coef; }
    std::int64_t kv_dim() const noexcept {
        return config_.num_kv_heads * config_.head_dim;
    }

    void session_opened() noexcept { open_sessions_.fetch_add(1); }
    void session_closed() noexcept { open_sessions_.fetch_sub(1); }

private:
    friend class LlamaSession;

    struct LayerLayout {
        std::int64_t attention_norm = 0;
        std::int64_t q_proj = 0;
        std::int64_t k_proj = 0;
        std::int64_t v_proj = 0;
        std::int64_t attention_out = 0;
        std::int64_t ffn_norm = 0;
        std::int64_t ffn_gate = 0;
        std::int64_t ffn_up = 0;
        std::int64_t ffn_down = 0;
        std::int64_t router = 0;
        std::int64_t experts_gate = 0;
        std::int64_t experts_up = 0;
        std::int64_t experts_down = 0;
    };

    LlamaModel(
        std::string checkpoint_path,
        LlamaInferenceConfig config,
        std::vector<float> weights);

    const float* at(std::int64_t offset) const;
    void require_open() const;

    const std::string checkpoint_path_;
    const LlamaInferenceConfig config_;
    const std::vector<float> weights_;
    std::int64_t token_embedding_ = 0;
    std::int64_t final_norm_ = 0;
    std::int64_t lm_head_ = 0;
    std::vector<LayerLayout> layers_;
    std::atomic<bool> closed_{false};
    mutable std::shared_mutex lifecycle_mutex_;
    std::atomic<std::int64_t> open_sessions_{0};
    mutable std::atomic<std::int64_t> forward_calls_{0};
};

}  // namespace neuralfn::resident_llama
