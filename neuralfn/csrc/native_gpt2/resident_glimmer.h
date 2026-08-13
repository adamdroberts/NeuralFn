#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "resident_dense.h"

namespace neuralfn::resident_glimmer_cuda {
class Model;
class Cache;
}

namespace neuralfn::resident_glimmer_assistant {
class Model;
class Session;
}

namespace neuralfn::resident_glimmer_vision {
class Model;
}

namespace neuralfn::resident_glimmer {

using neuralfn::resident_dense::DecodeResult;
using neuralfn::resident_dense::GenerationConfig;
using neuralfn::resident_dense::KVCacheMode;
using neuralfn::resident_dense::ModelStats;
using neuralfn::resident_dense::SessionStats;

enum class WeightContainer {
    NativeBf16,
    GgufKQuant,
};

struct GlimmerInferenceConfig {
    std::int64_t max_seq_len = 131072;
    std::int64_t vocab_size = 202048;
    std::int64_t num_layers = 52;
    std::int64_t model_dim = 6656;
    std::int64_t intermediate_dim = 19968;
    std::int64_t num_heads = 32;
    std::int64_t num_kv_heads = 2;
    std::int64_t head_dim = 128;
    std::int64_t sliding_window = 2048;
    double rope_theta = 500000.0;
    double norm_eps = 1.0e-5;
    double post_norm_eps = 1.0e-8;
    double q_scale_factor = 3.87;
    double output_multiplier = 0.19611613513818404;
    double logit_softcap = 20.0;
    WeightContainer container = WeightContainer::NativeBf16;
    std::string checkpoint_sha256;
    // The Python/C++ binding authenticates the selected artifact immediately
    // before calling load().  Standalone callers leave this false and the
    // mapped bytes are hashed here as well.
    bool checkpoint_sha256_preverified = false;
    bool whole_model_cuda = false;
    int cuda_device = 0;
    std::string tile_ops_lib;
    std::string cuda_runtime_lib;
};

class GlimmerModel;

struct GlimmerLayerCache {
    bool local = false;
    std::int64_t logical_length = 0;
    std::vector<float> keys;
    std::vector<float> values;
};

struct GlimmerCacheStorage {
    std::vector<GlimmerLayerCache> layers;
    std::vector<float> final_hidden;
    std::shared_ptr<neuralfn::resident_glimmer_cuda::Cache> cuda;
};

struct SpeculativeStepResult {
    std::vector<DecodeResult> tokens;
    std::int64_t proposed_tokens = 0;
    std::int64_t accepted_tokens = 0;
    std::int64_t rejected_tokens = 0;
    std::int64_t target_rows = 0;
    std::int64_t assistant_blocks = 0;
    bool target_only_warmup = false;
};

class GlimmerSession final {
public:
    GlimmerSession(
        std::shared_ptr<GlimmerModel> model,
        std::int64_t seed,
        KVCacheMode cache_mode,
        std::shared_ptr<neuralfn::resident_glimmer_assistant::Model> assistant = nullptr);
    ~GlimmerSession();

    GlimmerSession(const GlimmerSession&) = delete;
    GlimmerSession& operator=(const GlimmerSession&) = delete;

    void prefill(const std::vector<std::int64_t>& token_ids, std::int64_t start_position);
    void prefill_with_embeddings(
        const std::vector<std::int64_t>& token_ids,
        std::int64_t start_position,
        const std::vector<std::int64_t>& replacement_positions,
        const std::vector<float>& replacement_embeddings);
    std::vector<float> current_logits();
    DecodeResult decode_one(const GenerationConfig& config);
    SpeculativeStepResult decode_speculative_block(
        const GenerationConfig& config,
        std::int64_t max_tokens_remaining);
    void truncate(std::int64_t token_count);
    void reset();
    void cancel() noexcept;
    void close() noexcept;
    SessionStats stats() const;
    std::shared_ptr<GlimmerSession> fork_prefix(
        std::int64_t token_count,
        std::int64_t seed);

    const std::shared_ptr<GlimmerModel>& model() const noexcept { return model_; }

private:
    void require_open() const;
    void rebuild_cache();
    void append_cached_token(std::int64_t token, std::int64_t position);
    void append_cached_embedding(
        std::int64_t token,
        std::int64_t position,
        const std::vector<float>& embedding);
    std::int64_t cache_bytes() const;

    std::shared_ptr<GlimmerModel> model_;
    mutable std::mutex mutex_;
    std::vector<std::int64_t> tokens_;
    std::unordered_map<std::int64_t, std::vector<float>> embedding_overrides_;
    const KVCacheMode cache_mode_;
    std::unique_ptr<GlimmerCacheStorage> cache_;
    std::shared_ptr<neuralfn::resident_glimmer_assistant::Model> assistant_model_;
    std::unique_ptr<neuralfn::resident_glimmer_assistant::Session> assistant_session_;
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
    std::int64_t speculative_blocks_ = 0;
    std::int64_t speculative_proposed_ = 0;
    std::int64_t speculative_accepted_ = 0;
    std::int64_t speculative_rejected_ = 0;
    bool speculative_ready_ = false;
    bool strict_model_compute_ = false;
};

class GlimmerModel final : public std::enable_shared_from_this<GlimmerModel> {
public:
    static std::shared_ptr<GlimmerModel> load(
        const std::string& checkpoint_path,
        GlimmerInferenceConfig config);

    GlimmerModel(const GlimmerModel&) = delete;
    GlimmerModel& operator=(const GlimmerModel&) = delete;
    ~GlimmerModel();

    std::shared_ptr<GlimmerSession> create_session(
        std::int64_t seed,
        KVCacheMode cache_mode);
    std::shared_ptr<GlimmerSession> create_speculative_session(
        std::int64_t seed,
        KVCacheMode cache_mode,
        std::shared_ptr<neuralfn::resident_glimmer_assistant::Model> assistant);
    void close() noexcept;
    bool closed() const noexcept { return closed_.load(); }
    ModelStats stats() const;

    std::unique_ptr<GlimmerCacheStorage> create_cache() const;
    void forward_append_token(
        std::int64_t token,
        std::int64_t position,
        GlimmerCacheStorage* cache,
        const std::atomic<bool>& cancelled,
        const std::vector<std::int64_t>* tap_layers = nullptr,
        std::vector<float>* target_taps = nullptr) const;
    void forward_append_embedding(
        const std::vector<float>& embedding,
        std::int64_t position,
        GlimmerCacheStorage* cache,
        const std::atomic<bool>& cancelled,
        const std::vector<std::int64_t>* tap_layers = nullptr,
        std::vector<float>* target_taps = nullptr) const;
    std::vector<float> forward_last_logits(
        const std::vector<std::int64_t>& tokens,
        const std::atomic<bool>& cancelled) const;
    std::vector<float> logits_from_hidden(const float* hidden) const;
    std::vector<float> raw_logits_from_hidden(const float* hidden) const;
    std::vector<float> raw_token_embedding(std::int64_t token) const;
    std::vector<float> encode_media(
        const std::vector<float>& packed_patches,
        const std::vector<std::int64_t>& grid_thw,
        const std::atomic<bool>& cancelled) const;
    std::vector<float> logits_from_cache(const GlimmerCacheStorage& cache) const;
    DecodeResult select_token(
        const std::vector<float>& logits,
        const GenerationConfig& config,
        std::mt19937_64& rng) const;

    std::int64_t max_seq_len() const noexcept { return config_.max_seq_len; }
    std::int64_t vocab_size() const noexcept { return config_.vocab_size; }
    std::int64_t model_dim() const noexcept { return config_.model_dim; }
    std::int64_t intermediate_dim() const noexcept { return config_.intermediate_dim; }
    std::int64_t kv_dim() const noexcept { return config_.num_kv_heads * config_.head_dim; }
    std::int64_t num_layers() const noexcept { return config_.num_layers; }
    std::int64_t sliding_window() const noexcept { return config_.sliding_window; }
    std::int64_t num_heads() const noexcept { return config_.num_heads; }
    std::int64_t num_kv_heads() const noexcept { return config_.num_kv_heads; }
    std::int64_t head_dim() const noexcept { return config_.head_dim; }
    double rope_theta() const noexcept { return config_.rope_theta; }
    double norm_eps() const noexcept { return config_.norm_eps; }
    double post_norm_eps() const noexcept { return config_.post_norm_eps; }
    double q_scale_factor() const noexcept { return config_.q_scale_factor; }
    double output_multiplier() const noexcept { return config_.output_multiplier; }
    double logit_softcap() const noexcept { return config_.logit_softcap; }
    WeightContainer weight_container() const noexcept { return config_.container; }
    const std::string& checkpoint_sha256() const noexcept { return config_.checkpoint_sha256; }
    bool is_local_layer(std::int64_t layer) const noexcept { return layer % 4 != 3; }
    bool whole_model_cuda() const noexcept { return static_cast<bool>(cuda_model_); }
    bool has_vision() const noexcept { return static_cast<bool>(vision_model_); }
    bool has_lora_adapter() const noexcept { return static_cast<bool>(adapter_mapped_); }
    void load_vision_companion(const std::string& checkpoint_path);
    void load_lora_adapter(
        const std::string& checkpoint_path,
        const std::string& checkpoint_sha256,
        std::int64_t rank,
        double alpha,
        std::uint32_t target_mask);
    std::int64_t vision_output_size() const noexcept;
    std::int64_t vision_weight_bytes() const noexcept;
    std::int64_t cuda_resident_weight_bytes() const noexcept;
    std::int64_t cuda_workspace_bytes() const noexcept;
    std::int64_t cuda_kernel_launches() const noexcept;
    int cuda_device() const noexcept;
    std::string cuda_tile_ops_library() const;
    std::string cuda_runtime_library() const;
    void session_opened() noexcept { open_sessions_.fetch_add(1); }
    void session_closed() noexcept { open_sessions_.fetch_sub(1); }

private:
    friend class GlimmerSession;

public:
    // Public only so the translation unit's allocation-free math helpers can
    // consume typed views. These remain an internal C++ ABI, not a Python API.
    class MappedFile;
    struct WeightView;
    struct LayerLayout;
    struct LoraWeight;
    struct LoraLayerLayout;

private:
    GlimmerModel(
        std::string checkpoint_path,
        GlimmerInferenceConfig config,
        std::unique_ptr<MappedFile> mapped);
    void require_open() const;
    void build_native_bf16_layout();
    void build_gguf_layout();
    void initialize_cuda_backend();
    void forward_append_input(
        std::int64_t token,
        const std::vector<float>* embedding,
        std::int64_t position,
        GlimmerCacheStorage* cache,
        const std::atomic<bool>& cancelled,
        const std::vector<std::int64_t>* tap_layers,
        std::vector<float>* target_taps) const;

    const std::string checkpoint_path_;
    const GlimmerInferenceConfig config_;
    std::unique_ptr<MappedFile> mapped_;
    std::unique_ptr<MappedFile> adapter_mapped_;
    std::unique_ptr<WeightView> token_embedding_;
    std::unique_ptr<WeightView> final_norm_;
    std::unique_ptr<WeightView> lm_head_;
    std::vector<LayerLayout> layers_;
    std::vector<LoraLayerLayout> lora_layers_;
    std::int64_t lora_parameter_count_ = 0;
    std::shared_ptr<neuralfn::resident_glimmer_vision::Model> vision_model_;
    std::shared_ptr<neuralfn::resident_glimmer_cuda::Model> cuda_model_;
    std::int64_t parameter_count_ = 0;
    std::atomic<bool> closed_{false};
    mutable std::shared_mutex lifecycle_mutex_;
    std::atomic<std::int64_t> open_sessions_{0};
    mutable std::atomic<std::int64_t> forward_calls_{0};
};

}  // namespace neuralfn::resident_glimmer
