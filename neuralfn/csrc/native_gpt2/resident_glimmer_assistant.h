#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace neuralfn::resident_glimmer {
class GlimmerModel;
}

namespace neuralfn::resident_glimmer_cuda {
class DFlashModel;
class DFlashCache;
}

namespace neuralfn::resident_glimmer_assistant {

enum class WeightContainer {
    NativeBf16,
    GgufKQuant,
};

struct Config {
    std::int64_t max_seq_len = 131072;
    std::int64_t model_dim = 6656;
    std::int64_t intermediate_dim = 19968;
    std::int64_t num_layers = 5;
    std::int64_t num_heads = 32;
    std::int64_t num_kv_heads = 8;
    std::int64_t head_dim = 128;
    std::int64_t block_size = 16;
    std::int64_t mask_token_id = 201818;
    std::int64_t sliding_window = 2048;
    double rope_theta = 500000.0;
    double norm_eps = 1.0e-5;
    std::vector<std::int64_t> target_layer_ids = {1, 13, 25, 37, 49};
    WeightContainer container = WeightContainer::NativeBf16;
    std::string checkpoint_sha256;
    bool checkpoint_sha256_preverified = false;
};

struct Proposal {
    std::vector<std::int64_t> token_ids;
    // Row-major [proposal_tokens, target_vocab]. These are raw shared-head
    // logits, intentionally without the target multiplier or softcap. Greedy
    // CUDA proposal may leave this empty after device-side argmax.
    std::vector<float> logits;
};

class Model;

class Session final {
public:
    explicit Session(std::shared_ptr<Model> model);
    ~Session();

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    void reset();
    void record_target_taps(
        std::int64_t position,
        const std::vector<float>& concatenated_taps,
        const std::atomic<bool>& cancelled);
    void record_target_taps_batch(
        std::int64_t start_position,
        const float* concatenated_taps,
        std::int64_t rows,
        const std::atomic<bool>& cancelled);
    void record_target_taps_batch_device(
        std::int64_t start_position,
        const float* tap_major_device,
        int source_cuda_device,
        std::int64_t source_rows,
        std::int64_t rows,
        const std::atomic<bool>& cancelled);
    void record_target_taps_batch_device_and_prepare_lagged_anchor(
        std::int64_t start_position,
        const float* tap_major_device,
        int source_cuda_device,
        std::int64_t source_rows,
        std::int64_t rows,
        const std::atomic<bool>& cancelled);
    void prepare_lagged_anchor(
        std::int64_t anchor_position,
        const std::atomic<bool>& cancelled);
    Proposal propose(
        std::int64_t anchor_token,
        std::int64_t proposal_tokens,
        const std::atomic<bool>& cancelled,
        bool require_logits = true,
        bool fast_k_quant = false) const;

    std::int64_t context_length() const noexcept { return context_length_; }
    std::int64_t pending_position() const noexcept { return pending_position_; }
    std::int64_t cache_bytes() const noexcept;

private:
    struct LayerCache;
    void append_pending_context(const std::atomic<bool>& cancelled);

    std::shared_ptr<Model> model_;
    std::vector<LayerCache> layers_;
    std::shared_ptr<neuralfn::resident_glimmer_cuda::DFlashCache> cuda_cache_;
    std::vector<float> pending_taps_;
    std::int64_t pending_position_ = -1;
    std::int64_t lagged_anchor_position_ = -1;
    std::int64_t context_length_ = 0;
};

class Model final : public std::enable_shared_from_this<Model> {
public:
    static std::shared_ptr<Model> load(
        const std::string& checkpoint_path,
        Config config,
        std::shared_ptr<neuralfn::resident_glimmer::GlimmerModel> target);
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    std::unique_ptr<Session> create_session();
    void close() noexcept;
    bool closed() const noexcept { return closed_.load(); }

    const Config& config() const noexcept { return config_; }
    const std::vector<std::int64_t>& target_layer_ids() const noexcept {
        return config_.target_layer_ids;
    }
    std::int64_t target_tap_width() const noexcept {
        return config_.model_dim * static_cast<std::int64_t>(config_.target_layer_ids.size());
    }
    std::int64_t proposal_tokens() const noexcept { return config_.block_size - 1; }
    std::int64_t parameter_count() const noexcept { return parameter_count_; }
    std::int64_t weight_bytes() const noexcept { return weight_bytes_; }
    bool whole_model_cuda() const noexcept { return static_cast<bool>(cuda_model_); }
    bool cuda_device_tap_pack() const noexcept;
    std::int64_t cuda_resident_weight_bytes() const noexcept;
    std::int64_t cuda_workspace_bytes() const noexcept;
    std::int64_t cuda_kernel_launches() const noexcept;
    std::int64_t cuda_k_quant_mmq_linears() const noexcept;
    const std::shared_ptr<neuralfn::resident_glimmer::GlimmerModel>& target() const noexcept {
        return target_;
    }

public:
    // Internal typed views are public only so allocation-free translation-unit
    // helpers can consume them. They are not part of the Python/native ABI.
    friend class Session;
    class MappedFile;
    struct WeightView;
    struct Layer;

private:
    Model(
        std::string checkpoint_path,
        Config config,
        std::shared_ptr<neuralfn::resident_glimmer::GlimmerModel> target,
        std::unique_ptr<MappedFile> mapped);
    void require_open() const;
    void build_layout();
    void build_native_bf16_layout();
    void build_gguf_layout();
    void initialize_cuda_backend();

    const std::string checkpoint_path_;
    const Config config_;
    const std::shared_ptr<neuralfn::resident_glimmer::GlimmerModel> target_;
    std::unique_ptr<MappedFile> mapped_;
    std::vector<Layer> layers_;
    std::unique_ptr<WeightView> final_norm_;
    std::unique_ptr<WeightView> context_projection_;
    std::unique_ptr<WeightView> context_norm_;
    std::shared_ptr<neuralfn::resident_glimmer_cuda::DFlashModel> cuda_model_;
    std::int64_t parameter_count_ = 0;
    std::int64_t weight_bytes_ = 0;
    std::atomic<bool> closed_{false};
};

}  // namespace neuralfn::resident_glimmer_assistant
