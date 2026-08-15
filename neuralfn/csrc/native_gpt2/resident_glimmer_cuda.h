#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace neuralfn::resident_glimmer_cuda {

struct HostWeightView {
    const std::uint8_t* data = nullptr;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::int64_t row_stride_bytes = 0;
    std::int64_t nbytes = 0;
    std::uint32_t encoding = 0;
    bool centered = false;
};

struct HostLayerWeights {
    HostWeightView input_norm;
    HostWeightView post_attention_norm;
    HostWeightView pre_feedforward_norm;
    HostWeightView post_feedforward_norm;
    HostWeightView q;
    HostWeightView k;
    HostWeightView v;
    HostWeightView gate;
    HostWeightView output;
    HostWeightView mlp_gate;
    HostWeightView mlp_up;
    HostWeightView mlp_down;
    std::optional<HostWeightView> q_norm;
    std::optional<HostWeightView> k_norm;
};

struct HostWeightPlan {
    HostWeightView token_embedding;
    HostWeightView final_norm;
    HostWeightView lm_head;
    std::vector<HostLayerWeights> layers;
};

struct HostLoraWeight {
    HostWeightView a;
    HostWeightView b;
    float scaling = 1.0f;
};

struct HostLoraLayer {
    std::optional<HostLoraWeight> q;
    std::optional<HostLoraWeight> k;
    std::optional<HostLoraWeight> v;
    std::optional<HostLoraWeight> gate;
    std::optional<HostLoraWeight> output;
    std::optional<HostLoraWeight> mlp_gate;
    std::optional<HostLoraWeight> mlp_up;
    std::optional<HostLoraWeight> mlp_down;
};

struct HostLoraPlan {
    std::vector<HostLoraLayer> layers;
};

struct VisionHostLinear {
    HostWeightView weight;
    std::vector<float> bias;
};

struct VisionHostLayer {
    VisionHostLinear query;
    VisionHostLinear key;
    VisionHostLinear value;
    VisionHostLinear output;
    std::vector<float> norm1_weight;
    std::vector<float> norm1_bias;
    std::vector<float> norm2_weight;
    std::vector<float> norm2_bias;
    VisionHostLinear fc1;
    VisionHostLinear fc2;
};

struct VisionHostWeightPlan {
    HostWeightView patch;
    std::vector<float> position;
    std::vector<float> pre_norm_weight;
    std::vector<float> pre_norm_bias;
    std::vector<float> post_norm_weight;
    std::vector<float> post_norm_bias;
    std::vector<VisionHostLayer> layers;
    VisionHostLinear adapter_fc1;
    VisionHostLinear adapter_fc2;
    VisionHostLinear projection;
};

struct VisionConfig {
    std::int64_t hidden_size = 1536;
    std::int64_t intermediate_size = 8960;
    std::int64_t num_layers = 50;
    std::int64_t num_heads = 16;
    std::int64_t patch_width = 1176;
    std::int64_t merge_size = 2;
    std::int64_t position_side = 32;
    std::int64_t adapter_size = 4096;
    std::int64_t output_size = 6656;
    float rope_theta = 10000.0f;
    float norm_eps = 1.0e-5f;
    bool interleaved_rope = false;
};

struct Config {
    std::int64_t max_seq_len = 131072;
    std::int64_t vocab_size = 202048;
    std::int64_t num_layers = 52;
    std::int64_t model_dim = 6656;
    std::int64_t intermediate_dim = 19968;
    std::int64_t num_heads = 32;
    std::int64_t num_kv_heads = 2;
    std::int64_t head_dim = 128;
    std::int64_t sliding_window = 2048;
    float rope_theta = 500000.0f;
    float norm_eps = 1.0e-5f;
    float post_norm_eps = 1.0e-8f;
    float q_scale_factor = 3.87f;
    float output_multiplier = 0.19611613513818404f;
    float logit_softcap = 20.0f;
    bool gguf_interleaved = false;
    int cuda_device = 0;
    std::string tile_ops_lib;
    std::string cuda_runtime_lib;
};

class Cache;
class DFlashCache;
class Verification;

struct ArgmaxRows {
    std::vector<std::int64_t> indices;
    std::vector<float> values;
};

class Model final : public std::enable_shared_from_this<Model> {
public:
    static std::shared_ptr<Model> load(const Config& config, const HostWeightPlan& weights);
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    std::shared_ptr<Cache> create_cache() const;
    void load_lora_adapter(const HostLoraPlan& weights);
    void load_vision(
        const VisionConfig& config,
        const VisionHostWeightPlan& weights);
    std::vector<float> encode_vision(
        const std::vector<float>& packed_patches,
        const std::vector<std::int64_t>& grid_thw,
        const std::atomic<bool>& cancelled);
    bool has_vision() const noexcept;
    std::int64_t vision_weight_bytes() const noexcept;
    void append_token(
        std::int64_t token_id,
        std::int64_t position,
        const std::shared_ptr<Cache>& cache,
        const std::atomic<bool>& cancelled,
        const std::vector<std::int64_t>* tap_layers = nullptr,
        std::vector<float>* target_taps = nullptr,
        bool fast_k_quant = false);
    void append_embedding(
        const std::vector<float>& embedding,
        std::int64_t position,
        const std::shared_ptr<Cache>& cache,
        const std::atomic<bool>& cancelled,
        const std::vector<std::int64_t>* tap_layers = nullptr,
        std::vector<float>* target_taps = nullptr,
        bool fast_k_quant = false);
    std::vector<float> logits(const std::shared_ptr<Cache>& cache);
    ArgmaxRows argmax_logits(const std::shared_ptr<Cache>& cache);
    // Greedy target decode fast path. The first call captures LM-head argmax,
    // device-token embedding, all 52 decoder layers, and cache/final-hidden
    // commit as one CUDA graph; later calls replay it once per token.
    ArgmaxRows decode_argmax_and_append(
        const std::shared_ptr<Cache>& cache,
        const std::atomic<bool>& cancelled);
    std::vector<float> raw_logits(const float* hidden);
    std::vector<float> raw_logits_rows(
        const float* hidden,
        std::int64_t rows,
        bool fast_k_quant = false);
    ArgmaxRows raw_argmax_rows(
        const float* hidden,
        std::int64_t rows,
        bool fast_k_quant = false);
    // Internal same-device handoff used by the paired DFlash resident. The
    // source must remain valid until this call returns; rows are copied into
    // target-owned workspace before the shared LM head runs.
    ArgmaxRows raw_argmax_rows_device(
        const float* device_hidden,
        int source_cuda_device,
        std::int64_t rows,
        bool fast_k_quant = false);
    std::vector<float> raw_embedding(std::int64_t token_id);
    std::vector<float> raw_embeddings(
        const std::vector<std::int64_t>& token_ids);
    // Returns target-owned device workspace, valid until the next operation
    // that uses the target's raw batched hidden workspace.
    const float* raw_embeddings_device(
        const std::vector<std::int64_t>& token_ids);
    std::shared_ptr<Verification> verify_tokens(
        const std::vector<std::int64_t>& token_ids,
        std::int64_t position,
        const std::shared_ptr<Cache>& cache,
        const std::atomic<bool>& cancelled,
        const std::vector<std::int64_t>* tap_layers = nullptr,
        bool compute_logits = true,
        bool compute_argmax = false,
        bool fast_k_quant = false,
        bool copy_taps_to_host = true,
        bool exact_verifier_rows = false);
    void commit_verification(
        const std::shared_ptr<Cache>& cache,
        const std::shared_ptr<Verification>& verification,
        std::int64_t accepted_rows,
        bool synchronize = true);
    void synchronize() const;
    void close() noexcept;

    std::int64_t resident_weight_bytes() const noexcept;
    std::int64_t workspace_bytes() const noexcept;
    std::int64_t kernel_launches() const noexcept;
    std::int64_t k_quant_mmq_linears() const noexcept;
    std::int64_t q8_activation_quantizations() const noexcept;
    std::int64_t q8_packed_linears() const noexcept;
    std::int64_t device_argmax_calls() const noexcept;
    std::int64_t device_argmax_rows() const noexcept;
    bool has_device_argmax() const noexcept;
    bool has_decode_graphs() const noexcept;
    int cuda_device() const noexcept;
    const std::string& tile_ops_library() const noexcept;
    const std::string& cuda_runtime_library() const noexcept;

private:
    class Impl;
    explicit Model(std::unique_ptr<Impl> impl);
    void append_input_unlocked(
        std::int64_t token_id,
        const std::vector<float>* embedding,
        std::int64_t position,
        const std::shared_ptr<Cache>& cache,
        const std::atomic<bool>& cancelled,
        const std::vector<std::int64_t>* tap_layers,
        std::vector<float>* target_taps,
        bool fast_k_quant,
        const std::int64_t* device_token_id,
        const std::int64_t* device_position,
        bool synchronize,
        bool commit_logical_length);
    std::unique_ptr<Impl> impl_;
};

class Cache final {
public:
    ~Cache();
    Cache(const Cache&) = delete;
    Cache& operator=(const Cache&) = delete;

    std::int64_t logical_length() const noexcept;
    std::int64_t allocated_bytes() const noexcept;

private:
    friend class Model;
    class Impl;
    explicit Cache(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class Verification final {
public:
    ~Verification();
    Verification(const Verification&) = delete;
    Verification& operator=(const Verification&) = delete;

    std::int64_t rows() const noexcept;
    std::int64_t position() const noexcept;
    const std::vector<float>& logits() const noexcept;
    const std::vector<std::int64_t>& argmax_indices() const noexcept;
    const std::vector<float>& argmax_values() const noexcept;
    const std::vector<float>& target_taps() const noexcept;
    const float* device_target_taps() const noexcept;
    int cuda_device() const noexcept;

private:
    friend class Model;
    class Impl;
    explicit Verification(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

struct DFlashHostLayerWeights {
    HostWeightView input_norm;
    HostWeightView post_attention_norm;
    HostWeightView q;
    HostWeightView k;
    HostWeightView v;
    HostWeightView output;
    HostWeightView q_norm;
    HostWeightView k_norm;
    HostWeightView mlp_gate;
    HostWeightView mlp_up;
    HostWeightView mlp_down;
};

struct DFlashHostWeightPlan {
    HostWeightView context_projection;
    HostWeightView context_norm;
    HostWeightView final_norm;
    std::vector<DFlashHostLayerWeights> layers;
};

struct DFlashConfig {
    std::int64_t max_seq_len = 131072;
    std::int64_t model_dim = 6656;
    std::int64_t intermediate_dim = 19968;
    std::int64_t num_layers = 5;
    std::int64_t num_heads = 32;
    std::int64_t num_kv_heads = 8;
    std::int64_t head_dim = 128;
    std::int64_t block_size = 16;
    std::int64_t tap_count = 5;
    std::int64_t sliding_window = 2048;
    float rope_theta = 500000.0f;
    float norm_eps = 1.0e-5f;
    bool gguf_interleaved = false;
    int cuda_device = 0;
    std::string tile_ops_lib;
    std::string cuda_runtime_lib;
};

class DFlashModel final : public std::enable_shared_from_this<DFlashModel> {
public:
    static std::shared_ptr<DFlashModel> load(
        const DFlashConfig& config,
        const DFlashHostWeightPlan& weights);
    ~DFlashModel();

    DFlashModel(const DFlashModel&) = delete;
    DFlashModel& operator=(const DFlashModel&) = delete;

    std::shared_ptr<DFlashCache> create_cache() const;
    void append_context(
        const float* concatenated_target_taps,
        std::int64_t position,
        const std::shared_ptr<DFlashCache>& cache,
        const std::atomic<bool>& cancelled);
    void append_contexts(
        const float* concatenated_target_taps,
        std::int64_t rows,
        std::int64_t start_position,
        const std::shared_ptr<DFlashCache>& cache,
        const std::atomic<bool>& cancelled);
    std::vector<float> append_contexts_device_tap_major(
        const float* tap_major_device,
        int source_cuda_device,
        std::int64_t source_rows,
        std::int64_t rows,
        std::int64_t start_position,
        const std::shared_ptr<DFlashCache>& cache,
        const std::atomic<bool>& cancelled);
    void append_contexts_device_tap_major_all(
        const float* tap_major_device,
        int source_cuda_device,
        std::int64_t source_rows,
        std::int64_t rows,
        std::int64_t start_position,
        const std::shared_ptr<DFlashCache>& cache,
        const std::atomic<bool>& cancelled);
    bool has_device_tap_pack() const noexcept;
    // Returns final-normalized assistant rows on the host. All learned model
    // computation remains on the selected device; the host transfer is the
    // explicit seam to the target model's shared LM-head API.
    std::vector<float> forward_block(
        const float* raw_target_embeddings,
        std::int64_t rows,
        const std::shared_ptr<DFlashCache>& cache,
        const std::atomic<bool>& cancelled);
    // Same-device greedy seam. The input is copied device-to-device and the
    // returned assistant-owned normalized rows remain valid until the next
    // proposal block on this model.
    const float* forward_block_device(
        const float* raw_target_embeddings_device,
        int source_cuda_device,
        std::int64_t rows,
        const std::shared_ptr<DFlashCache>& cache,
        const std::atomic<bool>& cancelled);
    void close() noexcept;

    std::int64_t resident_weight_bytes() const noexcept;
    std::int64_t workspace_bytes() const noexcept;
    std::int64_t kernel_launches() const noexcept;
    std::int64_t k_quant_mmq_linears() const noexcept;

private:
    class Impl;
    explicit DFlashModel(std::unique_ptr<Impl> impl);
    void append_context_rows_locked(
        std::int64_t rows,
        std::int64_t start_position,
        const std::shared_ptr<DFlashCache>& cache,
        const std::atomic<bool>& cancelled);
    const float* forward_block_input(
        const float* raw_target_embeddings,
        bool input_is_device,
        int source_cuda_device,
        std::int64_t rows,
        const std::shared_ptr<DFlashCache>& cache,
        const std::atomic<bool>& cancelled,
        std::vector<float>* host_result);
    std::unique_ptr<Impl> impl_;
};

class DFlashCache final {
public:
    ~DFlashCache();
    DFlashCache(const DFlashCache&) = delete;
    DFlashCache& operator=(const DFlashCache&) = delete;

    std::int64_t logical_length() const noexcept;
    std::int64_t allocated_bytes() const noexcept;

private:
    friend class DFlashModel;
    class Impl;
    explicit DFlashCache(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace neuralfn::resident_glimmer_cuda
