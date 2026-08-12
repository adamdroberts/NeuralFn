#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "resident_llama.h"

namespace neuralfn::resident_moe {

using neuralfn::resident_dense::DecodeResult;
using neuralfn::resident_dense::GenerationConfig;
using neuralfn::resident_dense::KVCacheMode;
using neuralfn::resident_dense::ModelStats;
using neuralfn::resident_dense::SessionStats;
using MoeInferenceConfig = neuralfn::resident_llama::LlamaInferenceConfig;

class MoeModel;

class MoeSession final {
public:
    MoeSession(
        std::shared_ptr<MoeModel> model,
        std::shared_ptr<neuralfn::resident_llama::LlamaSession> implementation);

    void prefill(const std::vector<std::int64_t>& token_ids, std::int64_t start_position);
    std::vector<float> current_logits();
    DecodeResult decode_one(const GenerationConfig& config);
    void truncate(std::int64_t token_count);
    void reset();
    void cancel() noexcept;
    void close() noexcept;
    SessionStats stats() const;
    std::shared_ptr<MoeSession> fork_prefix(
        std::int64_t token_count,
        std::int64_t seed);

    const std::shared_ptr<MoeModel>& model() const noexcept { return model_; }

private:
    std::shared_ptr<MoeModel> model_;
    std::shared_ptr<neuralfn::resident_llama::LlamaSession> implementation_;
};

class MoeModel final : public std::enable_shared_from_this<MoeModel> {
public:
    static std::shared_ptr<MoeModel> load(
        const std::string& checkpoint_path,
        MoeInferenceConfig config);

    std::shared_ptr<MoeSession> create_session(
        std::int64_t seed,
        KVCacheMode cache_mode);
    void close() noexcept;
    bool closed() const noexcept;
    ModelStats stats() const;

    std::int64_t hidden_dim() const noexcept;
    std::int64_t num_kv_heads() const noexcept;
    std::int64_t head_dim() const noexcept;
    std::int64_t experts() const noexcept;
    std::int64_t top_k() const noexcept;
    double rope_theta() const noexcept;
    double rms_norm_eps() const noexcept;
    double mlp_multiplier() const noexcept;
    std::int64_t multiple_of() const noexcept;
    double router_aux_loss_coef() const noexcept;

private:
    explicit MoeModel(std::shared_ptr<neuralfn::resident_llama::LlamaModel> implementation);

    std::shared_ptr<neuralfn::resident_llama::LlamaModel> implementation_;
};

}  // namespace neuralfn::resident_moe
