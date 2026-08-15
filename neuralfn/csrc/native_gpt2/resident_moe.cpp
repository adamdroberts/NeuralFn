#include "resident_moe.h"

#include <stdexcept>
#include <utility>

namespace neuralfn::resident_moe {

MoeModel::MoeModel(
    std::shared_ptr<neuralfn::resident_llama::LlamaModel> implementation)
    : implementation_(std::move(implementation)) {}

std::shared_ptr<MoeModel> MoeModel::load(
    const std::string& checkpoint_path,
    MoeInferenceConfig config) {
    if (!config.standard_moe) {
        throw std::runtime_error("resident standard-MoE config must declare standard_moe=true");
    }
    return std::shared_ptr<MoeModel>(new MoeModel(
        neuralfn::resident_llama::LlamaModel::load(checkpoint_path, std::move(config))));
}

std::shared_ptr<MoeSession> MoeModel::create_session(
    std::int64_t seed,
    KVCacheMode cache_mode) {
    return std::make_shared<MoeSession>(
        shared_from_this(), implementation_->create_session(seed, cache_mode));
}

void MoeModel::close() noexcept { implementation_->close(); }
bool MoeModel::closed() const noexcept { return implementation_->closed(); }
ModelStats MoeModel::stats() const { return implementation_->stats(); }
std::int64_t MoeModel::hidden_dim() const noexcept { return implementation_->hidden_dim(); }
std::int64_t MoeModel::num_kv_heads() const noexcept { return implementation_->num_kv_heads(); }
std::int64_t MoeModel::head_dim() const noexcept { return implementation_->head_dim(); }
std::int64_t MoeModel::experts() const noexcept { return implementation_->experts(); }
std::int64_t MoeModel::top_k() const noexcept { return implementation_->top_k(); }
double MoeModel::rope_theta() const noexcept { return implementation_->rope_theta(); }
double MoeModel::rms_norm_eps() const noexcept { return implementation_->rms_norm_eps(); }
double MoeModel::mlp_multiplier() const noexcept { return implementation_->mlp_multiplier(); }
std::int64_t MoeModel::multiple_of() const noexcept { return implementation_->multiple_of(); }
double MoeModel::router_aux_loss_coef() const noexcept {
    return implementation_->router_aux_loss_coef();
}

MoeSession::MoeSession(
    std::shared_ptr<MoeModel> model,
    std::shared_ptr<neuralfn::resident_llama::LlamaSession> implementation)
    : model_(std::move(model)), implementation_(std::move(implementation)) {}

void MoeSession::prefill(
    const std::vector<std::int64_t>& token_ids,
    std::int64_t start_position) {
    implementation_->prefill(token_ids, start_position);
}

std::vector<float> MoeSession::current_logits() { return implementation_->current_logits(); }
DecodeResult MoeSession::decode_one(const GenerationConfig& config) {
    return implementation_->decode_one(config);
}
void MoeSession::truncate(std::int64_t token_count) { implementation_->truncate(token_count); }
void MoeSession::reset() { implementation_->reset(); }
void MoeSession::cancel() noexcept { implementation_->cancel(); }
void MoeSession::close() noexcept { implementation_->close(); }
SessionStats MoeSession::stats() const { return implementation_->stats(); }
std::shared_ptr<MoeSession> MoeSession::fork_prefix(
    std::int64_t token_count,
    std::int64_t seed) {
    return std::make_shared<MoeSession>(
        model_, implementation_->fork_prefix(token_count, seed));
}

}  // namespace neuralfn::resident_moe
