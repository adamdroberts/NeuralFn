#include "resident_llama.h"
#include "resident_sha256.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace neuralfn::resident_llama {
namespace {

using neuralfn::resident_dense::ResidentCancellationError;

std::int64_t checked_add(std::int64_t left, std::int64_t right, const char* label) {
    if (left < 0 || right < 0 || left > std::numeric_limits<std::int64_t>::max() - right) {
        throw std::runtime_error(std::string("native LLaMA checkpoint size overflow at ") + label);
    }
    return left + right;
}

std::int64_t checked_mul(std::int64_t left, std::int64_t right, const char* label) {
    if (left < 0 || right < 0 ||
        (left != 0 && right > std::numeric_limits<std::int64_t>::max() / left)) {
        throw std::runtime_error(std::string("native LLaMA checkpoint size overflow at ") + label);
    }
    return left * right;
}

std::int64_t llama_parameter_count(const LlamaInferenceConfig& config) {
    const std::int64_t d = config.model_dim;
    const std::int64_t f = config.hidden_dim;
    const std::int64_t kv = checked_mul(config.num_kv_heads, config.head_dim, "KV width");
    std::int64_t total = checked_mul(
        checked_mul(2, config.padded_vocab_size, "embedding/head rows"),
        d,
        "embedding/head parameters");
    total = checked_add(total, d, "final RMSNorm");
    std::int64_t layer = checked_mul(2, d, "layer RMSNorms");
    layer = checked_add(
        layer,
        checked_mul(2, checked_mul(d, d, "model square"), "Q and output projections"),
        "dense projections");
    layer = checked_add(
        layer,
        checked_mul(2, checked_mul(kv, d, "KV projection"), "K and V projections"),
        "KV projections");
    if (config.standard_moe) {
        layer = checked_add(
            layer,
            checked_mul(config.experts, d, "router projection"),
            "router projection");
        layer = checked_add(
            layer,
            checked_mul(
                checked_mul(3, config.experts, "expert projections"),
                checked_mul(d, f, "expert projection"),
                "expert projections"),
            "expert projections");
    } else {
        layer = checked_add(
            layer,
            checked_mul(3, checked_mul(d, f, "FFN projection"), "SwiGLU projections"),
            "FFN projections");
    }
    return checked_add(
        total,
        checked_mul(config.num_layers, layer, "LLaMA layers"),
        "LLaMA parameters");
}

void validate_config(const LlamaInferenceConfig& config) {
    if (config.max_seq_len <= 0 || config.vocab_size <= 0 ||
        config.padded_vocab_size < config.vocab_size || config.num_layers <= 0 ||
        config.model_dim <= 0 || config.hidden_dim <= 0 || config.num_heads <= 0 ||
        config.num_kv_heads <= 0 || config.num_heads % config.num_kv_heads != 0 ||
        config.model_dim % config.num_heads != 0 ||
        config.head_dim != config.model_dim / config.num_heads ||
        config.head_dim <= 0 || config.head_dim % 2 != 0) {
        throw std::runtime_error("native LLaMA checkpoint has invalid GQA model geometry");
    }
    if (!std::isfinite(config.rope_theta) || !(config.rope_theta > 0.0) ||
        config.rope_scaling_factor != 1.0) {
        throw std::runtime_error(
            "native canonical LLaMA resident inference requires positive unscaled RoPE");
    }
    if (!std::isfinite(config.rms_norm_eps) || config.rms_norm_eps != 1.0e-6) {
        throw std::runtime_error(
            "native canonical LLaMA resident inference requires RMSNorm epsilon 1e-6");
    }
    if (config.standard_moe) {
        if (config.experts <= 0 || config.top_k <= 0 || config.top_k > config.experts ||
            !std::isfinite(config.mlp_multiplier) || !(config.mlp_multiplier > 0.0) ||
            config.multiple_of < 0 || !std::isfinite(config.router_aux_loss_coef) ||
            config.router_aux_loss_coef < 0.0) {
            throw std::runtime_error(
                "native standard-MoE checkpoint has invalid router or expert geometry");
        }
        const double raw_hidden = static_cast<double>(config.model_dim) * config.mlp_multiplier;
        if (!std::isfinite(raw_hidden) || raw_hidden < 0.0 ||
            raw_hidden > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
            throw std::runtime_error("native standard-MoE checkpoint hidden width overflows");
        }
        std::int64_t derived = std::max<std::int64_t>(
            1, static_cast<std::int64_t>(raw_hidden));
        if (config.multiple_of > 0) {
            if (derived > std::numeric_limits<std::int64_t>::max() -
                    (config.multiple_of - 1)) {
                throw std::runtime_error("native standard-MoE aligned hidden width overflows");
            }
            derived = checked_mul(
                (derived + config.multiple_of - 1) / config.multiple_of,
                config.multiple_of,
                "standard-MoE aligned hidden width");
        }
        if (derived != config.hidden_dim) {
            throw std::runtime_error(
                "native standard-MoE hidden width disagrees with multiplier/alignment metadata");
        }
    } else if (config.experts != 0 || config.top_k != 0) {
        throw std::runtime_error("native dense LLaMA checkpoint declares MoE routing geometry");
    }
    if (config.checkpoint_sha256.size() != 64 ||
        !std::all_of(
            config.checkpoint_sha256.begin(),
            config.checkpoint_sha256.end(),
            [](unsigned char character) {
                return (character >= '0' && character <= '9') ||
                    (character >= 'a' && character <= 'f');
            })) {
        throw std::runtime_error(
            "native LLaMA checkpoint requires a lowercase SHA-256 fingerprint");
    }
    (void)llama_parameter_count(config);
}

void throw_if_cancelled(const std::atomic<bool>& cancelled) {
    if (cancelled.load(std::memory_order_relaxed)) {
        throw ResidentCancellationError("resident inference session was cancelled");
    }
}

void rms_norm(
    const std::vector<float>& input,
    const float* weight,
    double epsilon,
    std::vector<float>* output,
    const std::atomic<bool>& cancelled) {
    throw_if_cancelled(cancelled);
    double square_sum = 0.0;
    for (float value : input) {
        square_sum += static_cast<double>(value) * value;
    }
    const double inverse = 1.0 / std::sqrt(
        square_sum / static_cast<double>(input.size()) + epsilon);
    output->resize(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        (*output)[index] = static_cast<float>(
            static_cast<double>(input[index]) * inverse * weight[index]);
    }
}

void linear(
    const std::vector<float>& input,
    const float* weight,
    std::int64_t output_dim,
    std::vector<float>* output,
    const std::atomic<bool>& cancelled) {
    const std::int64_t input_dim = static_cast<std::int64_t>(input.size());
    output->assign(static_cast<std::size_t>(output_dim), 0.0f);
    for (std::int64_t out = 0; out < output_dim; ++out) {
        if ((out & 31) == 0) {
            throw_if_cancelled(cancelled);
        }
        const float* row = weight + out * input_dim;
        double value = 0.0;
        for (std::int64_t in = 0; in < input_dim; ++in) {
            value += static_cast<double>(input[static_cast<std::size_t>(in)]) * row[in];
        }
        (*output)[static_cast<std::size_t>(out)] = static_cast<float>(value);
    }
}

void apply_rope(
    std::vector<float>* values,
    std::int64_t heads,
    std::int64_t head_dim,
    std::int64_t position,
    double theta,
    const std::atomic<bool>& cancelled) {
    const std::int64_t half = head_dim / 2;
    for (std::int64_t head = 0; head < heads; ++head) {
        throw_if_cancelled(cancelled);
        float* row = values->data() + head * head_dim;
        for (std::int64_t dim = 0; dim < half; ++dim) {
            const double inverse_frequency = 1.0 / std::pow(
                theta,
                static_cast<double>(2 * dim) / static_cast<double>(head_dim));
            const double angle = static_cast<double>(position) * inverse_frequency;
            const double cosine = std::cos(angle);
            const double sine = std::sin(angle);
            const float first = row[dim];
            const float second = row[dim + half];
            // Match the production Tile kernel's half-split sign convention.
            row[dim] = static_cast<float>(first * cosine + second * sine);
            row[dim + half] = static_cast<float>(-first * sine + second * cosine);
        }
    }
}

float silu(float value) {
    const double source = value;
    if (source >= 0.0) {
        return static_cast<float>(source / (1.0 + std::exp(-source)));
    }
    const double exponential = std::exp(source);
    return static_cast<float>(source * exponential / (1.0 + exponential));
}

bool contains_token(const std::vector<std::int64_t>& values, std::int64_t token) {
    return std::find(values.begin(), values.end(), token) != values.end();
}

}  // namespace

LlamaModel::LlamaModel(
    std::string checkpoint_path,
    LlamaInferenceConfig config,
    std::vector<float> weights)
    : checkpoint_path_(std::move(checkpoint_path)),
      config_(config),
      weights_(std::move(weights)) {
    std::int64_t offset = 0;
    token_embedding_ = offset;
    offset = checked_add(
        offset,
        checked_mul(config_.padded_vocab_size, config_.model_dim, "token embedding"),
        "token embedding");
    final_norm_ = offset;
    offset = checked_add(offset, config_.model_dim, "final norm");
    lm_head_ = offset;
    offset = checked_add(
        offset,
        checked_mul(config_.padded_vocab_size, config_.model_dim, "LM head"),
        "LM head");
    const std::int64_t kv_width = kv_dim();
    layers_.reserve(static_cast<std::size_t>(config_.num_layers));
    for (std::int64_t layer_index = 0; layer_index < config_.num_layers; ++layer_index) {
        LayerLayout layer;
        layer.attention_norm = offset;
        offset = checked_add(offset, config_.model_dim, "attention norm");
        layer.q_proj = offset;
        offset = checked_add(
            offset,
            checked_mul(config_.model_dim, config_.model_dim, "query projection"),
            "query projection");
        layer.k_proj = offset;
        offset = checked_add(
            offset,
            checked_mul(kv_width, config_.model_dim, "key projection"),
            "key projection");
        layer.v_proj = offset;
        offset = checked_add(
            offset,
            checked_mul(kv_width, config_.model_dim, "value projection"),
            "value projection");
        layer.attention_out = offset;
        offset = checked_add(
            offset,
            checked_mul(config_.model_dim, config_.model_dim, "attention output"),
            "attention output");
        layer.ffn_norm = offset;
        offset = checked_add(offset, config_.model_dim, "FFN norm");
        if (config_.standard_moe) {
            layer.router = offset;
            offset = checked_add(
                offset,
                checked_mul(config_.experts, config_.model_dim, "router projection"),
                "router projection");
            const std::int64_t expert_projection = checked_mul(
                checked_mul(config_.experts, config_.model_dim, "expert input rows"),
                config_.hidden_dim,
                "expert projection");
            layer.experts_gate = offset;
            offset = checked_add(offset, expert_projection, "expert gate");
            layer.experts_up = offset;
            offset = checked_add(offset, expert_projection, "expert up");
            layer.experts_down = offset;
            offset = checked_add(offset, expert_projection, "expert down");
        } else {
            layer.ffn_gate = offset;
            offset = checked_add(
                offset,
                checked_mul(config_.hidden_dim, config_.model_dim, "FFN gate"),
                "FFN gate");
            layer.ffn_up = offset;
            offset = checked_add(
                offset,
                checked_mul(config_.hidden_dim, config_.model_dim, "FFN up"),
                "FFN up");
            layer.ffn_down = offset;
            offset = checked_add(
                offset,
                checked_mul(config_.model_dim, config_.hidden_dim, "FFN down"),
                "FFN down");
        }
        layers_.push_back(layer);
    }
    if (offset != static_cast<std::int64_t>(weights_.size()) ||
        offset != llama_parameter_count(config_)) {
        throw std::runtime_error("native LLaMA checkpoint tensor layout has the wrong length");
    }
}

std::shared_ptr<LlamaModel> LlamaModel::load(
    const std::string& checkpoint_path,
    LlamaInferenceConfig config) {
    validate_config(config);
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error("native LLaMA float32 checkpoints require a little-endian host");
    }
    const std::int64_t elements = llama_parameter_count(config);
    const std::int64_t bytes = checked_mul(
        elements, static_cast<std::int64_t>(sizeof(float)), "checkpoint bytes");
    std::error_code filesystem_error;
    const auto actual_bytes = std::filesystem::file_size(checkpoint_path, filesystem_error);
    if (filesystem_error || actual_bytes != static_cast<std::uintmax_t>(bytes)) {
        throw std::runtime_error(
            "native LLaMA float32 checkpoint length does not match its declared layout");
    }
    std::ifstream input(checkpoint_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open native LLaMA float32 checkpoint: " + checkpoint_path);
    }
    std::vector<float> weights(static_cast<std::size_t>(elements), 0.0f);
    input.read(
        reinterpret_cast<char*>(weights.data()),
        static_cast<std::streamsize>(bytes));
    if (input.gcount() != static_cast<std::streamsize>(bytes)) {
        throw std::runtime_error("native LLaMA float32 checkpoint payload is truncated");
    }
    char trailing = 0;
    input.read(&trailing, 1);
    if (input.gcount() != 0 || !input.eof()) {
        throw std::runtime_error(
            "native LLaMA float32 checkpoint changed length while it was loaded");
    }
    const std::string loaded_sha256 = neuralfn::resident_support::sha256_hex(
        reinterpret_cast<const std::uint8_t*>(weights.data()),
        static_cast<std::size_t>(bytes));
    if (loaded_sha256 != config.checkpoint_sha256) {
        throw std::runtime_error(
            "native LLaMA loaded bytes do not match the manifest SHA-256 fingerprint");
    }
    if (std::any_of(weights.begin(), weights.end(), [](float value) {
            return !std::isfinite(value);
        })) {
        throw std::runtime_error("native LLaMA float32 checkpoint contains non-finite weights");
    }
    return std::shared_ptr<LlamaModel>(new LlamaModel(
        checkpoint_path, config, std::move(weights)));
}

const float* LlamaModel::at(std::int64_t offset) const {
    return weights_.data() + offset;
}

void LlamaModel::require_open() const {
    if (closed()) {
        throw std::runtime_error("resident inference model is closed");
    }
}

std::shared_ptr<LlamaSession> LlamaModel::create_session(
    std::int64_t seed,
    KVCacheMode cache_mode) {
    std::shared_lock<std::shared_mutex> lock(lifecycle_mutex_);
    require_open();
    if (cache_mode == KVCacheMode::TurboQuant) {
        throw std::runtime_error(
            "canonical LLaMA resident inference has not proved TurboQuant GQA storage");
    }
    return std::make_shared<LlamaSession>(shared_from_this(), seed, cache_mode);
}

void LlamaModel::close() noexcept {
    std::unique_lock<std::shared_mutex> lock(lifecycle_mutex_);
    closed_.store(true);
}

ModelStats LlamaModel::stats() const {
    ModelStats result;
    result.checkpoint_path = checkpoint_path_;
    result.max_seq_len = config_.max_seq_len;
    result.vocab_size = config_.vocab_size;
    result.padded_vocab_size = config_.padded_vocab_size;
    result.num_layers = config_.num_layers;
    result.num_heads = config_.num_heads;
    result.channels = config_.model_dim;
    result.parameter_count = static_cast<std::int64_t>(weights_.size());
    result.weight_bytes = checked_mul(
        result.parameter_count, static_cast<std::int64_t>(sizeof(float)), "weight bytes");
    result.weights_load_count = 1;
    result.open_sessions = open_sessions_.load();
    result.forward_calls = forward_calls_.load();
    result.turboquant_table_load_count = 0;
    result.use_qk_norm = false;
    result.qk_norm_eps = 0.0;
    result.logit_softcap = 0.0;
    return result;
}

void LlamaModel::forward_append_token(
    std::int64_t token,
    std::int64_t position,
    std::vector<std::vector<float>>* key_cache,
    std::vector<std::vector<float>>* value_cache,
    std::vector<float>* final_hidden_cache,
    const std::atomic<bool>& cancelled) const {
    require_open();
    if (token < 0 || token >= config_.vocab_size) {
        throw std::runtime_error("resident LLaMA token id is outside the checkpoint vocabulary");
    }
    if (position < 0 || position >= config_.max_seq_len) {
        throw std::runtime_error("resident LLaMA cache position exceeds the checkpoint context window");
    }
    if (key_cache == nullptr || value_cache == nullptr || final_hidden_cache == nullptr ||
        key_cache->size() != static_cast<std::size_t>(config_.num_layers) ||
        value_cache->size() != static_cast<std::size_t>(config_.num_layers)) {
        throw std::runtime_error("resident LLaMA lossless cache has invalid layer storage");
    }
    const std::int64_t kv_width = kv_dim();
    const std::int64_t required_kv = checked_mul(position + 1, kv_width, "logical KV cache");
    const std::int64_t required_hidden = checked_mul(
        position + 1, config_.model_dim, "logical final hidden cache");
    for (std::int64_t layer = 0; layer < config_.num_layers; ++layer) {
        if ((*key_cache)[static_cast<std::size_t>(layer)].size() <
                static_cast<std::size_t>(required_kv) ||
            (*value_cache)[static_cast<std::size_t>(layer)].size() <
                static_cast<std::size_t>(required_kv)) {
            throw std::runtime_error("resident LLaMA lossless KV cache capacity is too small");
        }
    }
    if (final_hidden_cache->size() < static_cast<std::size_t>(required_hidden)) {
        throw std::runtime_error("resident LLaMA final-hidden cache capacity is too small");
    }

    throw_if_cancelled(cancelled);
    forward_calls_.fetch_add(1);
    std::vector<float> hidden(static_cast<std::size_t>(config_.model_dim), 0.0f);
    const float* embedding = at(token_embedding_) + token * config_.model_dim;
    std::copy(embedding, embedding + config_.model_dim, hidden.begin());
    std::vector<std::vector<float>> staged_keys(
        static_cast<std::size_t>(config_.num_layers),
        std::vector<float>(static_cast<std::size_t>(kv_width), 0.0f));
    std::vector<std::vector<float>> staged_values(
        static_cast<std::size_t>(config_.num_layers),
        std::vector<float>(static_cast<std::size_t>(kv_width), 0.0f));

    std::vector<float> normalized;
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    std::vector<float> attention(static_cast<std::size_t>(config_.model_dim), 0.0f);
    std::vector<float> projected;
    std::vector<float> residual;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> activated;
    std::vector<float> down;
    std::vector<float> router_logits;
    std::vector<std::int64_t> selected_experts;
    std::vector<double> selected_weights;
    std::vector<double> scores(static_cast<std::size_t>(position + 1), 0.0);
    const double attention_scale = 1.0 / std::sqrt(static_cast<double>(config_.head_dim));

    for (std::int64_t layer_index = 0; layer_index < config_.num_layers; ++layer_index) {
        throw_if_cancelled(cancelled);
        const LayerLayout& layout = layers_[static_cast<std::size_t>(layer_index)];
        rms_norm(
            hidden,
            at(layout.attention_norm),
            config_.rms_norm_eps,
            &normalized,
            cancelled);
        linear(normalized, at(layout.q_proj), config_.model_dim, &query, cancelled);
        linear(normalized, at(layout.k_proj), kv_width, &key, cancelled);
        linear(normalized, at(layout.v_proj), kv_width, &value, cancelled);
        apply_rope(
            &query,
            config_.num_heads,
            config_.head_dim,
            position,
            config_.rope_theta,
            cancelled);
        apply_rope(
            &key,
            config_.num_kv_heads,
            config_.head_dim,
            position,
            config_.rope_theta,
            cancelled);
        staged_keys[static_cast<std::size_t>(layer_index)] = key;
        staged_values[static_cast<std::size_t>(layer_index)] = value;
        std::fill(attention.begin(), attention.end(), 0.0f);
        const auto& layer_keys = (*key_cache)[static_cast<std::size_t>(layer_index)];
        const auto& layer_values = (*value_cache)[static_cast<std::size_t>(layer_index)];
        for (std::int64_t query_head = 0; query_head < config_.num_heads; ++query_head) {
            throw_if_cancelled(cancelled);
            const std::int64_t kv_head =
                query_head * config_.num_kv_heads / config_.num_heads;
            const float* query_row = query.data() + query_head * config_.head_dim;
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::int64_t key_position = 0; key_position <= position; ++key_position) {
                const float* key_row = key_position == position
                    ? key.data() + kv_head * config_.head_dim
                    : layer_keys.data() + key_position * kv_width +
                        kv_head * config_.head_dim;
                double score = 0.0;
                for (std::int64_t dim = 0; dim < config_.head_dim; ++dim) {
                    score += static_cast<double>(query_row[dim]) * key_row[dim];
                }
                score *= attention_scale;
                scores[static_cast<std::size_t>(key_position)] = score;
                maximum = std::max(maximum, score);
            }
            double denominator = 0.0;
            for (std::int64_t key_position = 0; key_position <= position; ++key_position) {
                const double probability = std::exp(
                    scores[static_cast<std::size_t>(key_position)] - maximum);
                scores[static_cast<std::size_t>(key_position)] = probability;
                denominator += probability;
            }
            if (!(denominator > 0.0) || !std::isfinite(denominator)) {
                throw std::runtime_error("resident LLaMA attention probabilities are invalid");
            }
            for (std::int64_t dim = 0; dim < config_.head_dim; ++dim) {
                double accumulated = 0.0;
                for (std::int64_t key_position = 0; key_position <= position; ++key_position) {
                    const float* value_row = key_position == position
                        ? value.data() + kv_head * config_.head_dim
                        : layer_values.data() + key_position * kv_width +
                            kv_head * config_.head_dim;
                    accumulated +=
                        scores[static_cast<std::size_t>(key_position)] /
                        denominator * value_row[dim];
                }
                attention[static_cast<std::size_t>(
                    query_head * config_.head_dim + dim)] = static_cast<float>(accumulated);
            }
        }
        linear(
            attention,
            at(layout.attention_out),
            config_.model_dim,
            &projected,
            cancelled);
        residual.resize(static_cast<std::size_t>(config_.model_dim));
        for (std::int64_t dim = 0; dim < config_.model_dim; ++dim) {
            residual[static_cast<std::size_t>(dim)] =
                hidden[static_cast<std::size_t>(dim)] +
                projected[static_cast<std::size_t>(dim)];
        }
        rms_norm(
            residual,
            at(layout.ffn_norm),
            config_.rms_norm_eps,
            &normalized,
            cancelled);
        if (config_.standard_moe) {
            // Router weights are [E,D].  The graph computes softmax over all
            // experts, takes top-k, then renormalises the selected values.  The
            // common softmax denominator cancels, so a selected-logit softmax
            // is byte-for-byte the same routing equation without materialising
            // probabilities for unselected experts.
            linear(normalized, at(layout.router), config_.experts, &router_logits, cancelled);
            selected_experts.resize(static_cast<std::size_t>(config_.experts));
            std::iota(selected_experts.begin(), selected_experts.end(), 0);
            std::partial_sort(
                selected_experts.begin(),
                selected_experts.begin() + config_.top_k,
                selected_experts.end(),
                [&](std::int64_t left, std::int64_t right) {
                    const float lhs = router_logits[static_cast<std::size_t>(left)];
                    const float rhs = router_logits[static_cast<std::size_t>(right)];
                    return lhs == rhs ? left < right : lhs > rhs;
                });
            selected_experts.resize(static_cast<std::size_t>(config_.top_k));
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::int64_t expert : selected_experts) {
                maximum = std::max(
                    maximum,
                    static_cast<double>(router_logits[static_cast<std::size_t>(expert)]));
            }
            selected_weights.resize(static_cast<std::size_t>(config_.top_k));
            double denominator = 0.0;
            for (std::int64_t route = 0; route < config_.top_k; ++route) {
                const double value = std::exp(
                    static_cast<double>(router_logits[static_cast<std::size_t>(
                        selected_experts[static_cast<std::size_t>(route)])]) - maximum);
                selected_weights[static_cast<std::size_t>(route)] = value;
                denominator += value;
            }
            if (!(denominator > 0.0) || !std::isfinite(denominator)) {
                throw std::runtime_error(
                    "resident standard-MoE routing probabilities are invalid");
            }
            down.assign(static_cast<std::size_t>(config_.model_dim), 0.0f);
            const std::int64_t expert_stride = checked_mul(
                config_.model_dim, config_.hidden_dim, "expert stride");
            for (std::int64_t route = 0; route < config_.top_k; ++route) {
                throw_if_cancelled(cancelled);
                const std::int64_t expert =
                    selected_experts[static_cast<std::size_t>(route)];
                const float* gate_weight = at(layout.experts_gate) + expert * expert_stride;
                const float* up_weight = at(layout.experts_up) + expert * expert_stride;
                gate.assign(static_cast<std::size_t>(config_.hidden_dim), 0.0f);
                up.assign(static_cast<std::size_t>(config_.hidden_dim), 0.0f);
                for (std::int64_t input_dim = 0; input_dim < config_.model_dim; ++input_dim) {
                    const double input = normalized[static_cast<std::size_t>(input_dim)];
                    const float* gate_row = gate_weight + input_dim * config_.hidden_dim;
                    const float* up_row = up_weight + input_dim * config_.hidden_dim;
                    for (std::int64_t hidden_dim = 0; hidden_dim < config_.hidden_dim; ++hidden_dim) {
                        gate[static_cast<std::size_t>(hidden_dim)] += static_cast<float>(
                            input * gate_row[hidden_dim]);
                        up[static_cast<std::size_t>(hidden_dim)] += static_cast<float>(
                            input * up_row[hidden_dim]);
                    }
                }
                activated.resize(static_cast<std::size_t>(config_.hidden_dim));
                for (std::int64_t hidden_dim = 0; hidden_dim < config_.hidden_dim; ++hidden_dim) {
                    activated[static_cast<std::size_t>(hidden_dim)] =
                        silu(gate[static_cast<std::size_t>(hidden_dim)]) *
                        up[static_cast<std::size_t>(hidden_dim)];
                }
                const float* down_weight = at(layout.experts_down) + expert * expert_stride;
                const double route_weight =
                    selected_weights[static_cast<std::size_t>(route)] / denominator;
                for (std::int64_t output_dim = 0; output_dim < config_.model_dim; ++output_dim) {
                    double value = 0.0;
                    for (std::int64_t hidden_dim = 0; hidden_dim < config_.hidden_dim; ++hidden_dim) {
                        value += static_cast<double>(
                            activated[static_cast<std::size_t>(hidden_dim)]) *
                            down_weight[hidden_dim * config_.model_dim + output_dim];
                    }
                    down[static_cast<std::size_t>(output_dim)] += static_cast<float>(
                        route_weight * value);
                }
            }
        } else {
            linear(normalized, at(layout.ffn_gate), config_.hidden_dim, &gate, cancelled);
            linear(normalized, at(layout.ffn_up), config_.hidden_dim, &up, cancelled);
            activated.resize(static_cast<std::size_t>(config_.hidden_dim));
            for (std::int64_t dim = 0; dim < config_.hidden_dim; ++dim) {
                activated[static_cast<std::size_t>(dim)] =
                    silu(gate[static_cast<std::size_t>(dim)]) *
                    up[static_cast<std::size_t>(dim)];
            }
            linear(activated, at(layout.ffn_down), config_.model_dim, &down, cancelled);
        }
        hidden.resize(static_cast<std::size_t>(config_.model_dim));
        for (std::int64_t dim = 0; dim < config_.model_dim; ++dim) {
            hidden[static_cast<std::size_t>(dim)] =
                residual[static_cast<std::size_t>(dim)] +
                down[static_cast<std::size_t>(dim)];
        }
    }

    rms_norm(hidden, at(final_norm_), config_.rms_norm_eps, &normalized, cancelled);
    throw_if_cancelled(cancelled);
    const std::size_t kv_offset = static_cast<std::size_t>(position * kv_width);
    for (std::int64_t layer_index = 0; layer_index < config_.num_layers; ++layer_index) {
        std::copy(
            staged_keys[static_cast<std::size_t>(layer_index)].begin(),
            staged_keys[static_cast<std::size_t>(layer_index)].end(),
            (*key_cache)[static_cast<std::size_t>(layer_index)].begin() + kv_offset);
        std::copy(
            staged_values[static_cast<std::size_t>(layer_index)].begin(),
            staged_values[static_cast<std::size_t>(layer_index)].end(),
            (*value_cache)[static_cast<std::size_t>(layer_index)].begin() + kv_offset);
    }
    const std::size_t hidden_offset = static_cast<std::size_t>(
        position * config_.model_dim);
    std::copy(
        normalized.begin(),
        normalized.end(),
        final_hidden_cache->begin() + hidden_offset);
    // Logical session state advances only after this final check. Any staged
    // row left behind by cancellation is overwritten before it can be read.
    throw_if_cancelled(cancelled);
}

std::vector<float> LlamaModel::forward_last_logits(
    const std::vector<std::int64_t>& tokens,
    const std::atomic<bool>& cancelled) const {
    require_open();
    if (tokens.empty()) {
        throw std::runtime_error("resident LLaMA decode requires at least one prompt token");
    }
    if (tokens.size() > static_cast<std::size_t>(config_.max_seq_len)) {
        throw std::runtime_error("resident LLaMA history exceeds the checkpoint context window");
    }
    const std::int64_t logical_rows = static_cast<std::int64_t>(tokens.size());
    const std::int64_t kv_values = checked_mul(logical_rows, kv_dim(), "recompute KV cache");
    std::vector<std::vector<float>> keys(
        static_cast<std::size_t>(config_.num_layers),
        std::vector<float>(static_cast<std::size_t>(kv_values), 0.0f));
    std::vector<std::vector<float>> values(
        static_cast<std::size_t>(config_.num_layers),
        std::vector<float>(static_cast<std::size_t>(kv_values), 0.0f));
    std::vector<float> final_hidden(
        static_cast<std::size_t>(logical_rows * config_.model_dim), 0.0f);
    for (std::int64_t position = 0; position < logical_rows; ++position) {
        forward_append_token(
            tokens[static_cast<std::size_t>(position)],
            position,
            &keys,
            &values,
            &final_hidden,
            cancelled);
    }
    return logits_from_hidden(
        final_hidden.data() + (logical_rows - 1) * config_.model_dim);
}

std::vector<float> LlamaModel::logits_from_hidden(const float* hidden) const {
    require_open();
    if (hidden == nullptr) {
        throw std::runtime_error("resident LLaMA final hidden state is null");
    }
    std::vector<float> logits(static_cast<std::size_t>(config_.vocab_size), 0.0f);
    const float* head = at(lm_head_);
    for (std::int64_t token = 0; token < config_.vocab_size; ++token) {
        const float* row = head + token * config_.model_dim;
        double value = 0.0;
        for (std::int64_t dim = 0; dim < config_.model_dim; ++dim) {
            value += static_cast<double>(hidden[dim]) * row[dim];
        }
        logits[static_cast<std::size_t>(token)] = static_cast<float>(value);
    }
    return logits;
}

DecodeResult LlamaModel::select_token(
    const std::vector<float>& logits,
    const GenerationConfig& config,
    std::mt19937_64& rng) const {
    if (!std::isfinite(config.temperature) || config.temperature < 0.0) {
        throw std::runtime_error("temperature must be finite and non-negative");
    }
    if (config.top_k < 0) {
        throw std::runtime_error("top_k must be non-negative");
    }
    if (!std::isfinite(config.top_p) || !(config.top_p > 0.0) || config.top_p > 1.0) {
        throw std::runtime_error("top_p must be finite and in the interval (0, 1]");
    }
    if (logits.size() != static_cast<std::size_t>(config_.vocab_size)) {
        throw std::runtime_error("resident LLaMA forward returned the wrong vocabulary width");
    }
    std::int64_t selected = 0;
    for (std::int64_t token = 1; token < config_.vocab_size; ++token) {
        if (logits[static_cast<std::size_t>(token)] >
            logits[static_cast<std::size_t>(selected)]) {
            selected = token;
        }
    }
    if (config.temperature > 0.0 && config.top_k != 1) {
        std::vector<std::int64_t> candidates(static_cast<std::size_t>(config_.vocab_size));
        std::iota(candidates.begin(), candidates.end(), 0);
        std::sort(candidates.begin(), candidates.end(), [&](std::int64_t left, std::int64_t right) {
            const float lhs = logits[static_cast<std::size_t>(left)];
            const float rhs = logits[static_cast<std::size_t>(right)];
            return lhs == rhs ? left < right : lhs > rhs;
        });
        if (config.top_k > 0 && config.top_k < static_cast<std::int64_t>(candidates.size())) {
            candidates.resize(static_cast<std::size_t>(config.top_k));
        }
        const double maximum = logits[static_cast<std::size_t>(candidates.front())];
        std::vector<double> weights;
        weights.reserve(candidates.size());
        double total = 0.0;
        for (std::int64_t token : candidates) {
            const double shifted =
                (static_cast<double>(logits[static_cast<std::size_t>(token)]) - maximum) /
                config.temperature;
            const double weight = std::exp(std::max(-745.0, shifted));
            weights.push_back(weight);
            total += weight;
        }
        if (!(total > 0.0) || !std::isfinite(total)) {
            throw std::runtime_error("resident LLaMA sampling probabilities are invalid");
        }
        double cumulative = 0.0;
        std::size_t retained = weights.size();
        for (std::size_t index = 0; index < weights.size(); ++index) {
            cumulative += weights[index] / total;
            if (cumulative >= config.top_p) {
                retained = index + 1;
                break;
            }
        }
        candidates.resize(retained);
        weights.resize(retained);
        std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
        selected = candidates[distribution(rng)];
    }
    DecodeResult result;
    result.token_id = selected;
    result.selected_logit = logits[static_cast<std::size_t>(selected)];
    if (contains_token(config.stop_token_ids, selected)) {
        result.finish_reason = "stop";
    }
    return result;
}

LlamaSession::LlamaSession(
    std::shared_ptr<LlamaModel> model,
    std::int64_t seed,
    KVCacheMode cache_mode)
    : model_(std::move(model)),
      cache_mode_(cache_mode),
      seed_(seed),
      rng_(static_cast<std::mt19937_64::result_type>(seed)) {
    if (cache_mode_ == KVCacheMode::TurboQuant) {
        throw std::runtime_error(
            "canonical LLaMA resident inference has not proved TurboQuant GQA storage");
    }
    tokens_.reserve(static_cast<std::size_t>(model_->max_seq_len()));
    if (cache_mode_ == KVCacheMode::Full) {
        full_cache_ = std::make_shared<FullCacheStorage>();
        const std::int64_t kv_values = checked_mul(
            model_->max_seq_len(), model_->kv_dim(), "resident LLaMA KV capacity");
        full_cache_->key_cache.assign(
            static_cast<std::size_t>(model_->config_.num_layers),
            std::vector<float>(static_cast<std::size_t>(kv_values), 0.0f));
        full_cache_->value_cache.assign(
            static_cast<std::size_t>(model_->config_.num_layers),
            std::vector<float>(static_cast<std::size_t>(kv_values), 0.0f));
        const std::int64_t hidden_values = checked_mul(
            model_->max_seq_len(), model_->model_dim(), "resident LLaMA hidden capacity");
        full_cache_->final_hidden_cache.assign(
            static_cast<std::size_t>(hidden_values), 0.0f);
    }
    model_->session_opened();
}

LlamaSession::LlamaSession(
    std::shared_ptr<LlamaModel> model,
    std::int64_t seed,
    std::vector<std::int64_t> tokens,
    std::int64_t cache_length,
    std::shared_ptr<FullCacheStorage> full_cache)
    : model_(std::move(model)),
      tokens_(std::move(tokens)),
      cache_mode_(KVCacheMode::Full),
      full_cache_(std::move(full_cache)),
      cache_length_(cache_length),
      seed_(seed),
      rng_(static_cast<std::mt19937_64::result_type>(seed)),
      prefix_cow_forked_from_tokens_(cache_length) {
    if (!full_cache_) {
        throw std::runtime_error(
            "resident LLaMA prefix fork requires complete full-cache storage");
    }
    tokens_.reserve(static_cast<std::size_t>(model_->max_seq_len()));
    model_->session_opened();
}

LlamaSession::~LlamaSession() {
    close();
}

void LlamaSession::require_open() const {
    if (closed_) {
        throw std::runtime_error("resident inference session is closed");
    }
    if (model_->closed()) {
        throw std::runtime_error("resident inference model is closed");
    }
}

std::int64_t LlamaSession::full_cache_capacity_bytes() const {
    const std::int64_t kv_vectors = checked_mul(
        checked_mul(model_->config_.num_layers, 2, "LLaMA K/V layers"),
        model_->kv_dim(),
        "LLaMA K/V row");
    const std::int64_t values_per_token = checked_add(
        kv_vectors, model_->model_dim(), "LLaMA K/V plus final hidden");
    const std::int64_t bytes_per_token = checked_mul(
        values_per_token,
        static_cast<std::int64_t>(sizeof(float)),
        "LLaMA cache bytes per token");
    return checked_mul(
        model_->max_seq_len(), bytes_per_token, "LLaMA cache capacity bytes");
}

void LlamaSession::detach_full_cache_before_write() {
    if (cache_mode_ != KVCacheMode::Full) {
        return;
    }
    if (!full_cache_) {
        throw std::runtime_error("resident LLaMA full cache storage is unavailable");
    }
    if (full_cache_.use_count() == 1) {
        return;
    }

    // The entire GQA K/V plus final-hidden allocation is one ownership unit.
    // Publish its private copy only after every component copied successfully.
    auto detached = std::make_shared<FullCacheStorage>(*full_cache_);
    full_cache_ = std::move(detached);
    ++prefix_cow_detach_count_;
    prefix_cow_detached_capacity_bytes_ = checked_add(
        prefix_cow_detached_capacity_bytes_,
        full_cache_capacity_bytes(),
        "resident LLaMA prefix COW detached bytes");
}

std::shared_ptr<LlamaSession> LlamaSession::fork_prefix(
    std::int64_t token_count,
    std::int64_t seed) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::shared_lock<std::shared_mutex> model_lock(model_->lifecycle_mutex_);
    require_open();
    if (cache_mode_ != KVCacheMode::Full) {
        throw std::runtime_error(
            "resident LLaMA prefix COW requires a lossless full-cache session");
    }
    if (cache_length_ != static_cast<std::int64_t>(tokens_.size())) {
        throw std::runtime_error("resident LLaMA cache length does not match session history");
    }
    if (token_count <= 0 || token_count > cache_length_) {
        throw std::runtime_error(
            "resident LLaMA prefix COW token_count must select a non-empty cached prefix");
    }
    if (!full_cache_) {
        throw std::runtime_error(
            "resident LLaMA prefix fork requires complete full-cache storage");
    }
    std::vector<std::int64_t> forked_tokens(
        tokens_.begin(), tokens_.begin() + static_cast<std::ptrdiff_t>(token_count));
    auto child = std::shared_ptr<LlamaSession>(new LlamaSession(
        model_, seed, std::move(forked_tokens), token_count, full_cache_));
    ++prefix_cow_forks_created_;
    return child;
}

void LlamaSession::prefill(
    const std::vector<std::int64_t>& token_ids,
    std::int64_t start_position) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::shared_lock<std::shared_mutex> model_lock(model_->lifecycle_mutex_);
    require_open();
    throw_if_cancelled(cancelled_);
    if (start_position != static_cast<std::int64_t>(tokens_.size())) {
        throw std::runtime_error("prefill start_position does not match native session history");
    }
    if (checked_add(
            start_position,
            static_cast<std::int64_t>(token_ids.size()),
            "prefill history") > model_->max_seq_len()) {
        throw std::runtime_error("prefill would exceed the checkpoint context window");
    }
    for (std::int64_t token : token_ids) {
        if (token < 0 || token >= model_->vocab_size()) {
            throw std::runtime_error("prefill token id is outside the checkpoint vocabulary");
        }
    }
    if (cache_mode_ == KVCacheMode::Full) {
        if (cache_length_ != start_position) {
            throw std::runtime_error("resident LLaMA cache length does not match session history");
        }
        const std::int64_t initial_length = cache_length_;
        const std::int64_t detach_count_before = prefix_cow_detach_count_;
        const std::int64_t detached_capacity_bytes_before =
            prefix_cow_detached_capacity_bytes_;
        std::shared_ptr<FullCacheStorage> shared_cache_before_write;
        try {
            if (!token_ids.empty()) {
                if (full_cache_.use_count() > 1) {
                    // Retain the original allocation until the whole operation
                    // commits. A concurrent sibling therefore still observes a
                    // shared owner and must detach before writing these rollback
                    // bytes.
                    shared_cache_before_write = full_cache_;
                }
                detach_full_cache_before_write();
            }
            for (std::int64_t token : token_ids) {
                model_->forward_append_token(
                    token,
                    cache_length_,
                    &full_cache_->key_cache,
                    &full_cache_->value_cache,
                    &full_cache_->final_hidden_cache,
                    cancelled_);
                tokens_.push_back(token);
                ++cache_length_;
            }
        } catch (...) {
            if (shared_cache_before_write) {
                full_cache_ = std::move(shared_cache_before_write);
                prefix_cow_detach_count_ = detach_count_before;
                prefix_cow_detached_capacity_bytes_ = detached_capacity_bytes_before;
            }
            tokens_.resize(static_cast<std::size_t>(initial_length));
            cache_length_ = initial_length;
            throw;
        }
    } else {
        tokens_.insert(tokens_.end(), token_ids.begin(), token_ids.end());
    }
    ++prefill_calls_;
    prefill_tokens_ += static_cast<std::int64_t>(token_ids.size());
}

std::vector<float> LlamaSession::current_logits() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::shared_lock<std::shared_mutex> model_lock(model_->lifecycle_mutex_);
    require_open();
    throw_if_cancelled(cancelled_);
    if (tokens_.empty()) {
        throw std::runtime_error("current_logits requires a non-empty prefilled token history");
    }
    if (cache_mode_ == KVCacheMode::Full) {
        if (cache_length_ != static_cast<std::int64_t>(tokens_.size())) {
            throw std::runtime_error("resident LLaMA cache length does not match session history");
        }
        return model_->logits_from_hidden(
            full_cache_->final_hidden_cache.data() +
            (cache_length_ - 1) * model_->model_dim());
    }
    return model_->forward_last_logits(tokens_, cancelled_);
}

DecodeResult LlamaSession::decode_one(const GenerationConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::shared_lock<std::shared_mutex> model_lock(model_->lifecycle_mutex_);
    require_open();
    throw_if_cancelled(cancelled_);
    if (!std::isfinite(config.temperature) || config.temperature < 0.0) {
        throw std::runtime_error("temperature must be finite and non-negative");
    }
    if (config.top_k < 0) {
        throw std::runtime_error("top_k must be non-negative");
    }
    if (!std::isfinite(config.top_p) || !(config.top_p > 0.0) || config.top_p > 1.0) {
        throw std::runtime_error("top_p must be finite and in the interval (0, 1]");
    }
    for (std::int64_t token : config.stop_token_ids) {
        if (token < 0 || token >= model_->vocab_size()) {
            throw std::runtime_error("stop_token_ids contains a token outside the checkpoint vocabulary");
        }
    }
    if (tokens_.empty()) {
        throw std::runtime_error("decode_one requires a non-empty prefilled token history");
    }
    if (static_cast<std::int64_t>(tokens_.size()) >= model_->max_seq_len()) {
        throw std::runtime_error("decode_one would exceed the checkpoint context window");
    }
    const std::mt19937_64 rng_before = rng_;
    const std::optional<std::int64_t> generation_seed_before = active_generation_seed_;
    const std::int64_t detach_count_before = prefix_cow_detach_count_;
    const std::int64_t detached_capacity_bytes_before =
        prefix_cow_detached_capacity_bytes_;
    std::shared_ptr<FullCacheStorage> shared_cache_before_write;
    try {
        if (config.seed.has_value() && config.seed != active_generation_seed_) {
            rng_.seed(static_cast<std::mt19937_64::result_type>(*config.seed));
            active_generation_seed_ = config.seed;
        }
        std::vector<float> logits;
        if (cache_mode_ == KVCacheMode::Full) {
            if (cache_length_ != static_cast<std::int64_t>(tokens_.size())) {
                throw std::runtime_error("resident LLaMA cache length does not match session history");
            }
            logits = model_->logits_from_hidden(
                full_cache_->final_hidden_cache.data() +
                (cache_length_ - 1) * model_->model_dim());
        } else {
            logits = model_->forward_last_logits(tokens_, cancelled_);
        }
        DecodeResult result = model_->select_token(logits, config, rng_);
        throw_if_cancelled(cancelled_);
        if (cache_mode_ == KVCacheMode::Full) {
            if (full_cache_.use_count() > 1) {
                // Keep the pre-call allocation alive and immutable until the
                // decoded token is fully committed.
                shared_cache_before_write = full_cache_;
            }
            detach_full_cache_before_write();
            model_->forward_append_token(
                result.token_id,
                cache_length_,
                &full_cache_->key_cache,
                &full_cache_->value_cache,
                &full_cache_->final_hidden_cache,
                cancelled_);
            ++cache_length_;
            ++decode_rows_processed_;
        } else {
            decode_rows_processed_ += static_cast<std::int64_t>(tokens_.size());
        }
        tokens_.push_back(result.token_id);
        strict_model_compute_ = config.temperature == 0.0;
        ++decode_calls_;
        return result;
    } catch (...) {
        if (shared_cache_before_write) {
            full_cache_ = std::move(shared_cache_before_write);
            prefix_cow_detach_count_ = detach_count_before;
            prefix_cow_detached_capacity_bytes_ = detached_capacity_bytes_before;
        }
        rng_ = rng_before;
        active_generation_seed_ = generation_seed_before;
        throw;
    }
}

void LlamaSession::truncate(std::int64_t token_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::shared_lock<std::shared_mutex> model_lock(model_->lifecycle_mutex_);
    require_open();
    if (token_count < 0 || token_count > static_cast<std::int64_t>(tokens_.size())) {
        throw std::runtime_error("truncate token_count is outside the native session history");
    }
    tokens_.resize(static_cast<std::size_t>(token_count));
    if (cache_mode_ == KVCacheMode::Full) {
        cache_length_ = token_count;
    }
    ++truncate_calls_;
}

void LlamaSession::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::shared_lock<std::shared_mutex> model_lock(model_->lifecycle_mutex_);
    require_open();
    tokens_.clear();
    cache_length_ = 0;
    active_generation_seed_.reset();
    rng_.seed(static_cast<std::mt19937_64::result_type>(seed_));
    cancelled_.store(false);
    strict_model_compute_ = false;
    ++reset_calls_;
}

void LlamaSession::cancel() noexcept {
    cancelled_.store(true);
}

void LlamaSession::close() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!closed_) {
        closed_ = true;
        cancelled_.store(true);
        tokens_.clear();
        cache_length_ = 0;
        full_cache_.reset();
        model_->session_closed();
    }
}

SessionStats LlamaSession::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    SessionStats result;
    result.token_count = static_cast<std::int64_t>(tokens_.size());
    result.prefill_calls = prefill_calls_;
    result.prefill_tokens = prefill_tokens_;
    result.decode_calls = decode_calls_;
    result.truncate_calls = truncate_calls_;
    result.reset_calls = reset_calls_;
    result.prefix_cow_forks_created = prefix_cow_forks_created_;
    result.prefix_cow_forked_from_tokens = prefix_cow_forked_from_tokens_;
    result.prefix_cow_detach_count = prefix_cow_detach_count_;
    result.prefix_cow_detached_capacity_bytes = prefix_cow_detached_capacity_bytes_;
    result.cache_mode = cache_mode_;
    result.cached_tokens = cache_mode_ == KVCacheMode::Full ? cache_length_ : 0;
    if (cache_mode_ == KVCacheMode::Full) {
        const std::int64_t kv_vectors = checked_mul(
            checked_mul(model_->config_.num_layers, 2, "LLaMA K/V layers"),
            model_->kv_dim(),
            "LLaMA K/V row");
        const std::int64_t values_per_token = checked_add(
            kv_vectors, model_->model_dim(), "LLaMA K/V plus final hidden");
        const std::int64_t bytes_per_token = checked_mul(
            values_per_token,
            static_cast<std::int64_t>(sizeof(float)),
            "LLaMA cache bytes per token");
        result.cache_bytes = checked_mul(
            cache_length_, bytes_per_token, "LLaMA logical cache bytes");
        result.uncompressed_cache_bytes = result.cache_bytes;
        result.cache_capacity_bytes = full_cache_capacity_bytes();
        if (!full_cache_) {
            throw std::runtime_error("resident LLaMA full cache storage is unavailable");
        }
        const std::int64_t storage_use_count = static_cast<std::int64_t>(
            full_cache_.use_count());
        result.prefix_cow_storage_use_count = storage_use_count;
        if (storage_use_count > 1) {
            result.prefix_cow_shared_cached_tokens = cache_length_;
            result.prefix_cow_shared_capacity_bytes = result.cache_capacity_bytes;
        }
    }
    result.decode_rows_processed = decode_rows_processed_;
    result.strict_model_compute = strict_model_compute_;
    result.lossy_cache = false;
    result.cancelled = cancelled_.load();
    result.closed = closed_;
    return result;
}

}  // namespace neuralfn::resident_llama
