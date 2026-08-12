#include "resident_dense.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace neuralfn::resident_dense {
namespace {

constexpr std::int32_t kCheckpointMagic = 20240326;
constexpr std::int32_t kCheckpointVersion = 5;
constexpr std::int64_t kCheckpointHeaderInts = 256;
constexpr std::int64_t kCheckpointHeaderBytes = kCheckpointHeaderInts * 4;
constexpr float kLayerNormEpsilon = 1.0e-5f;

std::int64_t checked_add(std::int64_t left, std::int64_t right, const char* label) {
    if (left < 0 || right < 0 || left > std::numeric_limits<std::int64_t>::max() - right) {
        throw std::runtime_error(std::string("native checkpoint size overflow at ") + label);
    }
    return left + right;
}

std::int64_t checked_mul(std::int64_t left, std::int64_t right, const char* label) {
    if (left < 0 || right < 0 || (left != 0 && right > std::numeric_limits<std::int64_t>::max() / left)) {
        throw std::runtime_error(std::string("native checkpoint size overflow at ") + label);
    }
    return left * right;
}

std::int64_t parameter_count(
    std::int64_t max_seq_len,
    std::int64_t padded_vocab_size,
    std::int64_t num_layers,
    std::int64_t channels) {
    const std::int64_t c2 = checked_mul(channels, channels, "channels squared");
    const std::int64_t layer_weights = checked_add(
        checked_add(
            checked_add(
                checked_add(checked_mul(3, c2, "QKV weight"), checked_mul(3, channels, "QKV bias"), "QKV"),
                checked_add(c2, channels, "attention projection"),
                "attention"),
            checked_add(checked_mul(4, c2, "MLP expansion"), checked_mul(4, channels, "MLP expansion bias"), "MLP expansion"),
            "block projections"),
        checked_add(checked_mul(4, c2, "MLP projection"), checked_mul(5, channels, "block vector parameters"), "MLP projection and norms"),
        "block");
    std::int64_t total = checked_mul(padded_vocab_size, channels, "token embedding");
    total = checked_add(total, checked_mul(max_seq_len, channels, "position embedding"), "embeddings");
    total = checked_add(total, checked_mul(num_layers, layer_weights, "transformer blocks"), "model blocks");
    return checked_add(total, checked_mul(2, channels, "final layer norm"), "model parameters");
}

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t raw = static_cast<std::uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

void throw_if_cancelled(const std::atomic<bool>& cancelled) {
    if (cancelled.load(std::memory_order_relaxed)) {
        throw ResidentCancellationError("resident inference session was cancelled");
    }
}

void layer_norm(
    const std::vector<float>& input,
    const float* weight,
    const float* bias,
    std::int64_t rows,
    std::int64_t channels,
    std::vector<float>* output,
    const std::atomic<bool>& cancelled) {
    output->resize(static_cast<std::size_t>(rows * channels));
    for (std::int64_t row = 0; row < rows; ++row) {
        if ((row & 31) == 0) {
            throw_if_cancelled(cancelled);
        }
        const float* source = input.data() + row * channels;
        double sum = 0.0;
        for (std::int64_t channel = 0; channel < channels; ++channel) {
            sum += source[channel];
        }
        const double mean = sum / static_cast<double>(channels);
        double variance_sum = 0.0;
        for (std::int64_t channel = 0; channel < channels; ++channel) {
            const double centered = static_cast<double>(source[channel]) - mean;
            variance_sum += centered * centered;
        }
        const double inverse_std = 1.0 / std::sqrt(variance_sum / static_cast<double>(channels) + kLayerNormEpsilon);
        float* target = output->data() + row * channels;
        for (std::int64_t channel = 0; channel < channels; ++channel) {
            const double normalized = (static_cast<double>(source[channel]) - mean) * inverse_std;
            target[channel] = static_cast<float>(normalized * weight[channel] + bias[channel]);
        }
    }
}

void linear(
    const std::vector<float>& input,
    const float* weight,
    const float* bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    std::vector<float>* output,
    const std::atomic<bool>& cancelled) {
    output->assign(static_cast<std::size_t>(rows * output_dim), 0.0f);
    for (std::int64_t row = 0; row < rows; ++row) {
        if ((row & 7) == 0) {
            throw_if_cancelled(cancelled);
        }
        const float* source = input.data() + row * input_dim;
        float* target = output->data() + row * output_dim;
        for (std::int64_t out = 0; out < output_dim; ++out) {
            const float* row_weight = weight + out * input_dim;
            double value = bias == nullptr ? 0.0 : static_cast<double>(bias[out]);
            for (std::int64_t in = 0; in < input_dim; ++in) {
                value += static_cast<double>(source[in]) * row_weight[in];
            }
            target[out] = static_cast<float>(value);
        }
    }
}

void normalize_query_key_heads(
    std::vector<float>* qkv,
    std::int64_t rows,
    std::int64_t channels,
    std::int64_t num_heads,
    double epsilon,
    const std::atomic<bool>& cancelled) {
    if (qkv == nullptr || rows <= 0 || channels <= 0 || num_heads <= 0 ||
        channels % num_heads != 0 ||
        qkv->size() != static_cast<std::size_t>(rows * channels * 3) ||
        !std::isfinite(epsilon) || !(epsilon > 0.0)) {
        throw std::runtime_error("resident QK normalization has invalid geometry or epsilon");
    }
    const std::int64_t head_dim = channels / num_heads;
    for (std::int64_t row = 0; row < rows; ++row) {
        throw_if_cancelled(cancelled);
        for (std::int64_t segment = 0; segment < 2; ++segment) {
            for (std::int64_t head = 0; head < num_heads; ++head) {
                float* values = qkv->data() + row * channels * 3 +
                    segment * channels + head * head_dim;
                double squared_sum = 0.0;
                for (std::int64_t dim = 0; dim < head_dim; ++dim) {
                    squared_sum += static_cast<double>(values[dim]) * values[dim];
                }
                const double inverse_rms = 1.0 / std::sqrt(
                    squared_sum / static_cast<double>(head_dim) + epsilon);
                for (std::int64_t dim = 0; dim < head_dim; ++dim) {
                    values[dim] = static_cast<float>(
                        static_cast<double>(values[dim]) * inverse_rms);
                }
            }
        }
    }
}

const char* mlp_activation_name(MlpActivation activation) {
    switch (activation) {
        case MlpActivation::GeluExact:
        case MlpActivation::GeluTanh:
            return "gelu";
        case MlpActivation::Relu:
            return "relu";
        case MlpActivation::Silu:
            return "silu";
        case MlpActivation::ReluSquared:
            return "relu2";
    }
    throw std::runtime_error("resident dense MLP activation is invalid");
}

void apply_mlp_activation(
    std::vector<float>* values,
    MlpActivation activation,
    const std::atomic<bool>& cancelled) {
    if (values == nullptr) {
        throw std::runtime_error("resident dense MLP activation output is null");
    }
    constexpr double kInverseSqrtTwo = 0.70710678118654752440;
    constexpr double kGeluScale = 0.7978845608028654;
    constexpr double kGeluCubic = 0.044715;
    for (std::size_t index = 0; index < values->size(); ++index) {
        if ((index & 4095u) == 0u) {
            throw_if_cancelled(cancelled);
        }
        const double source = (*values)[index];
        double result = 0.0;
        switch (activation) {
            case MlpActivation::GeluExact:
                result = 0.5 * source * (1.0 + std::erf(source * kInverseSqrtTwo));
                break;
            case MlpActivation::GeluTanh:
                result = 0.5 * source *
                    (1.0 + std::tanh(kGeluScale *
                        (source + kGeluCubic * source * source * source)));
                break;
            case MlpActivation::Relu:
                result = std::max(0.0, source);
                break;
            case MlpActivation::Silu:
                result = source / (1.0 + std::exp(-source));
                break;
            case MlpActivation::ReluSquared: {
                const double positive = std::max(0.0, source);
                result = positive * positive;
                break;
            }
        }
        if (!std::isfinite(result)) {
            throw std::runtime_error("resident dense MLP activation produced a non-finite value");
        }
        (*values)[index] = static_cast<float>(result);
    }
}

bool contains_token(const std::vector<std::int64_t>& values, std::int64_t token) {
    return std::find(values.begin(), values.end(), token) != values.end();
}

}  // namespace

DenseModel::DenseModel(
    std::string checkpoint_path,
    std::int64_t max_seq_len,
    std::int64_t vocab_size,
    std::int64_t num_layers,
    std::int64_t num_heads,
    std::int64_t channels,
    std::int64_t padded_vocab_size,
    DenseInferenceConfig inference_config,
    std::vector<float> weights)
    : checkpoint_path_(std::move(checkpoint_path)),
      max_seq_len_(max_seq_len),
      vocab_size_(vocab_size),
      num_layers_(num_layers),
      num_heads_(num_heads),
      channels_(channels),
      padded_vocab_size_(padded_vocab_size),
      inference_config_(inference_config),
      weights_(std::move(weights)) {
    std::int64_t offset = 0;
    auto take = [&](std::int64_t count) {
        const std::int64_t start = offset;
        offset = checked_add(offset, count, "resident layout");
        return start;
    };
    wte_weight_ = take(checked_mul(padded_vocab_size_, channels_, "token embedding layout"));
    wpe_weight_ = take(checked_mul(max_seq_len_, channels_, "position embedding layout"));
    blocks_.reserve(static_cast<std::size_t>(num_layers_));
    const std::int64_t c2 = checked_mul(channels_, channels_, "resident channels squared");
    for (std::int64_t layer = 0; layer < num_layers_; ++layer) {
        BlockLayout block;
        block.ln1_weight = take(channels_);
        block.ln1_bias = take(channels_);
        block.qkv_weight = take(checked_mul(3, c2, "resident QKV"));
        block.qkv_bias = take(checked_mul(3, channels_, "resident QKV bias"));
        block.attn_proj_weight = take(c2);
        block.attn_proj_bias = take(channels_);
        block.ln2_weight = take(channels_);
        block.ln2_bias = take(channels_);
        block.fc_weight = take(checked_mul(4, c2, "resident MLP expansion"));
        block.fc_bias = take(checked_mul(4, channels_, "resident MLP expansion bias"));
        block.mlp_proj_weight = take(checked_mul(4, c2, "resident MLP projection"));
        block.mlp_proj_bias = take(channels_);
        blocks_.push_back(block);
    }
    final_ln_weight_ = take(channels_);
    final_ln_bias_ = take(channels_);
    if (offset != static_cast<std::int64_t>(weights_.size())) {
        throw std::runtime_error("native checkpoint payload does not match the dense v5 tensor layout");
    }
}

std::shared_ptr<DenseModel> DenseModel::load(
    const std::string& checkpoint_path,
    DenseInferenceConfig inference_config) {
    if (!std::isfinite(inference_config.qk_norm_eps) ||
        !(inference_config.qk_norm_eps > 0.0) ||
        !std::isfinite(inference_config.logit_softcap) ||
        inference_config.logit_softcap < 0.0 ||
        (inference_config.moa_mode &&
            (inference_config.moa_interval <= 0 ||
             inference_config.mlp_activation == MlpActivation::GeluExact)) ||
        (!inference_config.moa_mode &&
            (inference_config.moa_interval != 0 ||
             inference_config.mlp_activation != MlpActivation::GeluExact))) {
        throw std::runtime_error("resident dense inference configuration is invalid");
    }
    std::ifstream input(checkpoint_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open native dense checkpoint: " + checkpoint_path);
    }
    std::vector<std::int32_t> header(static_cast<std::size_t>(kCheckpointHeaderInts), 0);
    input.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(kCheckpointHeaderBytes));
    if (input.gcount() != kCheckpointHeaderBytes) {
        throw std::runtime_error("native dense checkpoint header is truncated: " + checkpoint_path);
    }
    if (header[0] != kCheckpointMagic || header[1] != kCheckpointVersion) {
        throw std::runtime_error("resident dense inference requires a native bf16 checkpoint at version 5");
    }
    const std::int64_t max_seq_len = header[2];
    const std::int64_t vocab_size = header[3];
    const std::int64_t num_layers = header[4];
    const std::int64_t num_heads = header[5];
    const std::int64_t channels = header[6];
    const std::int64_t padded_vocab_size = header[7];
    if (max_seq_len <= 0 || vocab_size <= 0 || num_layers <= 0 || num_heads <= 0 ||
        channels <= 0 || padded_vocab_size < vocab_size || channels % num_heads != 0) {
        throw std::runtime_error("native dense checkpoint has invalid model geometry");
    }
    const std::int64_t count = parameter_count(max_seq_len, padded_vocab_size, num_layers, channels);
    const std::int64_t payload_bytes = checked_mul(count, 2, "bf16 checkpoint bytes");
    const std::int64_t expected_bytes = checked_add(kCheckpointHeaderBytes, payload_bytes, "checkpoint bytes");
    input.seekg(0, std::ios::end);
    const std::streamoff end = input.tellg();
    if (end < 0 || static_cast<std::int64_t>(end) != expected_bytes) {
        throw std::runtime_error("native dense checkpoint file size does not match its v5 geometry");
    }
    input.seekg(kCheckpointHeaderBytes, std::ios::beg);
    std::vector<std::uint16_t> bf16(static_cast<std::size_t>(count), 0);
    input.read(
        reinterpret_cast<char*>(bf16.data()),
        static_cast<std::streamsize>(payload_bytes));
    if (input.gcount() != payload_bytes) {
        throw std::runtime_error("native dense checkpoint payload is truncated: " + checkpoint_path);
    }
    std::vector<float> weights;
    weights.reserve(bf16.size());
    for (std::uint16_t value : bf16) {
        weights.push_back(bf16_to_float(value));
    }
    return std::shared_ptr<DenseModel>(new DenseModel(
        checkpoint_path,
        max_seq_len,
        vocab_size,
        num_layers,
        num_heads,
        channels,
        padded_vocab_size,
        inference_config,
        std::move(weights)));
}

const float* DenseModel::at(std::int64_t offset) const {
    return weights_.data() + offset;
}

void DenseModel::require_open() const {
    if (closed()) {
        throw std::runtime_error("resident inference model is closed");
    }
}

std::shared_ptr<DenseSession> DenseModel::create_session(
    std::int64_t seed,
    KVCacheMode cache_mode,
    std::optional<TurboQuantTables> turboquant_tables,
    bool tile_turboquant_attention) {
    std::shared_lock<std::shared_mutex> lock(lifecycle_mutex_);
    require_open();
    std::shared_ptr<const TurboQuantCodec> turboquant_codec;
    if (cache_mode == KVCacheMode::TurboQuant) {
        if (!turboquant_tables.has_value()) {
            throw std::runtime_error("TurboQuant resident cache requires codec tables");
        }
        turboquant_codec = resolve_turboquant_codec(std::move(*turboquant_tables));
    } else if (turboquant_tables.has_value()) {
        throw std::runtime_error("non-TurboQuant resident cache must not receive codec tables");
    }
    if (tile_turboquant_attention && cache_mode != KVCacheMode::TurboQuant) {
        throw std::runtime_error(
            "Tile-CUDA TurboQuant attention requires the TurboQuant resident cache");
    }
    std::unique_ptr<TileTurboQuantSession> tile_session;
    if (tile_turboquant_attention) {
        std::shared_ptr<TileTurboQuantModel> tile_model;
        {
            std::lock_guard<std::mutex> tile_lock(tile_turboquant_mutex_);
            tile_model = tile_turboquant_model_;
        }
        if (!tile_model) {
            throw std::runtime_error(
                "Tile-CUDA TurboQuant attention was requested before model configuration");
        }
        tile_session = tile_model->create_session(turboquant_codec);
    }
    return std::make_shared<DenseSession>(
        shared_from_this(),
        seed,
        cache_mode,
        std::move(turboquant_codec),
        std::move(tile_session));
}

TileTurboQuantModelStats DenseModel::configure_turboquant_attention(
    TileTurboQuantConfig config) {
    std::unique_lock<std::shared_mutex> lock(lifecycle_mutex_);
    require_open();
    if (open_sessions_.load(std::memory_order_relaxed) != 0) {
        throw std::runtime_error(
            "Tile-CUDA TurboQuant attention must be configured before creating sessions");
    }
    std::lock_guard<std::mutex> tile_lock(tile_turboquant_mutex_);
    if (tile_turboquant_model_) {
        throw std::runtime_error("Tile-CUDA TurboQuant attention is already configured");
    }
    std::shared_ptr<TileTurboQuantModel> configured = TileTurboQuantModel::configure(
        std::move(config), num_layers_, num_heads_, channels_, max_seq_len_);
    TileTurboQuantModelStats result = configured->stats();
    tile_turboquant_model_ = std::move(configured);
    return result;
}

std::shared_ptr<const TurboQuantCodec> DenseModel::resolve_turboquant_codec(
    TurboQuantTables tables) {
    auto candidate = std::make_shared<const TurboQuantCodec>(std::move(tables));
    std::lock_guard<std::mutex> lock(turboquant_table_mutex_);
    std::shared_ptr<const TurboQuantCodec>& slot =
        candidate->profile() == TurboQuantProfile::Qjl35
        ? qjl_turboquant_codec_
        : mse_turboquant_codec_;
    if (slot) {
        if (!slot->matches(candidate->tables())) {
            throw std::runtime_error(
                "TurboQuant codec tables changed after the model loaded this profile");
        }
        return slot;
    }
    slot = std::move(candidate);
    turboquant_table_load_count_.fetch_add(1);
    return slot;
}

void DenseModel::close() noexcept {
    std::unique_lock<std::shared_mutex> lock(lifecycle_mutex_);
    closed_.store(true);
}

ModelStats DenseModel::stats() const {
    ModelStats result;
    result.checkpoint_path = checkpoint_path_;
    result.max_seq_len = max_seq_len_;
    result.vocab_size = vocab_size_;
    result.padded_vocab_size = padded_vocab_size_;
    result.num_layers = num_layers_;
    result.num_heads = num_heads_;
    result.channels = channels_;
    result.parameter_count = static_cast<std::int64_t>(weights_.size());
    result.weight_bytes = static_cast<std::int64_t>(weights_.size() * sizeof(float));
    result.weights_load_count = 1;
    result.open_sessions = open_sessions_.load();
    result.forward_calls = forward_calls_.load();
    result.turboquant_table_load_count = turboquant_table_load_count_.load();
    {
        std::lock_guard<std::mutex> tile_lock(tile_turboquant_mutex_);
        if (tile_turboquant_model_) {
            const TileTurboQuantModelStats tile_stats = tile_turboquant_model_->stats();
            result.turboquant_tile_attention_configured = tile_stats.configured;
            result.turboquant_attention_backend = tile_stats.backend;
            result.turboquant_tile_ops_lib = tile_stats.tile_ops_lib;
            result.turboquant_cuda_runtime_lib = tile_stats.cuda_runtime_lib;
            result.turboquant_cuda_device = tile_stats.device;
        }
    }
    result.use_qk_norm = inference_config_.use_qk_norm;
    result.qk_norm_eps = inference_config_.qk_norm_eps;
    result.logit_softcap = inference_config_.logit_softcap;
    result.moa_mode = inference_config_.moa_mode;
    result.moa_interval = inference_config_.moa_interval;
    result.mlp_activation = mlp_activation_name(inference_config_.mlp_activation);
    return result;
}

std::vector<float> DenseModel::forward_last_logits(
    const std::vector<std::int64_t>& tokens,
    const std::atomic<bool>& cancelled) const {
    require_open();
    if (tokens.empty()) {
        throw std::runtime_error("resident dense decode requires at least one prompt token");
    }
    if (static_cast<std::int64_t>(tokens.size()) > max_seq_len_) {
        throw std::runtime_error("resident dense token history exceeds the checkpoint context window");
    }
    throw_if_cancelled(cancelled);
    forward_calls_.fetch_add(1);
    const std::int64_t rows = static_cast<std::int64_t>(tokens.size());
    const std::int64_t channels = channels_;
    const std::int64_t head_dim = channels_ / num_heads_;
    std::vector<float> hidden(static_cast<std::size_t>(rows * channels), 0.0f);
    const float* token_embedding = at(wte_weight_);
    const float* position_embedding = at(wpe_weight_);
    for (std::int64_t row = 0; row < rows; ++row) {
        const std::int64_t token = tokens[static_cast<std::size_t>(row)];
        if (token < 0 || token >= vocab_size_) {
            throw std::runtime_error("resident dense token id is outside the checkpoint vocabulary");
        }
        for (std::int64_t channel = 0; channel < channels; ++channel) {
            hidden[static_cast<std::size_t>(row * channels + channel)] =
                token_embedding[token * channels + channel] +
                position_embedding[row * channels + channel];
        }
    }

    std::vector<float> normalized;
    std::vector<float> qkv;
    std::vector<float> attention(static_cast<std::size_t>(rows * channels), 0.0f);
    std::vector<float> projected;
    std::vector<float> residual;
    std::vector<float> expanded;
    std::vector<float> mlp;
    std::vector<double> scores;
    const double attention_scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

    for (std::int64_t layer = 0; layer < num_layers_; ++layer) {
        throw_if_cancelled(cancelled);
        const BlockLayout& block = blocks_[static_cast<std::size_t>(layer)];
        layer_norm(hidden, at(block.ln1_weight), at(block.ln1_bias), rows, channels, &normalized, cancelled);
        linear(normalized, at(block.qkv_weight), at(block.qkv_bias), rows, channels, channels * 3, &qkv, cancelled);
        if (inference_config_.use_qk_norm) {
            normalize_query_key_heads(
                &qkv,
                rows,
                channels,
                num_heads_,
                inference_config_.qk_norm_eps,
                cancelled);
        }
        std::fill(attention.begin(), attention.end(), 0.0f);
        for (std::int64_t row = 0; row < rows; ++row) {
            if ((row & 7) == 0) {
                throw_if_cancelled(cancelled);
            }
            scores.resize(static_cast<std::size_t>(row + 1));
            for (std::int64_t head = 0; head < num_heads_; ++head) {
                double max_score = -std::numeric_limits<double>::infinity();
                const std::int64_t q_base = row * channels * 3 + head * head_dim;
                for (std::int64_t key_row = 0; key_row <= row; ++key_row) {
                    const std::int64_t k_base = key_row * channels * 3 + channels + head * head_dim;
                    double score = 0.0;
                    for (std::int64_t dim = 0; dim < head_dim; ++dim) {
                        score += static_cast<double>(qkv[static_cast<std::size_t>(q_base + dim)]) *
                            qkv[static_cast<std::size_t>(k_base + dim)];
                    }
                    score *= attention_scale;
                    scores[static_cast<std::size_t>(key_row)] = score;
                    max_score = std::max(max_score, score);
                }
                double denominator = 0.0;
                for (std::int64_t key_row = 0; key_row <= row; ++key_row) {
                    const double probability = std::exp(scores[static_cast<std::size_t>(key_row)] - max_score);
                    scores[static_cast<std::size_t>(key_row)] = probability;
                    denominator += probability;
                }
                for (std::int64_t dim = 0; dim < head_dim; ++dim) {
                    double value = 0.0;
                    for (std::int64_t key_row = 0; key_row <= row; ++key_row) {
                        const std::int64_t v_base = key_row * channels * 3 + channels * 2 + head * head_dim;
                        value += (scores[static_cast<std::size_t>(key_row)] / denominator) *
                            qkv[static_cast<std::size_t>(v_base + dim)];
                    }
                    attention[static_cast<std::size_t>(row * channels + head * head_dim + dim)] =
                        static_cast<float>(value);
                }
            }
        }
        linear(attention, at(block.attn_proj_weight), at(block.attn_proj_bias), rows, channels, channels, &projected, cancelled);
        residual.resize(hidden.size());
        for (std::size_t index = 0; index < hidden.size(); ++index) {
            residual[index] = hidden[index] + projected[index];
        }
        layer_norm(residual, at(block.ln2_weight), at(block.ln2_bias), rows, channels, &normalized, cancelled);
        linear(normalized, at(block.fc_weight), at(block.fc_bias), rows, channels, channels * 4, &expanded, cancelled);
        apply_mlp_activation(&expanded, inference_config_.mlp_activation, cancelled);
        linear(expanded, at(block.mlp_proj_weight), at(block.mlp_proj_bias), rows, channels * 4, channels, &mlp, cancelled);
        hidden.resize(residual.size());
        for (std::size_t index = 0; index < hidden.size(); ++index) {
            hidden[index] = residual[index] + mlp[index];
        }
    }

    layer_norm(hidden, at(final_ln_weight_), at(final_ln_bias_), rows, channels, &normalized, cancelled);
    const float* last_hidden = normalized.data() + (rows - 1) * channels;
    std::vector<float> logits = logits_from_hidden(last_hidden);
    throw_if_cancelled(cancelled);
    return logits;
}

std::vector<float> DenseModel::logits_from_hidden(const float* hidden) const {
    if (hidden == nullptr) {
        throw std::runtime_error("resident dense logits require a final hidden row");
    }
    const float* token_embedding = at(wte_weight_);
    std::vector<float> logits(static_cast<std::size_t>(vocab_size_), 0.0f);
    for (std::int64_t token = 0; token < vocab_size_; ++token) {
        double value = 0.0;
        const float* row_weight = token_embedding + token * channels_;
        for (std::int64_t channel = 0; channel < channels_; ++channel) {
            value += static_cast<double>(hidden[channel]) * row_weight[channel];
        }
        if (!std::isfinite(value)) {
            throw std::runtime_error("resident dense forward produced non-finite logits");
        }
        if (inference_config_.logit_softcap > 0.0) {
            value = inference_config_.logit_softcap *
                std::tanh(value / inference_config_.logit_softcap);
        }
        logits[static_cast<std::size_t>(token)] = static_cast<float>(value);
    }
    return logits;
}

void DenseModel::forward_append_token(
    std::int64_t token,
    std::int64_t position,
    std::vector<std::vector<float>>* key_cache,
    std::vector<std::vector<float>>* value_cache,
    TurboQuantCache* turboquant_cache,
    std::vector<float>* final_hidden_cache,
    const std::atomic<bool>& cancelled) const {
    require_open();
    if (token < 0 || token >= vocab_size_) {
        throw std::runtime_error("resident dense token id is outside the checkpoint vocabulary");
    }
    if (position < 0 || position >= max_seq_len_) {
        throw std::runtime_error("resident dense cache position is outside the checkpoint context window");
    }
    const bool compressed_cache = turboquant_cache != nullptr;
    if (final_hidden_cache == nullptr ||
        final_hidden_cache->size() != static_cast<std::size_t>(max_seq_len_ * channels_)) {
        throw std::runtime_error("resident dense cache has invalid final-hidden storage geometry");
    }
    if (compressed_cache) {
        if (key_cache != nullptr || value_cache != nullptr) {
            throw std::runtime_error("resident dense cache cannot mix lossless and TurboQuant storage");
        }
    } else {
        if (key_cache == nullptr || value_cache == nullptr ||
            key_cache->size() != static_cast<std::size_t>(num_layers_) ||
            value_cache->size() != static_cast<std::size_t>(num_layers_)) {
            throw std::runtime_error("resident dense lossless cache has invalid storage geometry");
        }
        const std::size_t layer_capacity = static_cast<std::size_t>(max_seq_len_ * channels_);
        for (std::int64_t layer = 0; layer < num_layers_; ++layer) {
            if ((*key_cache)[static_cast<std::size_t>(layer)].size() != layer_capacity ||
                (*value_cache)[static_cast<std::size_t>(layer)].size() != layer_capacity) {
                throw std::runtime_error("resident dense lossless cache has invalid layer capacity");
            }
        }
    }

    throw_if_cancelled(cancelled);
    forward_calls_.fetch_add(1);
    const std::int64_t channels = channels_;
    const std::int64_t head_dim = channels_ / num_heads_;
    std::vector<float> hidden(static_cast<std::size_t>(channels), 0.0f);
    const float* token_embedding = at(wte_weight_);
    const float* position_embedding = at(wpe_weight_);
    for (std::int64_t channel = 0; channel < channels; ++channel) {
        hidden[static_cast<std::size_t>(channel)] =
            token_embedding[token * channels + channel] +
            position_embedding[position * channels + channel];
    }

    std::vector<std::vector<float>> staged_keys(
        static_cast<std::size_t>(num_layers_),
        std::vector<float>(static_cast<std::size_t>(channels), 0.0f));
    std::vector<std::vector<float>> staged_values(
        static_cast<std::size_t>(num_layers_),
        std::vector<float>(static_cast<std::size_t>(channels), 0.0f));
    std::vector<float> normalized;
    std::vector<float> qkv;
    std::vector<float> attention(static_cast<std::size_t>(channels), 0.0f);
    std::vector<float> projected;
    std::vector<float> residual;
    std::vector<float> expanded;
    std::vector<float> mlp;
    std::vector<double> scores(static_cast<std::size_t>(position + 1), 0.0);
    const double attention_scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

    for (std::int64_t layer = 0; layer < num_layers_; ++layer) {
        throw_if_cancelled(cancelled);
        const BlockLayout& block = blocks_[static_cast<std::size_t>(layer)];
        layer_norm(hidden, at(block.ln1_weight), at(block.ln1_bias), 1, channels, &normalized, cancelled);
        linear(normalized, at(block.qkv_weight), at(block.qkv_bias), 1, channels, channels * 3, &qkv, cancelled);
        if (inference_config_.use_qk_norm) {
            normalize_query_key_heads(
                &qkv,
                1,
                channels,
                num_heads_,
                inference_config_.qk_norm_eps,
                cancelled);
        }
        std::copy_n(
            qkv.data() + channels,
            channels,
            staged_keys[static_cast<std::size_t>(layer)].data());
        std::copy_n(
            qkv.data() + channels * 2,
            channels,
            staged_values[static_cast<std::size_t>(layer)].data());
        std::fill(attention.begin(), attention.end(), 0.0f);
        const std::vector<float>* layer_keys = compressed_cache
            ? nullptr
            : &(*key_cache)[static_cast<std::size_t>(layer)];
        const std::vector<float>* layer_values = compressed_cache
            ? nullptr
            : &(*value_cache)[static_cast<std::size_t>(layer)];
        const auto& current_key = staged_keys[static_cast<std::size_t>(layer)];
        const auto& current_value = staged_values[static_cast<std::size_t>(layer)];
        if (compressed_cache && turboquant_cache->tile_attention_enabled()) {
            turboquant_cache->tile_attention(
                layer,
                position,
                qkv.data(),
                current_key.data(),
                current_value.data(),
                attention.data(),
                static_cast<float>(attention_scale),
                cancelled);
        } else {
            for (std::int64_t head = 0; head < num_heads_; ++head) {
                double max_score = -std::numeric_limits<double>::infinity();
                const std::int64_t head_offset = head * head_dim;
                const float* query = qkv.data() + head_offset;
                for (std::int64_t key_row = 0; key_row <= position; ++key_row) {
                    double score = 0.0;
                    if (key_row != position && compressed_cache) {
                        score = turboquant_cache->key_inner_product(
                            layer, key_row, head, query);
                    } else {
                        const float* key = key_row == position
                            ? current_key.data() + head_offset
                            : layer_keys->data() + key_row * channels + head_offset;
                        for (std::int64_t dim = 0; dim < head_dim; ++dim) {
                            score += static_cast<double>(query[dim]) * key[dim];
                        }
                    }
                    score *= attention_scale;
                    scores[static_cast<std::size_t>(key_row)] = score;
                    max_score = std::max(max_score, score);
                }
                double denominator = 0.0;
                for (std::int64_t key_row = 0; key_row <= position; ++key_row) {
                    const double probability =
                        std::exp(scores[static_cast<std::size_t>(key_row)] - max_score);
                    scores[static_cast<std::size_t>(key_row)] = probability;
                    denominator += probability;
                }
                if (compressed_cache) {
                    for (std::int64_t key_row = 0; key_row <= position; ++key_row) {
                        const double weight =
                            scores[static_cast<std::size_t>(key_row)] / denominator;
                        if (key_row == position) {
                            for (std::int64_t dim = 0; dim < head_dim; ++dim) {
                                attention[static_cast<std::size_t>(head_offset + dim)] =
                                    static_cast<float>(
                                        static_cast<double>(attention[static_cast<std::size_t>(head_offset + dim)]) +
                                        weight * current_value[static_cast<std::size_t>(head_offset + dim)]);
                            }
                        } else {
                            turboquant_cache->accumulate_value(
                                layer,
                                key_row,
                                head,
                                weight,
                                attention.data() + head_offset);
                        }
                    }
                } else {
                    for (std::int64_t dim = 0; dim < head_dim; ++dim) {
                        double value = 0.0;
                        for (std::int64_t key_row = 0; key_row <= position; ++key_row) {
                            const float* cached_value = key_row == position
                                ? current_value.data() + head_offset
                                : layer_values->data() + key_row * channels + head_offset;
                            value += (scores[static_cast<std::size_t>(key_row)] / denominator) *
                                cached_value[dim];
                        }
                        attention[static_cast<std::size_t>(head_offset + dim)] =
                            static_cast<float>(value);
                    }
                }
            }
        }
        linear(attention, at(block.attn_proj_weight), at(block.attn_proj_bias), 1, channels, channels, &projected, cancelled);
        residual.resize(static_cast<std::size_t>(channels));
        for (std::int64_t channel = 0; channel < channels; ++channel) {
            residual[static_cast<std::size_t>(channel)] =
                hidden[static_cast<std::size_t>(channel)] + projected[static_cast<std::size_t>(channel)];
        }
        layer_norm(residual, at(block.ln2_weight), at(block.ln2_bias), 1, channels, &normalized, cancelled);
        linear(normalized, at(block.fc_weight), at(block.fc_bias), 1, channels, channels * 4, &expanded, cancelled);
        apply_mlp_activation(&expanded, inference_config_.mlp_activation, cancelled);
        linear(expanded, at(block.mlp_proj_weight), at(block.mlp_proj_bias), 1, channels * 4, channels, &mlp, cancelled);
        hidden.resize(static_cast<std::size_t>(channels));
        for (std::int64_t channel = 0; channel < channels; ++channel) {
            hidden[static_cast<std::size_t>(channel)] =
                residual[static_cast<std::size_t>(channel)] + mlp[static_cast<std::size_t>(channel)];
        }
    }

    layer_norm(hidden, at(final_ln_weight_), at(final_ln_bias_), 1, channels, &normalized, cancelled);
    const std::size_t position_offset = static_cast<std::size_t>(position * channels);
    for (std::int64_t layer = 0; layer < num_layers_; ++layer) {
        if (compressed_cache) {
            turboquant_cache->encode_row(
                layer,
                position,
                staged_keys[static_cast<std::size_t>(layer)].data(),
                staged_values[static_cast<std::size_t>(layer)].data(),
                cancelled);
        } else {
            std::copy(
                staged_keys[static_cast<std::size_t>(layer)].begin(),
                staged_keys[static_cast<std::size_t>(layer)].end(),
                (*key_cache)[static_cast<std::size_t>(layer)].begin() + position_offset);
            std::copy(
                staged_values[static_cast<std::size_t>(layer)].begin(),
                staged_values[static_cast<std::size_t>(layer)].end(),
                (*value_cache)[static_cast<std::size_t>(layer)].begin() + position_offset);
        }
    }
    std::copy(
        normalized.begin(),
        normalized.end(),
        final_hidden_cache->begin() + position_offset);
    // The caller advances its logical cache length only after this final
    // cancellation check, so partially written rows remain unreachable.
    throw_if_cancelled(cancelled);
}

DecodeResult DenseModel::select_token(
    const std::vector<float>& logits,
    const std::vector<std::int64_t>&,
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
    if (logits.size() != static_cast<std::size_t>(vocab_size_)) {
        throw std::runtime_error("resident dense forward returned the wrong vocabulary width");
    }
    std::int64_t selected = 0;
    for (std::int64_t token = 1; token < vocab_size_; ++token) {
        if (logits[static_cast<std::size_t>(token)] > logits[static_cast<std::size_t>(selected)]) {
            selected = token;
        }
    }
    if (config.temperature > 0.0 && config.top_k != 1) {
        std::vector<std::int64_t> candidates(static_cast<std::size_t>(vocab_size_));
        std::iota(candidates.begin(), candidates.end(), 0);
        std::sort(candidates.begin(), candidates.end(), [&](std::int64_t left, std::int64_t right) {
            const float a = logits[static_cast<std::size_t>(left)];
            const float b = logits[static_cast<std::size_t>(right)];
            return a == b ? left < right : a > b;
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
            throw std::runtime_error("resident dense sampling probabilities are invalid");
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

DenseSession::DenseSession(
    std::shared_ptr<DenseModel> model,
    std::int64_t seed,
    KVCacheMode cache_mode,
    std::shared_ptr<const TurboQuantCodec> turboquant_codec,
    std::unique_ptr<TileTurboQuantSession> tile_turboquant_session)
    : model_(std::move(model)),
      cache_mode_(cache_mode),
      seed_(seed),
      rng_(static_cast<std::mt19937_64::result_type>(seed)) {
    tokens_.reserve(static_cast<std::size_t>(model_->max_seq_len_));
    if (cache_mode_ == KVCacheMode::Full || cache_mode_ == KVCacheMode::TurboQuant) {
        const std::int64_t layer_values = checked_mul(
            model_->max_seq_len_, model_->channels_, "resident cache layer capacity");
        if (cache_mode_ == KVCacheMode::Full) {
            if (turboquant_codec || tile_turboquant_session) {
                throw std::runtime_error(
                    "lossless resident cache must not receive TurboQuant state");
            }
            key_cache_ = std::make_shared<std::vector<std::vector<float>>>(
                static_cast<std::size_t>(model_->num_layers_),
                std::vector<float>(static_cast<std::size_t>(layer_values), 0.0f));
            value_cache_ = std::make_shared<std::vector<std::vector<float>>>(
                static_cast<std::size_t>(model_->num_layers_),
                std::vector<float>(static_cast<std::size_t>(layer_values), 0.0f));
        } else {
            if (!turboquant_codec) {
                throw std::runtime_error("TurboQuant resident cache requires codec tables");
            }
            turboquant_cache_ = std::make_unique<TurboQuantCache>(
                model_->num_layers_,
                model_->num_heads_,
                model_->max_seq_len_,
                model_->channels_,
                std::move(turboquant_codec),
                std::move(tile_turboquant_session));
        }
        final_hidden_cache_ = std::make_shared<std::vector<float>>(
            static_cast<std::size_t>(layer_values), 0.0f);
    } else if (turboquant_codec || tile_turboquant_session) {
        throw std::runtime_error("cache-off resident session must not receive TurboQuant tables");
    }
    model_->session_opened();
}

DenseSession::DenseSession(
    std::shared_ptr<DenseModel> model,
    std::int64_t seed,
    std::vector<std::int64_t> tokens,
    std::int64_t cache_length,
    std::unique_ptr<TurboQuantCache> turboquant_cache,
    std::shared_ptr<std::vector<float>> final_hidden_cache)
    : model_(std::move(model)),
      tokens_(std::move(tokens)),
      cache_mode_(KVCacheMode::TurboQuant),
      turboquant_cache_(std::move(turboquant_cache)),
      final_hidden_cache_(std::move(final_hidden_cache)),
      cache_length_(cache_length),
      seed_(seed),
      rng_(static_cast<std::mt19937_64::result_type>(seed)),
      prefix_cow_forked_from_tokens_(cache_length) {
    if (!turboquant_cache_ || !final_hidden_cache_ ||
        turboquant_cache_->tile_attention_enabled()) {
        throw std::runtime_error(
            "resident dense TurboQuant prefix fork requires complete CPU cache storage");
    }
    tokens_.reserve(static_cast<std::size_t>(model_->max_seq_len_));
    model_->session_opened();
}

DenseSession::DenseSession(
    std::shared_ptr<DenseModel> model,
    std::int64_t seed,
    std::vector<std::int64_t> tokens,
    std::int64_t cache_length,
    std::shared_ptr<std::vector<std::vector<float>>> key_cache,
    std::shared_ptr<std::vector<std::vector<float>>> value_cache,
    std::shared_ptr<std::vector<float>> final_hidden_cache)
    : model_(std::move(model)),
      tokens_(std::move(tokens)),
      cache_mode_(KVCacheMode::Full),
      key_cache_(std::move(key_cache)),
      value_cache_(std::move(value_cache)),
      final_hidden_cache_(std::move(final_hidden_cache)),
      cache_length_(cache_length),
      seed_(seed),
      rng_(static_cast<std::mt19937_64::result_type>(seed)),
      prefix_cow_forked_from_tokens_(cache_length) {
    if (!key_cache_ || !value_cache_ || !final_hidden_cache_) {
        throw std::runtime_error("resident dense prefix fork requires complete full-cache storage");
    }
    tokens_.reserve(static_cast<std::size_t>(model_->max_seq_len_));
    model_->session_opened();
}

DenseSession::~DenseSession() {
    close();
}

void DenseSession::require_open() const {
    if (closed_) {
        throw std::runtime_error("resident inference session is closed");
    }
    if (model_->closed()) {
        throw std::runtime_error("resident inference model is closed");
    }
}

std::int64_t DenseSession::full_cache_capacity_bytes() const {
    const std::int64_t cached_vectors_per_token = checked_add(
        checked_mul(model_->num_layers_, 2, "resident K/V vectors"),
        1,
        "resident K/V plus final-hidden vectors");
    const std::int64_t bytes_per_token = checked_mul(
        checked_mul(cached_vectors_per_token, model_->channels_, "resident cache row"),
        static_cast<std::int64_t>(sizeof(float)),
        "resident cache row bytes");
    return checked_mul(
        model_->max_seq_len_, bytes_per_token, "resident cache capacity bytes");
}

std::int64_t DenseSession::turboquant_cache_capacity_bytes() const {
    if (!turboquant_cache_) {
        throw std::runtime_error("TurboQuant session is missing its compressed cache");
    }
    const std::int64_t hidden_capacity_bytes = checked_mul(
        checked_mul(model_->max_seq_len_, model_->channels_, "resident final hidden capacity"),
        static_cast<std::int64_t>(sizeof(float)),
        "resident final hidden capacity bytes");
    return checked_add(
        turboquant_cache_->capacity_bytes(),
        hidden_capacity_bytes,
        "TurboQuant total cache capacity bytes");
}

DenseSession::CowDetachSnapshot DenseSession::detach_cache_before_write() {
    CowDetachSnapshot snapshot;
    if (cache_mode_ == KVCacheMode::Off) {
        return snapshot;
    }

    std::int64_t detached_capacity = 0;
    if (cache_mode_ == KVCacheMode::Full) {
        if (!key_cache_ || !value_cache_ || !final_hidden_cache_) {
            throw std::runtime_error("resident dense full cache storage is unavailable");
        }
        const bool shared = key_cache_.use_count() > 1 ||
            value_cache_.use_count() > 1 || final_hidden_cache_.use_count() > 1;
        if (!shared) {
            return snapshot;
        }
        detached_capacity = full_cache_capacity_bytes();
        const std::int64_t detached_bytes_after = checked_add(
            prefix_cow_detached_capacity_bytes_,
            detached_capacity,
            "resident prefix COW detached bytes");
        auto detached_keys =
            std::make_shared<std::vector<std::vector<float>>>(*key_cache_);
        auto detached_values =
            std::make_shared<std::vector<std::vector<float>>>(*value_cache_);
        auto detached_hidden = std::make_shared<std::vector<float>>(*final_hidden_cache_);
        snapshot.key_cache = key_cache_;
        snapshot.value_cache = value_cache_;
        snapshot.final_hidden_cache = final_hidden_cache_;
        snapshot.detached = true;
        snapshot.detach_count = prefix_cow_detach_count_;
        snapshot.detached_capacity_bytes = prefix_cow_detached_capacity_bytes_;
        key_cache_ = std::move(detached_keys);
        value_cache_ = std::move(detached_values);
        final_hidden_cache_ = std::move(detached_hidden);
        prefix_cow_detach_count_ = snapshot.detach_count + 1;
        prefix_cow_detached_capacity_bytes_ = detached_bytes_after;
        return snapshot;
    }

    if (!turboquant_cache_ || !final_hidden_cache_) {
        throw std::runtime_error("resident dense TurboQuant cache storage is unavailable");
    }
    if (turboquant_cache_->tile_attention_enabled()) {
        return snapshot;
    }
    const bool shared = turboquant_cache_->storage_use_count() > 1 ||
        final_hidden_cache_.use_count() > 1;
    if (!shared) {
        return snapshot;
    }
    detached_capacity = turboquant_cache_capacity_bytes();
    const std::int64_t detached_bytes_after = checked_add(
        prefix_cow_detached_capacity_bytes_,
        detached_capacity,
        "resident TurboQuant prefix COW detached bytes");
    auto detached_storage = turboquant_cache_->clone_storage();
    auto detached_hidden = std::make_shared<std::vector<float>>(*final_hidden_cache_);
    snapshot.turboquant_storage = turboquant_cache_->storage_handle();
    snapshot.final_hidden_cache = final_hidden_cache_;
    snapshot.detached = true;
    snapshot.detach_count = prefix_cow_detach_count_;
    snapshot.detached_capacity_bytes = prefix_cow_detached_capacity_bytes_;
    // Both allocations are complete before either session-visible pointer is
    // published.  Under the session mutex, readers can only observe the old
    // joint store or the new private joint store.
    turboquant_cache_->replace_storage(std::move(detached_storage));
    final_hidden_cache_ = std::move(detached_hidden);
    prefix_cow_detach_count_ = snapshot.detach_count + 1;
    prefix_cow_detached_capacity_bytes_ = detached_bytes_after;
    return snapshot;
}

void DenseSession::rollback_cache_detach(CowDetachSnapshot snapshot) noexcept {
    if (!snapshot.detached) {
        return;
    }
    if (cache_mode_ == KVCacheMode::Full) {
        key_cache_ = std::move(snapshot.key_cache);
        value_cache_ = std::move(snapshot.value_cache);
    } else if (cache_mode_ == KVCacheMode::TurboQuant && turboquant_cache_) {
        turboquant_cache_->replace_storage(std::move(snapshot.turboquant_storage));
    }
    final_hidden_cache_ = std::move(snapshot.final_hidden_cache);
    prefix_cow_detach_count_ = snapshot.detach_count;
    prefix_cow_detached_capacity_bytes_ = snapshot.detached_capacity_bytes;
}

std::shared_ptr<DenseSession> DenseSession::fork_prefix(
    std::int64_t token_count,
    std::int64_t seed) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::shared_lock<std::shared_mutex> model_lock(model_->lifecycle_mutex_);
    require_open();
    if (cache_mode_ == KVCacheMode::Off) {
        throw std::runtime_error(
            "resident prefix COW requires a full-cache or CPU TurboQuant dense session");
    }
    if (cache_length_ != static_cast<std::int64_t>(tokens_.size())) {
        throw std::runtime_error("resident dense cache length does not match session history");
    }
    if (token_count <= 0 || token_count > cache_length_) {
        throw std::runtime_error(
            "resident prefix COW token_count must select a non-empty cached prefix");
    }
    if (cache_mode_ == KVCacheMode::Full) {
        if (!key_cache_ || !value_cache_ || !final_hidden_cache_) {
            throw std::runtime_error(
                "resident dense prefix fork requires complete full-cache storage");
        }
    } else if (!turboquant_cache_ || !final_hidden_cache_ ||
        turboquant_cache_->tile_attention_enabled()) {
        throw std::runtime_error(
            "resident TurboQuant prefix COW requires complete CPU packed cache storage");
    }
    std::vector<std::int64_t> forked_tokens(
        tokens_.begin(), tokens_.begin() + static_cast<std::ptrdiff_t>(token_count));
    std::shared_ptr<DenseSession> child;
    if (cache_mode_ == KVCacheMode::Full) {
        child = std::shared_ptr<DenseSession>(new DenseSession(
            model_,
            seed,
            std::move(forked_tokens),
            token_count,
            key_cache_,
            value_cache_,
            final_hidden_cache_));
    } else {
        child = std::shared_ptr<DenseSession>(new DenseSession(
            model_,
            seed,
            std::move(forked_tokens),
            token_count,
            turboquant_cache_->fork_shared_cpu(),
            final_hidden_cache_));
    }
    ++prefix_cow_forks_created_;
    return child;
}

void DenseSession::prefill(const std::vector<std::int64_t>& token_ids, std::int64_t start_position) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::shared_lock<std::shared_mutex> model_lock(model_->lifecycle_mutex_);
    require_open();
    throw_if_cancelled(cancelled_);
    if (start_position != static_cast<std::int64_t>(tokens_.size())) {
        throw std::runtime_error("prefill start_position does not match native session history");
    }
    if (checked_add(start_position, static_cast<std::int64_t>(token_ids.size()), "prefill history") >
        model_->max_seq_len()) {
        throw std::runtime_error("prefill would exceed the checkpoint context window");
    }
    for (std::int64_t token : token_ids) {
        if (token < 0 || token >= model_->vocab_size()) {
            throw std::runtime_error("prefill token id is outside the checkpoint vocabulary");
        }
    }
    if (cache_mode_ != KVCacheMode::Off) {
        if (cache_length_ != start_position) {
            throw std::runtime_error("resident dense cache length does not match session history");
        }
        const std::int64_t initial_length = cache_length_;
        CowDetachSnapshot detach_snapshot;
        try {
            if (!token_ids.empty()) {
                detach_snapshot = detach_cache_before_write();
            }
            for (std::int64_t token : token_ids) {
                model_->forward_append_token(
                    token,
                    cache_length_,
                    cache_mode_ == KVCacheMode::Full ? key_cache_.get() : nullptr,
                    cache_mode_ == KVCacheMode::Full ? value_cache_.get() : nullptr,
                    turboquant_cache_.get(),
                    final_hidden_cache_.get(),
                    cancelled_);
                tokens_.push_back(token);
                ++cache_length_;
            }
        } catch (...) {
            tokens_.resize(static_cast<std::size_t>(initial_length));
            cache_length_ = initial_length;
            rollback_cache_detach(std::move(detach_snapshot));
            throw;
        }
    } else {
        tokens_.insert(tokens_.end(), token_ids.begin(), token_ids.end());
    }
    ++prefill_calls_;
    prefill_tokens_ += static_cast<std::int64_t>(token_ids.size());
}

std::vector<float> DenseSession::current_logits() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::shared_lock<std::shared_mutex> model_lock(model_->lifecycle_mutex_);
    require_open();
    throw_if_cancelled(cancelled_);
    if (tokens_.empty()) {
        throw std::runtime_error("current_logits requires a non-empty prefilled token history");
    }
    if (cache_mode_ != KVCacheMode::Off) {
        if (cache_length_ != static_cast<std::int64_t>(tokens_.size())) {
            throw std::runtime_error("resident dense cache length does not match session history");
        }
        const std::size_t hidden_offset = static_cast<std::size_t>(
            (cache_length_ - 1) * model_->channels_);
        return model_->logits_from_hidden(final_hidden_cache_->data() + hidden_offset);
    }
    return model_->forward_last_logits(tokens_, cancelled_);
}

DecodeResult DenseSession::decode_one(const GenerationConfig& config) {
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
    const std::optional<std::int64_t> active_generation_seed_before = active_generation_seed_;
    const std::int64_t cache_length_before = cache_length_;
    const std::size_t token_count_before = tokens_.size();
    CowDetachSnapshot detach_snapshot;
    try {
        if (config.seed.has_value() && config.seed != active_generation_seed_) {
            rng_.seed(static_cast<std::mt19937_64::result_type>(*config.seed));
            active_generation_seed_ = config.seed;
        }
        std::vector<float> logits;
        if (cache_mode_ != KVCacheMode::Off) {
            if (cache_length_ != static_cast<std::int64_t>(tokens_.size())) {
                throw std::runtime_error("resident dense cache length does not match session history");
            }
            const std::size_t hidden_offset = static_cast<std::size_t>(
                (cache_length_ - 1) * model_->channels_);
            logits = model_->logits_from_hidden(final_hidden_cache_->data() + hidden_offset);
        } else {
            logits = model_->forward_last_logits(tokens_, cancelled_);
        }
        DecodeResult result = model_->select_token(logits, tokens_, config, rng_);
        throw_if_cancelled(cancelled_);
        if (cache_mode_ != KVCacheMode::Off) {
            detach_snapshot = detach_cache_before_write();
            model_->forward_append_token(
                result.token_id,
                cache_length_,
                cache_mode_ == KVCacheMode::Full ? key_cache_.get() : nullptr,
                cache_mode_ == KVCacheMode::Full ? value_cache_.get() : nullptr,
                turboquant_cache_.get(),
                final_hidden_cache_.get(),
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
        tokens_.resize(token_count_before);
        cache_length_ = cache_length_before;
        rollback_cache_detach(std::move(detach_snapshot));
        rng_ = rng_before;
        active_generation_seed_ = active_generation_seed_before;
        throw;
    }
}

void DenseSession::truncate(std::int64_t token_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::shared_lock<std::shared_mutex> model_lock(model_->lifecycle_mutex_);
    require_open();
    if (token_count < 0 || token_count > static_cast<std::int64_t>(tokens_.size())) {
        throw std::runtime_error("truncate token_count is outside the native session history");
    }
    tokens_.resize(static_cast<std::size_t>(token_count));
    if (cache_mode_ != KVCacheMode::Off) {
        cache_length_ = token_count;
    }
    ++truncate_calls_;
}

void DenseSession::reset() {
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

void DenseSession::cancel() noexcept {
    cancelled_.store(true);
}

void DenseSession::close() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!closed_) {
        closed_ = true;
        cancelled_.store(true);
        tokens_.clear();
        cache_length_ = 0;
        key_cache_.reset();
        value_cache_.reset();
        final_hidden_cache_.reset();
        turboquant_cache_.reset();
        model_->session_closed();
    }
}

SessionStats DenseSession::stats() const {
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
    result.cached_tokens = cache_mode_ == KVCacheMode::Off ? 0 : cache_length_;
    if (cache_mode_ == KVCacheMode::Full) {
        const std::int64_t cached_vectors_per_token = checked_add(
            checked_mul(model_->num_layers_, 2, "resident K/V vectors"),
            1,
            "resident K/V plus final-hidden vectors");
        const std::int64_t bytes_per_token = checked_mul(
            checked_mul(cached_vectors_per_token, model_->channels_, "resident cache row"),
            static_cast<std::int64_t>(sizeof(float)),
            "resident cache row bytes");
        result.cache_bytes = checked_mul(cache_length_, bytes_per_token, "resident cache bytes");
        result.uncompressed_cache_bytes = result.cache_bytes;
        result.cache_capacity_bytes = full_cache_capacity_bytes();
        if (!key_cache_ || !value_cache_ || !final_hidden_cache_) {
            throw std::runtime_error("resident dense full cache storage is unavailable");
        }
        result.prefix_cow_storage_use_count = static_cast<std::int64_t>(
            key_cache_.use_count());
        const bool shared = key_cache_.use_count() > 1 ||
            value_cache_.use_count() > 1 || final_hidden_cache_.use_count() > 1;
        if (shared) {
            result.prefix_cow_shared_cached_tokens = cache_length_;
            result.prefix_cow_shared_capacity_bytes = result.cache_capacity_bytes;
        }
    } else if (cache_mode_ == KVCacheMode::TurboQuant) {
        if (!turboquant_cache_) {
            throw std::runtime_error("TurboQuant session is missing its compressed cache");
        }
        const std::int64_t hidden_bytes_per_token = checked_mul(
            model_->channels_, static_cast<std::int64_t>(sizeof(float)),
            "resident final hidden bytes");
        const std::int64_t actual_bytes_per_token = checked_add(
            turboquant_cache_->actual_bytes_per_token(),
            hidden_bytes_per_token,
            "TurboQuant K/V plus final hidden bytes");
        const std::int64_t uncompressed_bytes_per_token = checked_add(
            turboquant_cache_->uncompressed_bytes_per_token(),
            hidden_bytes_per_token,
            "lossless K/V plus final hidden bytes");
        result.cache_bytes = checked_mul(
            cache_length_, actual_bytes_per_token, "TurboQuant logical cache bytes");
        result.uncompressed_cache_bytes = checked_mul(
            cache_length_, uncompressed_bytes_per_token,
            "TurboQuant uncompressed cache bytes");
        result.cache_capacity_bytes = turboquant_cache_capacity_bytes();
        result.turboquant_profile = turboquant_cache_->profile_name();
        result.turboquant_cpu_compressed_attention_calls =
            turboquant_cache_->cpu_compressed_attention_calls();
        const TileTurboQuantSessionStats tile_stats = turboquant_cache_->tile_stats();
        result.turboquant_attention_backend = tile_stats.backend;
        result.turboquant_tile_ops_lib = tile_stats.tile_ops_lib;
        result.turboquant_cuda_runtime_lib = tile_stats.cuda_runtime_lib;
        result.turboquant_cuda_device = tile_stats.device;
        result.turboquant_gpu_launches = tile_stats.gpu_launches;
        result.turboquant_row_uploads = tile_stats.row_uploads;
        result.turboquant_h2d_bytes = tile_stats.h2d_bytes;
        result.turboquant_d2h_bytes = tile_stats.d2h_bytes;
        if (!turboquant_cache_->tile_attention_enabled()) {
            result.prefix_cow_storage_use_count =
                turboquant_cache_->storage_use_count();
            const bool shared = turboquant_cache_->storage_use_count() > 1 ||
                final_hidden_cache_.use_count() > 1;
            if (shared) {
                result.prefix_cow_shared_cached_tokens = cache_length_;
                result.prefix_cow_shared_capacity_bytes = result.cache_capacity_bytes;
            }
        }
    }
    result.decode_rows_processed = decode_rows_processed_;
    result.strict_model_compute = strict_model_compute_;
    result.lossy_cache = cache_mode_ == KVCacheMode::TurboQuant;
    result.cancelled = cancelled_.load();
    result.closed = closed_;
    return result;
}

}  // namespace neuralfn::resident_dense
