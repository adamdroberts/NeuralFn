#pragma once

// Native, dependency-free transformer encoder used by the embedding trainer.
// Matrices are row-major [out, in].  The implementation intentionally keeps a
// scalar CPU reference path so imported checkpoints and exact-resume behavior
// can be validated without Python or Torch; the checkpoint layout is shared by
// the accelerated frontend.

struct Parameter {
    std::vector<float> value;
    std::vector<float> grad;
    std::vector<float> first;
    std::vector<float> second;

    void resize(size_t n, float fill = 0.0f) {
        value.assign(n, fill); grad.assign(n, 0.0f); first.assign(n, 0.0f); second.assign(n, 0.0f);
    }
    void zero_grad() { std::fill(grad.begin(), grad.end(), 0.0f); }
};

struct Linear {
    int input_dim = 0;
    int output_dim = 0;
    Parameter weight;
    Parameter bias;
    Parameter lora_a;
    Parameter lora_b;
    std::vector<uint8_t> quantized;
    std::vector<float> quantized_scales;
};

struct TransformerLayer {
    Parameter ln1_weight, ln1_bias, ln2_weight, ln2_bias;
    Linear query, key, value, attention_output, ff_input, ff_output;
};

struct LinearCache {
    std::vector<float> input;
    std::vector<float> adapter_input;
    std::vector<float> adapter_multiplier;
    int rows = 0;
};

struct NormCache {
    std::vector<float> input;
    std::vector<float> output;
    std::vector<float> mean;
    std::vector<float> rstd;
    int rows = 0;
};

struct AttentionCache {
    LinearCache query_linear, key_linear, value_linear, output_linear;
    std::vector<float> query, key, value, probabilities, context;
    int tokens = 0;
};

struct LayerCache {
    std::vector<float> input;
    NormCache norm1, norm2;
    AttentionCache attention;
    std::vector<float> after_attention;
    LinearCache ff_input_linear, ff_output_linear;
    std::vector<float> ff_pre, ff_activation, ff_output;
};

struct SequenceCache {
    std::vector<uint32_t> tokens;
    std::vector<float> embedding_sum;
    NormCache embedding_norm;
    std::vector<LayerCache> layers;
    NormCache final_norm;
    std::vector<float> hidden;
    bool training = false;
};

struct Encoded {
    SequenceCache sequence;
    std::vector<size_t> pooled_positions;
    std::vector<float> pooled_weights;
    std::vector<float> hidden;
    std::vector<float> pre_norm;
    std::vector<float> value;
    LinearCache projection_linear;
};

static float nfn_gelu(float x) {
    constexpr float k = 0.7978845608028654f;
    return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
}

static float nfn_gelu_grad(float x) {
    constexpr float k = 0.7978845608028654f;
    const float x2 = x * x;
    const float inner = k * (x + 0.044715f * x * x2);
    const float t = std::tanh(inner);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * k * (1.0f + 3.0f * 0.044715f * x2);
}

static float nfn_gelu_exact(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865475f));
}

static float nfn_gelu_exact_grad(float x) {
    constexpr float inv_sqrt_two = 0.7071067811865475f;
    constexpr float inv_sqrt_two_pi = 0.3989422804014327f;
    return 0.5f * (1.0f + std::erf(x * inv_sqrt_two)) + x * inv_sqrt_two_pi * std::exp(-0.5f * x * x);
}

static void add_inplace(std::vector<float>& target, const std::vector<float>& source) {
    if (target.size() != source.size()) throw std::runtime_error("native transformer tensor size mismatch");
    for (size_t i = 0; i < target.size(); ++i) target[i] += source[i];
}

struct Model {
    Config cfg;
    uint32_t step = 0;
    Parameter token, position, token_type, embedding_ln_weight, embedding_ln_bias;
    std::vector<TransformerLayer> layers;
    Parameter final_ln_weight, final_ln_bias;
    Linear projection;

    explicit Model(Config config) : cfg(std::move(config)) { initialize(); }

    bool adapters_enabled() const { return cfg.adapter_type != "none"; }
    bool causal() const { return cfg.architecture == "gpt-derived"; }
    bool bert_post_norm() const { return !causal(); }

    void random_parameter(Parameter& parameter, size_t size, std::mt19937& random, float stddev) {
        parameter.resize(size);
        std::normal_distribution<float> normal(0.0f, stddev);
        for (float& value : parameter.value) value = normal(random);
    }

    void init_norm(Parameter& weight, Parameter& bias) {
        weight.resize(static_cast<size_t>(cfg.hidden_dim), 1.0f);
        bias.resize(static_cast<size_t>(cfg.hidden_dim), 0.0f);
    }

    void init_linear(Linear& linear, int input, int output, std::mt19937& random) {
        linear.input_dim = input; linear.output_dim = output;
        random_parameter(linear.weight, static_cast<size_t>(input) * output, random, 0.02f / std::sqrt(std::max(1, input)));
        linear.bias.resize(static_cast<size_t>(output), 0.0f);
    }

    void initialize() {
        if (cfg.intermediate_dim <= 0) cfg.intermediate_dim = cfg.hidden_dim * 4;
        if (cfg.output_dim <= 0) cfg.output_dim = cfg.hidden_dim;
        if (cfg.num_layers <= 0 || cfg.num_heads <= 0 || cfg.hidden_dim % cfg.num_heads != 0)
            throw std::runtime_error("transformer layers/heads must be positive and hidden_dim must divide num_heads");
        std::mt19937 random(static_cast<uint32_t>(cfg.seed));
        random_parameter(token, static_cast<size_t>(cfg.vocab_size) * cfg.hidden_dim, random, 0.02f);
        random_parameter(position, static_cast<size_t>(cfg.max_tokens) * cfg.hidden_dim, random, 0.02f);
        token_type.resize(bert_post_norm() ? static_cast<size_t>(2 * cfg.hidden_dim) : 0, 0.0f);
        init_norm(embedding_ln_weight, embedding_ln_bias);
        layers.resize(static_cast<size_t>(cfg.num_layers));
        for (auto& layer : layers) {
            init_norm(layer.ln1_weight, layer.ln1_bias); init_norm(layer.ln2_weight, layer.ln2_bias);
            init_linear(layer.query, cfg.hidden_dim, cfg.hidden_dim, random);
            init_linear(layer.key, cfg.hidden_dim, cfg.hidden_dim, random);
            init_linear(layer.value, cfg.hidden_dim, cfg.hidden_dim, random);
            init_linear(layer.attention_output, cfg.hidden_dim, cfg.hidden_dim, random);
            init_linear(layer.ff_input, cfg.hidden_dim, cfg.intermediate_dim, random);
            init_linear(layer.ff_output, cfg.intermediate_dim, cfg.hidden_dim, random);
        }
        init_norm(final_ln_weight, final_ln_bias);
        init_linear(projection, cfg.hidden_dim, cfg.output_dim, random);
        if (cfg.output_dim == cfg.hidden_dim) {
            std::fill(projection.weight.value.begin(), projection.weight.value.end(), 0.0f);
            for (int i = 0; i < cfg.hidden_dim; ++i) projection.weight.value[static_cast<size_t>(i) * cfg.hidden_dim + i] = 1.0f;
        }
        if (adapters_enabled()) initialize_adapter(cfg.adapter_type);
    }

    std::vector<Linear*> linear_modules() {
        std::vector<Linear*> result;
        for (auto& layer : layers) {
            result.insert(result.end(), {&layer.query, &layer.key, &layer.value, &layer.attention_output, &layer.ff_input, &layer.ff_output});
        }
        result.push_back(&projection);
        return result;
    }
    std::vector<const Linear*> linear_modules() const {
        std::vector<const Linear*> result;
        for (const auto& layer : layers) {
            result.insert(result.end(), {&layer.query, &layer.key, &layer.value, &layer.attention_output, &layer.ff_input, &layer.ff_output});
        }
        result.push_back(&projection);
        return result;
    }

    std::vector<Parameter*> base_parameters() {
        std::vector<Parameter*> result{&token, &position, &token_type, &embedding_ln_weight, &embedding_ln_bias};
        for (auto& layer : layers) {
            result.insert(result.end(), {&layer.ln1_weight, &layer.ln1_bias, &layer.ln2_weight, &layer.ln2_bias});
            for (Linear* linear : std::vector<Linear*>{&layer.query, &layer.key, &layer.value, &layer.attention_output, &layer.ff_input, &layer.ff_output}) {
                result.push_back(&linear->weight); result.push_back(&linear->bias);
            }
        }
        result.insert(result.end(), {&final_ln_weight, &final_ln_bias, &projection.weight, &projection.bias});
        return result;
    }
    std::vector<const Parameter*> base_parameters() const {
        std::vector<const Parameter*> result{&token, &position, &token_type, &embedding_ln_weight, &embedding_ln_bias};
        for (const auto& layer : layers) {
            result.insert(result.end(), {&layer.ln1_weight, &layer.ln1_bias, &layer.ln2_weight, &layer.ln2_bias});
            for (const Linear* linear : std::vector<const Linear*>{&layer.query, &layer.key, &layer.value, &layer.attention_output, &layer.ff_input, &layer.ff_output}) {
                result.push_back(&linear->weight); result.push_back(&linear->bias);
            }
        }
        result.insert(result.end(), {&final_ln_weight, &final_ln_bias, &projection.weight, &projection.bias});
        return result;
    }
    std::vector<Parameter*> adapter_parameters() {
        std::vector<Parameter*> result;
        for (Linear* linear : linear_modules()) { result.push_back(&linear->lora_a); result.push_back(&linear->lora_b); }
        return result;
    }
    std::vector<const Parameter*> adapter_parameters() const {
        std::vector<const Parameter*> result;
        for (const Linear* linear : linear_modules()) { result.push_back(&linear->lora_a); result.push_back(&linear->lora_b); }
        return result;
    }

    void quantize_linear(Linear& linear) {
        linear.quantized.clear(); linear.quantized_scales.clear();
        if (cfg.adapter_type != "qlora") return;
        static constexpr std::array<float, 16> codebook{-1.0f,-0.6961928f,-0.5250731f,-0.3949175f,-0.2844414f,-0.1847734f,-0.09105f,0.0f,0.0795803f,0.1609302f,0.2461123f,0.3379152f,0.4407098f,0.562617f,0.7229568f,1.0f};
        constexpr size_t group = 64;
        linear.quantized.assign((linear.weight.value.size() + 1) / 2, 0);
        linear.quantized_scales.resize((linear.weight.value.size() + group - 1) / group, 1.0f);
        for (size_t g = 0; g < linear.quantized_scales.size(); ++g) {
            const size_t begin = g * group, end = std::min(begin + group, linear.weight.value.size());
            float scale = 1e-12f;
            for (size_t i = begin; i < end; ++i) scale = std::max(scale, std::abs(linear.weight.value[i]));
            linear.quantized_scales[g] = scale;
            for (size_t i = begin; i < end; ++i) {
                int best = 0; const float normalized = linear.weight.value[i] / scale;
                for (int q = 1; q < 16; ++q) if (std::abs(normalized - codebook[q]) < std::abs(normalized - codebook[best])) best = q;
                if ((i & 1u) == 0) linear.quantized[i / 2] = static_cast<uint8_t>(best);
                else linear.quantized[i / 2] |= static_cast<uint8_t>(best << 4);
            }
        }
    }

    float base_weight(const Linear& linear, size_t index) const {
        if (cfg.adapter_type != "qlora" || linear.quantized.empty()) return linear.weight.value[index];
        static constexpr std::array<float, 16> codebook{-1.0f,-0.6961928f,-0.5250731f,-0.3949175f,-0.2844414f,-0.1847734f,-0.09105f,0.0f,0.0795803f,0.1609302f,0.2461123f,0.3379152f,0.4407098f,0.562617f,0.7229568f,1.0f};
        const uint8_t packed = linear.quantized[index / 2];
        const uint8_t code = (index & 1u) == 0 ? packed & 0x0fu : packed >> 4;
        return linear.quantized_scales[index / 64] * codebook[code];
    }

    float effective_weight(const Linear& linear, int output, int input) const {
        const size_t index = static_cast<size_t>(output) * linear.input_dim + input;
        float value = base_weight(linear, index);
        if (!linear.lora_a.value.empty()) {
            float delta = 0.0f;
            for (int rank = 0; rank < cfg.lora_rank; ++rank)
                delta += linear.lora_b.value[static_cast<size_t>(output) * cfg.lora_rank + rank] * linear.lora_a.value[static_cast<size_t>(rank) * linear.input_dim + input];
            value += (cfg.lora_alpha / static_cast<float>(cfg.lora_rank)) * delta;
        }
        return value;
    }

    void initialize_adapter(const std::string& requested) {
        cfg.adapter_type = requested;
        std::mt19937 random(static_cast<uint32_t>(cfg.seed + 17));
        std::normal_distribution<float> normal(0.0f, 0.02f);
        for (Linear* linear : linear_modules()) {
            linear->lora_a.resize(requested == "none" ? 0 : static_cast<size_t>(cfg.lora_rank) * linear->input_dim);
            linear->lora_b.resize(requested == "none" ? 0 : static_cast<size_t>(linear->output_dim) * cfg.lora_rank, 0.0f);
            for (float& value : linear->lora_a.value) value = normal(random);
            quantize_linear(*linear);
        }
    }

    std::vector<float> layer_norm_forward(const std::vector<float>& input, const Parameter& weight, const Parameter& bias, int rows, NormCache& cache) const {
        const int dim = cfg.hidden_dim;
        cache.input = input; cache.rows = rows; cache.mean.assign(rows, 0.0f); cache.rstd.assign(rows, 0.0f); cache.output.resize(input.size());
        for (int row = 0; row < rows; ++row) {
            float mean = 0.0f;
            for (int d = 0; d < dim; ++d) mean += input[static_cast<size_t>(row) * dim + d];
            mean /= static_cast<float>(dim);
            float variance = 0.0f;
            for (int d = 0; d < dim; ++d) { const float centered = input[static_cast<size_t>(row) * dim + d] - mean; variance += centered * centered; }
            const float rstd = 1.0f / std::sqrt(variance / static_cast<float>(dim) + cfg.layer_norm_epsilon);
            cache.mean[row] = mean; cache.rstd[row] = rstd;
            for (int d = 0; d < dim; ++d) {
                const size_t index = static_cast<size_t>(row) * dim + d;
                cache.output[index] = (input[index] - mean) * rstd * weight.value[d] + bias.value[d];
            }
        }
        return cache.output;
    }

    std::vector<float> layer_norm_backward(const std::vector<float>& grad_output, const NormCache& cache, Parameter& weight, Parameter& bias) {
        const int dim = cfg.hidden_dim; std::vector<float> grad_input(grad_output.size(), 0.0f);
        for (int row = 0; row < cache.rows; ++row) {
            float sum = 0.0f, sum_xhat = 0.0f;
            for (int d = 0; d < dim; ++d) {
                const size_t index = static_cast<size_t>(row) * dim + d;
                const float xhat = (cache.input[index] - cache.mean[row]) * cache.rstd[row];
                const float scaled = grad_output[index] * weight.value[d];
                sum += scaled; sum_xhat += scaled * xhat;
                if (!adapters_enabled()) { weight.grad[d] += grad_output[index] * xhat; bias.grad[d] += grad_output[index]; }
            }
            for (int d = 0; d < dim; ++d) {
                const size_t index = static_cast<size_t>(row) * dim + d;
                const float xhat = (cache.input[index] - cache.mean[row]) * cache.rstd[row];
                const float scaled = grad_output[index] * weight.value[d];
                grad_input[index] = cache.rstd[row] * (scaled - sum / dim - xhat * sum_xhat / dim);
            }
        }
        return grad_input;
    }

    std::vector<float> linear_forward(const std::vector<float>& input, const Linear& linear, int rows, LinearCache& cache, bool training, uint32_t salt) const {
        cache.input = input; cache.rows = rows; cache.adapter_input = input; cache.adapter_multiplier.assign(input.size(), 1.0f);
        if (!linear.lora_a.value.empty() && training && cfg.lora_dropout > 0.0f) {
            const float keep = 1.0f - cfg.lora_dropout;
            for (size_t i = 0; i < input.size(); ++i) {
                uint32_t bits = (step + 1u) * 2654435761u ^ static_cast<uint32_t>(i + 1u) * 2246822519u ^ salt;
                const float sample = static_cast<float>(bits & 0x00ffffffu) / static_cast<float>(0x01000000u);
                cache.adapter_multiplier[i] = sample < cfg.lora_dropout ? 0.0f : 1.0f / keep;
                cache.adapter_input[i] *= cache.adapter_multiplier[i];
            }
        }
        std::vector<float> output(static_cast<size_t>(rows) * linear.output_dim, 0.0f);
        for (int row = 0; row < rows; ++row) for (int out = 0; out < linear.output_dim; ++out) {
            float sum = linear.bias.value[out];
            for (int in = 0; in < linear.input_dim; ++in) sum += base_weight(linear, static_cast<size_t>(out) * linear.input_dim + in) * input[static_cast<size_t>(row) * linear.input_dim + in];
            if (!linear.lora_a.value.empty()) {
                float adapter_sum = 0.0f;
                for (int rank = 0; rank < cfg.lora_rank; ++rank) {
                    float hidden = 0.0f;
                    for (int in = 0; in < linear.input_dim; ++in) hidden += linear.lora_a.value[static_cast<size_t>(rank) * linear.input_dim + in] * cache.adapter_input[static_cast<size_t>(row) * linear.input_dim + in];
                    adapter_sum += linear.lora_b.value[static_cast<size_t>(out) * cfg.lora_rank + rank] * hidden;
                }
                sum += (cfg.lora_alpha / static_cast<float>(cfg.lora_rank)) * adapter_sum;
            }
            output[static_cast<size_t>(row) * linear.output_dim + out] = sum;
        }
        return output;
    }

    std::vector<float> linear_backward(const std::vector<float>& grad_output, Linear& linear, const LinearCache& cache) {
        std::vector<float> grad_input(cache.input.size(), 0.0f);
        for (int row = 0; row < cache.rows; ++row) for (int out = 0; out < linear.output_dim; ++out) {
            const float grad = grad_output[static_cast<size_t>(row) * linear.output_dim + out];
            if (!adapters_enabled()) linear.bias.grad[out] += grad;
            for (int in = 0; in < linear.input_dim; ++in) {
                const size_t wi = static_cast<size_t>(out) * linear.input_dim + in;
                grad_input[static_cast<size_t>(row) * linear.input_dim + in] += base_weight(linear, wi) * grad;
                if (!adapters_enabled()) linear.weight.grad[wi] += grad * cache.input[static_cast<size_t>(row) * linear.input_dim + in];
            }
        }
        if (!linear.lora_a.value.empty()) {
            const float scale = cfg.lora_alpha / static_cast<float>(cfg.lora_rank);
            for (int row = 0; row < cache.rows; ++row) for (int rank = 0; rank < cfg.lora_rank; ++rank) {
                float hidden = 0.0f, grad_hidden = 0.0f;
                for (int in = 0; in < linear.input_dim; ++in) hidden += linear.lora_a.value[static_cast<size_t>(rank) * linear.input_dim + in] * cache.adapter_input[static_cast<size_t>(row) * linear.input_dim + in];
                for (int out = 0; out < linear.output_dim; ++out) {
                    const float grad = grad_output[static_cast<size_t>(row) * linear.output_dim + out];
                    linear.lora_b.grad[static_cast<size_t>(out) * cfg.lora_rank + rank] += scale * grad * hidden;
                    grad_hidden += scale * linear.lora_b.value[static_cast<size_t>(out) * cfg.lora_rank + rank] * grad;
                }
                for (int in = 0; in < linear.input_dim; ++in) {
                    const size_t ai = static_cast<size_t>(rank) * linear.input_dim + in;
                    const size_t xi = static_cast<size_t>(row) * linear.input_dim + in;
                    linear.lora_a.grad[ai] += grad_hidden * cache.adapter_input[xi];
                    grad_input[xi] += grad_hidden * linear.lora_a.value[ai] * cache.adapter_multiplier[xi];
                }
            }
        }
        return grad_input;
    }

    std::vector<float> attention_forward(const std::vector<float>& input, TransformerLayer& layer, int tokens, AttentionCache& cache, bool training, uint32_t salt) const {
        const int hidden = cfg.hidden_dim, heads = cfg.num_heads, head_dim = hidden / heads;
        cache.tokens = tokens;
        cache.query = linear_forward(input, layer.query, tokens, cache.query_linear, training, salt + 1);
        cache.key = linear_forward(input, layer.key, tokens, cache.key_linear, training, salt + 2);
        cache.value = linear_forward(input, layer.value, tokens, cache.value_linear, training, salt + 3);
        cache.probabilities.assign(static_cast<size_t>(heads) * tokens * tokens, 0.0f);
        cache.context.assign(static_cast<size_t>(tokens) * hidden, 0.0f);
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        for (int head = 0; head < heads; ++head) for (int query_pos = 0; query_pos < tokens; ++query_pos) {
            const int limit = causal() ? query_pos + 1 : tokens; float maximum = -std::numeric_limits<float>::infinity();
            for (int key_pos = 0; key_pos < limit; ++key_pos) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) score += cache.query[static_cast<size_t>(query_pos) * hidden + head * head_dim + d] * cache.key[static_cast<size_t>(key_pos) * hidden + head * head_dim + d];
                score *= scale; cache.probabilities[(static_cast<size_t>(head) * tokens + query_pos) * tokens + key_pos] = score; maximum = std::max(maximum, score);
            }
            float denominator = 0.0f;
            for (int key_pos = 0; key_pos < limit; ++key_pos) { float& item = cache.probabilities[(static_cast<size_t>(head) * tokens + query_pos) * tokens + key_pos]; item = std::exp(item - maximum); denominator += item; }
            for (int key_pos = 0; key_pos < limit; ++key_pos) {
                float& probability = cache.probabilities[(static_cast<size_t>(head) * tokens + query_pos) * tokens + key_pos]; probability /= denominator;
                for (int d = 0; d < head_dim; ++d) cache.context[static_cast<size_t>(query_pos) * hidden + head * head_dim + d] += probability * cache.value[static_cast<size_t>(key_pos) * hidden + head * head_dim + d];
            }
        }
        return linear_forward(cache.context, layer.attention_output, tokens, cache.output_linear, training, salt + 4);
    }

    std::vector<float> attention_backward(const std::vector<float>& grad_output, TransformerLayer& layer, AttentionCache& cache) {
        const int tokens = cache.tokens, hidden = cfg.hidden_dim, heads = cfg.num_heads, head_dim = hidden / heads;
        std::vector<float> grad_context = linear_backward(grad_output, layer.attention_output, cache.output_linear);
        std::vector<float> grad_q(cache.query.size(), 0.0f), grad_k(cache.key.size(), 0.0f), grad_v(cache.value.size(), 0.0f);
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        for (int head = 0; head < heads; ++head) for (int query_pos = 0; query_pos < tokens; ++query_pos) {
            const int limit = causal() ? query_pos + 1 : tokens; std::vector<float> grad_probability(limit, 0.0f); float weighted = 0.0f;
            for (int key_pos = 0; key_pos < limit; ++key_pos) {
                const float probability = cache.probabilities[(static_cast<size_t>(head) * tokens + query_pos) * tokens + key_pos];
                for (int d = 0; d < head_dim; ++d) {
                    const size_t qi = static_cast<size_t>(query_pos) * hidden + head * head_dim + d, ki = static_cast<size_t>(key_pos) * hidden + head * head_dim + d;
                    grad_probability[key_pos] += grad_context[qi] * cache.value[ki]; grad_v[ki] += probability * grad_context[qi];
                }
                weighted += probability * grad_probability[key_pos];
            }
            for (int key_pos = 0; key_pos < limit; ++key_pos) {
                const float probability = cache.probabilities[(static_cast<size_t>(head) * tokens + query_pos) * tokens + key_pos];
                const float grad_score = probability * (grad_probability[key_pos] - weighted) * scale;
                for (int d = 0; d < head_dim; ++d) {
                    const size_t qi = static_cast<size_t>(query_pos) * hidden + head * head_dim + d, ki = static_cast<size_t>(key_pos) * hidden + head * head_dim + d;
                    grad_q[qi] += grad_score * cache.key[ki]; grad_k[ki] += grad_score * cache.query[qi];
                }
            }
        }
        auto grad_input = linear_backward(grad_q, layer.query, cache.query_linear);
        add_inplace(grad_input, linear_backward(grad_k, layer.key, cache.key_linear));
        add_inplace(grad_input, linear_backward(grad_v, layer.value, cache.value_linear));
        return grad_input;
    }

    SequenceCache encode_sequence(const std::vector<uint32_t>& ids, bool training) {
        SequenceCache cache; cache.training = training;
        cache.tokens.assign(ids.begin(), ids.begin() + std::min(ids.size(), static_cast<size_t>(cfg.max_tokens)));
        const int tokens_count = static_cast<int>(cache.tokens.size()), hidden = cfg.hidden_dim;
        cache.embedding_sum.assign(static_cast<size_t>(tokens_count) * hidden, 0.0f);
        for (int t = 0; t < tokens_count; ++t) for (int h = 0; h < hidden; ++h) {
            cache.embedding_sum[static_cast<size_t>(t) * hidden + h] = token.value[static_cast<size_t>(cache.tokens[t] % cfg.vocab_size) * hidden + h] + position.value[static_cast<size_t>(t) * hidden + h];
            if (bert_post_norm() && !token_type.value.empty()) cache.embedding_sum[static_cast<size_t>(t) * hidden + h] += token_type.value[h];
        }
        std::vector<float> current = bert_post_norm() ? layer_norm_forward(cache.embedding_sum, embedding_ln_weight, embedding_ln_bias, tokens_count, cache.embedding_norm) : cache.embedding_sum;
        cache.layers.resize(layers.size());
        for (size_t index = 0; index < layers.size(); ++index) {
            auto& layer = layers[index]; auto& layer_cache = cache.layers[index]; layer_cache.input = current;
            if (bert_post_norm()) {
                auto attention = attention_forward(current, layer, tokens_count, layer_cache.attention, training, static_cast<uint32_t>(index * 17));
                add_inplace(attention, current);
                layer_cache.after_attention = layer_norm_forward(attention, layer.ln1_weight, layer.ln1_bias, tokens_count, layer_cache.norm1);
                layer_cache.ff_pre = linear_forward(layer_cache.after_attention, layer.ff_input, tokens_count, layer_cache.ff_input_linear, training, static_cast<uint32_t>(index * 17 + 5));
                layer_cache.ff_activation.resize(layer_cache.ff_pre.size());
                std::transform(layer_cache.ff_pre.begin(), layer_cache.ff_pre.end(), layer_cache.ff_activation.begin(), [this](float value) { return cfg.activation == "gelu" ? nfn_gelu_exact(value) : nfn_gelu(value); });
                layer_cache.ff_output = linear_forward(layer_cache.ff_activation, layer.ff_output, tokens_count, layer_cache.ff_output_linear, training, static_cast<uint32_t>(index * 17 + 6));
                add_inplace(layer_cache.ff_output, layer_cache.after_attention);
                current = layer_norm_forward(layer_cache.ff_output, layer.ln2_weight, layer.ln2_bias, tokens_count, layer_cache.norm2);
            } else {
                auto normed = layer_norm_forward(current, layer.ln1_weight, layer.ln1_bias, tokens_count, layer_cache.norm1);
                auto attention = attention_forward(normed, layer, tokens_count, layer_cache.attention, training, static_cast<uint32_t>(index * 17));
                layer_cache.after_attention = current; add_inplace(layer_cache.after_attention, attention);
                auto ff_normed = layer_norm_forward(layer_cache.after_attention, layer.ln2_weight, layer.ln2_bias, tokens_count, layer_cache.norm2);
                layer_cache.ff_pre = linear_forward(ff_normed, layer.ff_input, tokens_count, layer_cache.ff_input_linear, training, static_cast<uint32_t>(index * 17 + 5));
                layer_cache.ff_activation.resize(layer_cache.ff_pre.size());
                std::transform(layer_cache.ff_pre.begin(), layer_cache.ff_pre.end(), layer_cache.ff_activation.begin(), [this](float value) { return cfg.activation == "gelu" ? nfn_gelu_exact(value) : nfn_gelu(value); });
                layer_cache.ff_output = linear_forward(layer_cache.ff_activation, layer.ff_output, tokens_count, layer_cache.ff_output_linear, training, static_cast<uint32_t>(index * 17 + 6));
                current = layer_cache.after_attention; add_inplace(current, layer_cache.ff_output);
            }
        }
        cache.hidden = causal() ? layer_norm_forward(current, final_ln_weight, final_ln_bias, tokens_count, cache.final_norm) : current;
        return cache;
    }

    void backward_sequence(SequenceCache& cache, std::vector<float> grad) {
        const int rows = static_cast<int>(cache.tokens.size());
        if (causal()) grad = layer_norm_backward(grad, cache.final_norm, final_ln_weight, final_ln_bias);
        for (size_t reverse = layers.size(); reverse-- > 0;) {
            auto& layer = layers[reverse]; auto& lc = cache.layers[reverse];
            if (bert_post_norm()) {
                auto grad_ff_residual = layer_norm_backward(grad, lc.norm2, layer.ln2_weight, layer.ln2_bias);
                auto grad_ff_activation = linear_backward(grad_ff_residual, layer.ff_output, lc.ff_output_linear);
                for (size_t i = 0; i < grad_ff_activation.size(); ++i) grad_ff_activation[i] *= cfg.activation == "gelu" ? nfn_gelu_exact_grad(lc.ff_pre[i]) : nfn_gelu_grad(lc.ff_pre[i]);
                auto grad_after_attention = linear_backward(grad_ff_activation, layer.ff_input, lc.ff_input_linear);
                add_inplace(grad_after_attention, grad_ff_residual);
                auto grad_attention_residual = layer_norm_backward(grad_after_attention, lc.norm1, layer.ln1_weight, layer.ln1_bias);
                auto grad_attention_input = attention_backward(grad_attention_residual, layer, lc.attention);
                add_inplace(grad_attention_input, grad_attention_residual); grad = std::move(grad_attention_input);
            } else {
                auto grad_after_attention = grad;
                auto grad_ff_activation = linear_backward(grad, layer.ff_output, lc.ff_output_linear);
                for (size_t i = 0; i < grad_ff_activation.size(); ++i) grad_ff_activation[i] *= cfg.activation == "gelu" ? nfn_gelu_exact_grad(lc.ff_pre[i]) : nfn_gelu_grad(lc.ff_pre[i]);
                auto grad_ff_normed = linear_backward(grad_ff_activation, layer.ff_input, lc.ff_input_linear);
                add_inplace(grad_after_attention, layer_norm_backward(grad_ff_normed, lc.norm2, layer.ln2_weight, layer.ln2_bias));
                auto grad_input = grad_after_attention;
                auto grad_attn_normed = attention_backward(grad_after_attention, layer, lc.attention);
                add_inplace(grad_input, layer_norm_backward(grad_attn_normed, lc.norm1, layer.ln1_weight, layer.ln1_bias)); grad = std::move(grad_input);
            }
        }
        if (bert_post_norm()) grad = layer_norm_backward(grad, cache.embedding_norm, embedding_ln_weight, embedding_ln_bias);
        if (!adapters_enabled()) for (int row = 0; row < rows; ++row) for (int h = 0; h < cfg.hidden_dim; ++h) {
            const float value = grad[static_cast<size_t>(row) * cfg.hidden_dim + h];
            token.grad[static_cast<size_t>(cache.tokens[row] % cfg.vocab_size) * cfg.hidden_dim + h] += value;
            position.grad[static_cast<size_t>(row) * cfg.hidden_dim + h] += value;
            if (bert_post_norm() && !token_type.grad.empty()) token_type.grad[h] += value;
        }
    }

    Encoded encode(const std::vector<uint32_t>& ids, bool training = false) {
        Encoded out; out.sequence = encode_sequence(ids, training);
        const size_t token_count = out.sequence.tokens.size();
        if (cfg.pooling == "cls") out.pooled_positions = {0};
        else if (cfg.pooling == "last") out.pooled_positions = {token_count - 1};
        else { out.pooled_positions.resize(token_count); std::iota(out.pooled_positions.begin(), out.pooled_positions.end(), 0); }
        out.pooled_weights.assign(out.pooled_positions.size(), 1.0f);
        out.hidden.assign(cfg.hidden_dim, 0.0f); const float denominator = static_cast<float>(out.pooled_positions.size());
        for (size_t position : out.pooled_positions) for (int h = 0; h < cfg.hidden_dim; ++h) out.hidden[h] += out.sequence.hidden[position * cfg.hidden_dim + h] / denominator;
        out.pre_norm = linear_forward(out.hidden, projection, 1, out.projection_linear, training, 0x9e3779b9u);
        out.value = out.pre_norm;
        if (cfg.normalize) { float norm = 0.0f; for (float value : out.value) norm += value * value; norm = std::sqrt(std::max(norm, 1e-12f)); for (float& value : out.value) value /= norm; }
        return out;
    }

    void backward(Encoded& encoded, const std::vector<float>& grad_value) {
        std::vector<float> grad_pre = grad_value;
        if (cfg.normalize) {
            float norm = 0.0f, dot = 0.0f; for (float value : encoded.pre_norm) norm += value * value; norm = std::sqrt(std::max(norm, 1e-12f));
            for (int i = 0; i < cfg.output_dim; ++i) dot += grad_value[i] * encoded.value[i];
            for (int i = 0; i < cfg.output_dim; ++i) grad_pre[i] = (grad_value[i] - encoded.value[i] * dot) / norm;
        }
        auto grad_pooled = linear_backward(grad_pre, projection, encoded.projection_linear);
        std::vector<float> grad_sequence(encoded.sequence.hidden.size(), 0.0f); const float denominator = static_cast<float>(encoded.pooled_positions.size());
        for (size_t position : encoded.pooled_positions) for (int h = 0; h < cfg.hidden_dim; ++h) grad_sequence[position * cfg.hidden_dim + h] += grad_pooled[h] / denominator;
        backward_sequence(encoded.sequence, std::move(grad_sequence));
    }

    float mlm_backward(Encoded& encoded, const std::vector<uint32_t>& targets, const std::vector<size_t>& positions, float loss_scale) {
        std::vector<float> grad_hidden(encoded.sequence.hidden.size(), 0.0f); double total_loss = 0.0;
        for (size_t position_index : positions) {
            const float* hidden = encoded.sequence.hidden.data() + position_index * cfg.hidden_dim;
            std::vector<float> logits(cfg.vocab_size); float maximum = -std::numeric_limits<float>::infinity();
            for (int vocab = 0; vocab < cfg.vocab_size; ++vocab) { float sum = 0.0f; for (int h = 0; h < cfg.hidden_dim; ++h) sum += hidden[h] * token.value[static_cast<size_t>(vocab) * cfg.hidden_dim + h]; logits[vocab] = sum; maximum = std::max(maximum, sum); }
            float denominator = 0.0f; for (float& value : logits) { value = std::exp(value - maximum); denominator += value; }
            const uint32_t target = targets[position_index] % static_cast<uint32_t>(cfg.vocab_size);
            total_loss -= std::log(std::max(logits[target] / denominator, 1e-30f));
            for (int vocab = 0; vocab < cfg.vocab_size; ++vocab) {
                const float grad = loss_scale * (logits[vocab] / denominator - (static_cast<uint32_t>(vocab) == target ? 1.0f : 0.0f)) / static_cast<float>(positions.size());
                for (int h = 0; h < cfg.hidden_dim; ++h) {
                    grad_hidden[position_index * cfg.hidden_dim + h] += grad * token.value[static_cast<size_t>(vocab) * cfg.hidden_dim + h];
                    if (!adapters_enabled()) token.grad[static_cast<size_t>(vocab) * cfg.hidden_dim + h] += grad * hidden[h];
                }
            }
        }
        backward_sequence(encoded.sequence, std::move(grad_hidden));
        return static_cast<float>(total_loss / static_cast<double>(positions.size()));
    }

    void zero_grad() { for (Parameter* parameter : base_parameters()) parameter->zero_grad(); for (Parameter* parameter : adapter_parameters()) parameter->zero_grad(); }

    void update_parameter(Parameter& parameter, float lr, float scale) {
        if (parameter.value.empty()) return;
        const float correction1 = 1.0f - std::pow(cfg.beta1, static_cast<float>(step));
        const float correction2 = 1.0f - std::pow(cfg.beta2, static_cast<float>(step));
        for (size_t i = 0; i < parameter.value.size(); ++i) {
            const float grad = parameter.grad[i] * scale + cfg.weight_decay * parameter.value[i];
            parameter.first[i] = cfg.beta1 * parameter.first[i] + (1.0f - cfg.beta1) * grad;
            parameter.second[i] = cfg.beta2 * parameter.second[i] + (1.0f - cfg.beta2) * grad * grad;
            parameter.value[i] -= lr * (parameter.first[i] / correction1) / (std::sqrt(parameter.second[i] / correction2) + cfg.adam_eps);
        }
    }

    void update(float lr, float batch_scale) {
        ++step; float squared = 0.0f;
        const auto parameters = adapters_enabled() ? adapter_parameters() : base_parameters();
        for (const Parameter* parameter : parameters) for (float grad : parameter->grad) squared += grad * grad;
        const float norm = std::sqrt(squared) * batch_scale;
        const float clip = norm > cfg.grad_clip && cfg.grad_clip > 0.0f ? cfg.grad_clip / norm : 1.0f;
        for (Parameter* parameter : parameters) update_parameter(*parameter, lr, batch_scale * clip);
    }

    static void write_parameter(std::ofstream& out, const Parameter& parameter, const std::vector<float>* override_values = nullptr) {
        write_vector(out, override_values == nullptr ? parameter.value : *override_values);
    }
    static void read_parameter(std::ifstream& in, Parameter& parameter) {
        read_vector(in, parameter.value); parameter.grad.assign(parameter.value.size(), 0.0f); parameter.first.assign(parameter.value.size(), 0.0f); parameter.second.assign(parameter.value.size(), 0.0f);
    }

    std::vector<float> merged_weight(const Linear& linear) const {
        std::vector<float> result(static_cast<size_t>(linear.input_dim) * linear.output_dim);
        for (int out = 0; out < linear.output_dim; ++out) for (int in = 0; in < linear.input_dim; ++in) result[static_cast<size_t>(out) * linear.input_dim + in] = effective_weight(linear, out, in);
        return result;
    }

    void save(const fs::path& path, bool merged) const {
        fs::create_directories(path.parent_path()); std::ofstream out(path, std::ios::binary);
        const std::array<char, 8> magic{'N','F','N','E','M','B','2','\0'}; out.write(magic.data(), magic.size());
        const std::array<uint32_t, 16> header{2u,static_cast<uint32_t>(cfg.vocab_size),static_cast<uint32_t>(cfg.hidden_dim),static_cast<uint32_t>(cfg.output_dim),static_cast<uint32_t>(cfg.max_tokens),step,static_cast<uint32_t>(merged?0:(cfg.adapter_type=="qlora"?2:cfg.adapter_type=="lora"?1:0)),static_cast<uint32_t>(causal()?1:0),static_cast<uint32_t>(cfg.pooling=="cls"?1:cfg.pooling=="last"?2:0),static_cast<uint32_t>(cfg.normalize),static_cast<uint32_t>(cfg.num_layers),static_cast<uint32_t>(cfg.num_heads),static_cast<uint32_t>(cfg.intermediate_dim),static_cast<uint32_t>(cfg.lora_rank),static_cast<uint32_t>(cfg.activation=="gelu-tanh"),static_cast<uint32_t>(cfg.mask_token_id)};
        out.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(header.size()*sizeof(uint32_t)));
        out.write(reinterpret_cast<const char*>(&cfg.margin),sizeof(float)); out.write(reinterpret_cast<const char*>(&cfg.lora_alpha),sizeof(float)); out.write(reinterpret_cast<const char*>(&cfg.lora_dropout),sizeof(float)); out.write(reinterpret_cast<const char*>(&cfg.layer_norm_epsilon),sizeof(float));
        for (const Parameter* parameter : std::vector<const Parameter*>{&token,&position,&token_type,&embedding_ln_weight,&embedding_ln_bias}) write_parameter(out,*parameter);
        const std::vector<float> empty;
        for (const auto& layer : layers) {
            for (const Parameter* parameter : std::vector<const Parameter*>{&layer.ln1_weight,&layer.ln1_bias,&layer.ln2_weight,&layer.ln2_bias}) write_parameter(out,*parameter);
            for (const Linear* linear : std::vector<const Linear*>{&layer.query,&layer.key,&layer.value,&layer.attention_output,&layer.ff_input,&layer.ff_output}) {
                auto effective = merged ? merged_weight(*linear) : std::vector<float>{}; write_parameter(out,linear->weight,merged?&effective:nullptr); write_parameter(out,linear->bias);
                write_parameter(out,linear->lora_a,merged?&empty:nullptr);
                write_parameter(out,linear->lora_b,merged?&empty:nullptr);
            }
        }
        write_parameter(out,final_ln_weight); write_parameter(out,final_ln_bias);
        auto effective = merged ? merged_weight(projection) : std::vector<float>{}; write_parameter(out,projection.weight,merged?&effective:nullptr); write_parameter(out,projection.bias);
        write_parameter(out,projection.lora_a,merged?&empty:nullptr); write_parameter(out,projection.lora_b,merged?&empty:nullptr);
        if (!out) throw std::runtime_error("failed writing embedding checkpoint: " + path.string());
    }

    void load(const fs::path& path) {
        fs::path resolved = fs::is_directory(path) ? path / "embedding_model.bin" : path; std::ifstream in(resolved,std::ios::binary);
        if (!in) throw std::runtime_error("cannot open embedding checkpoint: " + resolved.string());
        std::array<char,8> magic{}; in.read(magic.data(),magic.size());
        if (std::string(magic.data(),7)!="NFNEMB2") throw std::runtime_error("embedding checkpoint is not a full transformer NFNEMB2 artifact: " + resolved.string());
        std::array<uint32_t,16> header{}; in.read(reinterpret_cast<char*>(header.data()),static_cast<std::streamsize>(header.size()*sizeof(uint32_t)));
        cfg.vocab_size=header[1];cfg.hidden_dim=header[2];cfg.output_dim=header[3];cfg.max_tokens=header[4];step=header[5];cfg.adapter_type=header[6]==2?"qlora":header[6]==1?"lora":"none";cfg.architecture=header[7]?"gpt-derived":"bert";cfg.pooling=header[8]==1?"cls":header[8]==2?"last":"mean";cfg.normalize=header[9]!=0;cfg.num_layers=header[10];cfg.num_heads=header[11];cfg.intermediate_dim=header[12];cfg.lora_rank=header[13];cfg.activation=header[14]?"gelu-tanh":"gelu";cfg.mask_token_id=header[15];
        in.read(reinterpret_cast<char*>(&cfg.margin),sizeof(float));in.read(reinterpret_cast<char*>(&cfg.lora_alpha),sizeof(float));in.read(reinterpret_cast<char*>(&cfg.lora_dropout),sizeof(float));in.read(reinterpret_cast<char*>(&cfg.layer_norm_epsilon),sizeof(float));
        layers.clear(); initialize(); step=header[5]; cfg.adapter_type=header[6]==2?"qlora":header[6]==1?"lora":"none";
        for (Parameter* parameter : std::vector<Parameter*>{&token,&position,&token_type,&embedding_ln_weight,&embedding_ln_bias}) read_parameter(in,*parameter);
        for (auto& layer : layers) {
            for (Parameter* parameter : std::vector<Parameter*>{&layer.ln1_weight,&layer.ln1_bias,&layer.ln2_weight,&layer.ln2_bias}) read_parameter(in,*parameter);
            for (Linear* linear : std::vector<Linear*>{&layer.query,&layer.key,&layer.value,&layer.attention_output,&layer.ff_input,&layer.ff_output}) { read_parameter(in,linear->weight);read_parameter(in,linear->bias);read_parameter(in,linear->lora_a);read_parameter(in,linear->lora_b);quantize_linear(*linear); }
        }
        read_parameter(in,final_ln_weight);read_parameter(in,final_ln_bias);read_parameter(in,projection.weight);read_parameter(in,projection.bias);read_parameter(in,projection.lora_a);read_parameter(in,projection.lora_b);quantize_linear(projection);
        if (!in) throw std::runtime_error("truncated embedding transformer checkpoint: " + resolved.string());
    }

    void save_optimizer(std::ofstream& out) const {
        for (const Parameter* parameter : base_parameters()) { write_vector(out,parameter->first);write_vector(out,parameter->second); }
        for (const Parameter* parameter : adapter_parameters()) { write_vector(out,parameter->first);write_vector(out,parameter->second); }
    }
    void load_optimizer(std::ifstream& in) {
        for (Parameter* parameter : base_parameters()) { read_vector(in,parameter->first);read_vector(in,parameter->second); }
        for (Parameter* parameter : adapter_parameters()) { read_vector(in,parameter->first);read_vector(in,parameter->second); }
    }
};
