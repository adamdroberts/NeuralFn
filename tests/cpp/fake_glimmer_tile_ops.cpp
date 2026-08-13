#include "tile_ops.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

float bf16(const std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16u);
}

std::uint16_t to_bf16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t rounding = 0x7fffu + ((bits >> 16u) & 1u);
    return static_cast<std::uint16_t>((bits + rounding) >> 16u);
}

bool valid(const NfnNativeTilePackedWeightDescriptorV1* weight) {
    if (weight == nullptr || weight->version != 1 || weight->flags != 0 ||
        weight->data == nullptr || weight->output_dim <= 0 || weight->input_dim <= 0)
        return false;
    std::int64_t row_bytes = 0;
    if (weight->encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_BF16) {
        row_bytes = weight->input_dim * 2;
    } else if (weight->encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_NF4_GROUP64) {
        row_bytes = ((weight->input_dim + 63) / 64) * 36;
    } else {
        return false;
    }
    return weight->row_stride_bytes == row_bytes &&
        weight->data_nbytes == row_bytes * weight->output_dim;
}

float value(const NfnNativeTilePackedWeightDescriptorV1& weight,
            std::int64_t row, std::int64_t col) {
    const auto* row_data = weight.data + row * weight.row_stride_bytes;
    if (weight.encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_BF16) {
        const auto* values = reinterpret_cast<const std::uint16_t*>(row_data);
        return bf16(values[col]);
    }
    constexpr float codebook[16] = {
        -1.0f, -0.6961928009986877f, -0.5250730514526367f,
        -0.39491748809814453f, -0.28444138169288635f,
        -0.18477343022823334f, -0.09105003625154495f, 0.0f,
        0.07958029955625534f, 0.16093020141124725f,
        0.24611230194568634f, 0.33791524171829224f,
        0.44070982933044434f, 0.5626170039176941f,
        0.7229568362236023f, 1.0f};
    const auto* group = row_data + (col / 64) * 36;
    float scale = 0.0f;
    std::memcpy(&scale, group, sizeof(scale));
    const std::uint8_t packed = group[4 + (col % 64) / 2];
    const std::uint8_t code = (col & 1) == 0 ? packed & 0x0f : packed >> 4;
    return scale * codebook[code];
}

}  // namespace

extern "C" {

int nfn_native_tile_ops_abi_version() { return 1; }
int nfn_native_tile_strict_math_abi_version() { return 1; }
int nfn_native_tile_packed_weight_abi_version() { return 1; }
int nfn_native_tile_glimmer_inference_abi_version() { return 1; }
int nfn_native_tile_glimmer_training_abi_version() { return 1; }
const char* nfn_native_tile_ops_error_string(int status) {
    return status == 0 ? "success" : "fake Tile failure";
}
int nfn_native_tile_packed_weight_validate_v1(
    const NfnNativeTilePackedWeightDescriptorV1* weight) {
    return valid(weight) ? 0 : 1;
}
int nfn_native_tile_linear_packed_weight_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* weight,
    const float* input,
    const float*,
    float* output,
    std::int64_t rows,
    bool has_bias) {
    if (!valid(weight) || input == nullptr || output == nullptr || rows <= 0 || has_bias) return 1;
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t out = 0; out < weight->output_dim; ++out) {
            double sum = 0.0;
            for (std::int64_t col = 0; col < weight->input_dim; ++col) {
                sum += input[row * weight->input_dim + col] * value(*weight, out, col);
            }
            output[row * weight->output_dim + out] = static_cast<float>(sum);
        }
    }
    return 0;
}
int nfn_native_tile_linear_backward_input_packed_weight_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* weight,
    const float* grad_output,
    float* grad_input,
    std::int64_t rows) {
    if (!valid(weight) || grad_output == nullptr || grad_input == nullptr || rows <= 0) return 1;
    for (std::int64_t row = 0; row < rows; ++row)
        for (std::int64_t input = 0; input < weight->input_dim; ++input) {
            double sum = 0;
            for (std::int64_t output = 0; output < weight->output_dim; ++output)
                sum += grad_output[row*weight->output_dim+output] * value(*weight, output, input);
            grad_input[row*weight->input_dim+input] = static_cast<float>(sum);
        }
    return 0;
}
int nfn_native_tile_linear_backward_weight_float32(
    const float* input, const float* grad_output, float* grad_weight,
    std::int64_t rows, std::int64_t input_dim, std::int64_t output_dim, void*) {
    if (!input || !grad_output || !grad_weight || rows <= 0) return 1;
    for (std::int64_t output = 0; output < output_dim; ++output)
        for (std::int64_t col = 0; col < input_dim; ++col) {
            double sum = 0;
            for (std::int64_t row = 0; row < rows; ++row)
                sum += input[row*input_dim+col] * grad_output[row*output_dim+output];
            grad_weight[output*input_dim+col] = static_cast<float>(sum);
        }
    return 0;
}
int nfn_native_tile_glimmer_embedding_gather_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* weight,
    std::int64_t token,
    float* output) {
    if (!valid(weight) || token < 0 || token >= weight->output_dim || output == nullptr) return 1;
    for (std::int64_t col = 0; col < weight->input_dim; ++col) output[col] = value(*weight, token, col);
    return 0;
}
int nfn_native_tile_glimmer_embedding_batch_i32_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* weight,
    const std::int32_t* token_ids,
    float* output,
    std::int64_t rows) {
    if (!valid(weight) || token_ids == nullptr || output == nullptr || rows <= 0) return 1;
    for (std::int64_t row = 0; row < rows; ++row) {
        if (token_ids[row] < 0 || token_ids[row] >= weight->output_dim) return 1;
        for (std::int64_t col = 0; col < weight->input_dim; ++col)
            output[row * weight->input_dim + col] = value(*weight, token_ids[row], col);
    }
    return 0;
}
int nfn_native_tile_glimmer_rms_norm_affine_float32_v1(
    const float* input,
    const NfnNativeTilePackedWeightDescriptorV1* weight,
    float* output,
    std::int64_t rows,
    std::int64_t width,
    float eps,
    bool centered,
    void*) {
    if (input == nullptr || output == nullptr || rows <= 0 || width <= 0 ||
        (weight != nullptr && (!valid(weight) || weight->output_dim != 1 || weight->input_dim != width))) return 1;
    for (std::int64_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (std::int64_t col = 0; col < width; ++col) {
            const double x = input[row * width + col]; sum += x * x;
        }
        const double inverse = 1.0 / std::sqrt(sum / width + eps);
        for (std::int64_t col = 0; col < width; ++col) {
            double scale = weight == nullptr ? 1.0 : value(*weight, 0, col) + (centered ? 1.0 : 0.0);
            output[row * width + col] = static_cast<float>(input[row * width + col] * inverse * scale);
        }
    }
    return 0;
}
int nfn_native_tile_glimmer_positioned_rope_float32_v1(
    float* query, float* key, std::int64_t q_heads, std::int64_t kv_heads,
    std::int64_t head_dim, std::int64_t position, float theta,
    std::uint32_t layout, void*) {
    if (query == nullptr || key == nullptr || head_dim <= 0 || head_dim % 2) return 1;
    for (std::int64_t head = 0; head < q_heads + kv_heads; ++head) {
        float* row = head < q_heads ? query + head * head_dim : key + (head - q_heads) * head_dim;
        for (std::int64_t pair = 0; pair < head_dim / 2; ++pair) {
            const std::int64_t first = layout == 1 ? 2 * pair : pair;
            const std::int64_t second = layout == 1 ? 2 * pair + 1 : pair + head_dim / 2;
            const double angle = position / std::pow(theta, static_cast<double>(2 * pair) / head_dim);
            const float lhs = row[first], rhs = row[second];
            row[first] = static_cast<float>(lhs * std::cos(angle) - rhs * std::sin(angle));
            row[second] = static_cast<float>(rhs * std::cos(angle) + lhs * std::sin(angle));
        }
    }
    return 0;
}
int nfn_native_tile_glimmer_gqa_decode_float32_v1(
    const NfnNativeTileGlimmerGqaDecodeDescriptorV1* d) {
    if (d == nullptr || d->version != 1 || d->query == nullptr || d->output == nullptr) return 1;
    for (std::int64_t qh = 0; qh < d->query_heads; ++qh) {
        const std::int64_t kh = qh * d->kv_heads / d->query_heads;
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::int64_t pos = d->first_key_position; pos <= d->position; ++pos) {
            double score = 0.0;
            for (std::int64_t dim = 0; dim < d->head_dim; ++dim) {
                const auto index = (pos % d->cache_capacity) * d->cache_row_stride + kh * d->head_dim + dim;
                const float k = pos == d->position ? d->current_key[kh * d->head_dim + dim] : bf16(d->key_cache_bf16[index]);
                score += d->query[qh * d->head_dim + dim] * k;
            }
            maximum = std::max(maximum, score * d->scale);
        }
        double denominator = 0.0;
        for (std::int64_t pos = d->first_key_position; pos <= d->position; ++pos) {
            double score = 0.0;
            for (std::int64_t dim = 0; dim < d->head_dim; ++dim) {
                const auto index = (pos % d->cache_capacity) * d->cache_row_stride + kh * d->head_dim + dim;
                const float k = pos == d->position ? d->current_key[kh * d->head_dim + dim] : bf16(d->key_cache_bf16[index]);
                score += d->query[qh * d->head_dim + dim] * k;
            }
            denominator += std::exp(score * d->scale - maximum);
        }
        for (std::int64_t dim = 0; dim < d->head_dim; ++dim) {
            double result = 0.0;
            for (std::int64_t pos = d->first_key_position; pos <= d->position; ++pos) {
                double score = 0.0;
                for (std::int64_t inner = 0; inner < d->head_dim; ++inner) {
                    const auto index = (pos % d->cache_capacity) * d->cache_row_stride + kh * d->head_dim + inner;
                    const float k = pos == d->position ? d->current_key[kh * d->head_dim + inner] : bf16(d->key_cache_bf16[index]);
                    score += d->query[qh * d->head_dim + inner] * k;
                }
                const auto index = (pos % d->cache_capacity) * d->cache_row_stride + kh * d->head_dim + dim;
                const float v = pos == d->position ? d->current_value[kh * d->head_dim + dim] : bf16(d->value_cache_bf16[index]);
                result += std::exp(score * d->scale - maximum) / denominator * v;
            }
            d->output[qh * d->head_dim + dim] = static_cast<float>(result);
        }
    }
    return 0;
}
int nfn_native_tile_glimmer_cache_commit_bf16_v1(
    const NfnNativeTileGlimmerCacheCommitDescriptorV1* d) {
    if (d == nullptr || d->version != 1) return 1;
    const std::int64_t width = d->kv_heads * d->head_dim;
    const std::int64_t offset = (d->position % d->cache_capacity) * d->cache_row_stride;
    for (std::int64_t index = 0; index < width; ++index) {
        d->key_cache_bf16[offset + index] = to_bf16(d->current_key[index]);
        d->value_cache_bf16[offset + index] = to_bf16(d->current_value[index]);
    }
    return 0;
}
int nfn_native_tile_dflash_block_attention_float32_v1(
    const NfnNativeTileDFlashBlockAttentionDescriptorV1* d) {
    if (d == nullptr || d->version != 1 || d->query == nullptr ||
        d->block_key == nullptr || d->block_value == nullptr ||
        d->key_cache_bf16 == nullptr || d->value_cache_bf16 == nullptr ||
        d->output == nullptr || d->query_rows <= 0 || d->block_rows <= 0 ||
        d->query_rows > d->block_rows || d->query_heads <= 0 || d->kv_heads <= 0 ||
        d->query_heads % d->kv_heads != 0 || d->head_dim <= 0 ||
        d->context_length < 0 || d->sliding_window <= 0 ||
        d->cache_capacity < d->sliding_window) return 1;
    for (std::int64_t query_row = 0; query_row < d->query_rows; ++query_row) {
        const std::int64_t query_position = d->context_length + query_row;
        const bool causal =
            (d->flags & NFN_NATIVE_TILE_BLOCK_ATTENTION_CAUSAL) != 0;
        const std::int64_t first = std::max<std::int64_t>(
            0, query_position - d->sliding_window + (causal ? 1 : 0));
        const std::int64_t last = causal
            ? query_position
            : std::min<std::int64_t>(
                  d->context_length + d->block_rows - 1,
                  query_position + d->sliding_window);
        for (std::int64_t qh = 0; qh < d->query_heads; ++qh) {
            const std::int64_t kh = qh * d->kv_heads / d->query_heads;
            const float* query = d->query +
                (query_row * d->query_heads + qh) * d->head_dim;
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::int64_t pos = first; pos <= last; ++pos) {
                double score = 0.0;
                for (std::int64_t dim = 0; dim < d->head_dim; ++dim) {
                    const bool block = pos >= d->context_length;
                    const std::int64_t index = block
                        ? ((pos - d->context_length) * d->kv_heads + kh) * d->head_dim + dim
                        : (pos % d->cache_capacity) * d->cache_row_stride + kh * d->head_dim + dim;
                    const float key = block ? d->block_key[index] : bf16(d->key_cache_bf16[index]);
                    score += query[dim] * key;
                }
                maximum = std::max(maximum, score * d->scale);
            }
            double denominator = 0.0;
            for (std::int64_t pos = first; pos <= last; ++pos) {
                double score = 0.0;
                for (std::int64_t dim = 0; dim < d->head_dim; ++dim) {
                    const bool block = pos >= d->context_length;
                    const std::int64_t index = block
                        ? ((pos - d->context_length) * d->kv_heads + kh) * d->head_dim + dim
                        : (pos % d->cache_capacity) * d->cache_row_stride + kh * d->head_dim + dim;
                    score += query[dim] * (block ? d->block_key[index] : bf16(d->key_cache_bf16[index]));
                }
                denominator += std::exp(score * d->scale - maximum);
            }
            for (std::int64_t dim = 0; dim < d->head_dim; ++dim) {
                double sum = 0.0;
                for (std::int64_t pos = first; pos <= last; ++pos) {
                    double score = 0.0;
                    for (std::int64_t inner = 0; inner < d->head_dim; ++inner) {
                        const bool block = pos >= d->context_length;
                        const std::int64_t index = block
                            ? ((pos - d->context_length) * d->kv_heads + kh) * d->head_dim + inner
                            : (pos % d->cache_capacity) * d->cache_row_stride + kh * d->head_dim + inner;
                        score += query[inner] *
                            (block ? d->block_key[index] : bf16(d->key_cache_bf16[index]));
                    }
                    const bool block = pos >= d->context_length;
                    const std::int64_t value_index = block
                        ? ((pos - d->context_length) * d->kv_heads + kh) * d->head_dim + dim
                        : (pos % d->cache_capacity) * d->cache_row_stride + kh * d->head_dim + dim;
                    const float value_row = block
                        ? d->block_value[value_index]
                        : bf16(d->value_cache_bf16[value_index]);
                    sum += std::exp(score * d->scale - maximum) / denominator * value_row;
                }
                d->output[(query_row * d->query_heads + qh) * d->head_dim + dim] =
                    static_cast<float>(sum);
            }
        }
    }
    return 0;
}
int nfn_native_tile_glimmer_sigmoid_gate_float32_v1(
    const float* values, const float* gate, float* output, std::int64_t count, void*) {
    for (std::int64_t i = 0; i < count; ++i) output[i] = values[i] / (1.0f + std::exp(-gate[i]));
    return 0;
}
int nfn_native_tile_glimmer_logit_transform_float32_v1(
    float* logits, std::int64_t count, float multiplier, float softcap, void*) {
    for (std::int64_t i = 0; i < count; ++i) logits[i] = softcap * std::tanh(multiplier * logits[i] / softcap);
    return 0;
}
int nfn_native_tile_glimmer_attention_forward_float32_v1(
    const NfnNativeTileGlimmerAttentionTrainingDescriptorV1* d) {
    if (d == nullptr || d->version != 1 || d->query == nullptr || d->key == nullptr ||
        d->value == nullptr || d->output == nullptr || d->logsumexp == nullptr ||
        d->query_heads <= 0 || d->kv_heads <= 0 || d->query_heads % d->kv_heads) return 1;
    for (std::int64_t batch = 0; batch < d->batch_size; ++batch) {
        for (std::int64_t query_pos = 0; query_pos < d->sequence_length; ++query_pos) {
            const auto first = d->window > 0
                ? std::max<std::int64_t>(0, query_pos - d->window + 1) : 0;
            const auto last = (d->flags & NFN_NATIVE_TILE_GLIMMER_TRAIN_CAUSAL)
                ? query_pos : d->sequence_length - 1;
            for (std::int64_t qh = 0; qh < d->query_heads; ++qh) {
                const auto kh = qh * d->kv_heads / d->query_heads;
                const auto qbase = ((batch * d->sequence_length + query_pos) * d->query_heads + qh) * d->head_dim;
                double maximum = -std::numeric_limits<double>::infinity();
                for (std::int64_t key_pos = first; key_pos <= last; ++key_pos) {
                    if (d->sequence_ids &&
                        d->sequence_ids[batch*d->sequence_length+key_pos] !=
                            d->sequence_ids[batch*d->sequence_length+query_pos]) continue;
                    const auto kbase = ((batch * d->sequence_length + key_pos) * d->kv_heads + kh) * d->head_dim;
                    double score = 0;
                    for (std::int64_t dim = 0; dim < d->head_dim; ++dim) score += d->query[qbase + dim] * d->key[kbase + dim];
                    maximum = std::max(maximum, score * d->scale);
                }
                double denominator = 0;
                for (std::int64_t key_pos = first; key_pos <= last; ++key_pos) {
                    if (d->sequence_ids &&
                        d->sequence_ids[batch*d->sequence_length+key_pos] !=
                            d->sequence_ids[batch*d->sequence_length+query_pos]) continue;
                    const auto kbase = ((batch * d->sequence_length + key_pos) * d->kv_heads + kh) * d->head_dim;
                    double score = 0;
                    for (std::int64_t dim = 0; dim < d->head_dim; ++dim) score += d->query[qbase + dim] * d->key[kbase + dim];
                    denominator += std::exp(score * d->scale - maximum);
                }
                d->logsumexp[(batch * d->sequence_length + query_pos) * d->query_heads + qh] =
                    static_cast<float>(maximum + std::log(denominator));
                for (std::int64_t dim = 0; dim < d->head_dim; ++dim) {
                    double out = 0;
                    for (std::int64_t key_pos = first; key_pos <= last; ++key_pos) {
                        if (d->sequence_ids &&
                            d->sequence_ids[batch*d->sequence_length+key_pos] !=
                                d->sequence_ids[batch*d->sequence_length+query_pos]) continue;
                        const auto kbase = ((batch * d->sequence_length + key_pos) * d->kv_heads + kh) * d->head_dim;
                        double score = 0;
                        for (std::int64_t inner = 0; inner < d->head_dim; ++inner) score += d->query[qbase + inner] * d->key[kbase + inner];
                        out += std::exp(score * d->scale - maximum) / denominator * d->value[kbase + dim];
                    }
                    d->output[qbase + dim] = static_cast<float>(out);
                }
            }
        }
    }
    return 0;
}
int nfn_native_tile_glimmer_attention_backward_float32_v1(
    const NfnNativeTileGlimmerAttentionTrainingDescriptorV1* d) {
    if (d == nullptr || d->grad_output == nullptr || d->grad_query == nullptr ||
        d->grad_key == nullptr || d->grad_value == nullptr) return 1;
    std::fill(d->grad_query, d->grad_query + d->batch_size * d->sequence_length * d->query_heads * d->head_dim, 0.0f);
    std::fill(d->grad_key, d->grad_key + d->batch_size * d->sequence_length * d->kv_heads * d->head_dim, 0.0f);
    std::fill(d->grad_value, d->grad_value + d->batch_size * d->sequence_length * d->kv_heads * d->head_dim, 0.0f);
    for (std::int64_t batch = 0; batch < d->batch_size; ++batch) {
        for (std::int64_t query_pos = 0; query_pos < d->sequence_length; ++query_pos) {
            const auto first = d->window > 0 ? std::max<std::int64_t>(0, query_pos - d->window + 1) : 0;
            const auto last = (d->flags & NFN_NATIVE_TILE_GLIMMER_TRAIN_CAUSAL) ? query_pos : d->sequence_length - 1;
            for (std::int64_t qh = 0; qh < d->query_heads; ++qh) {
                const auto kh = qh * d->kv_heads / d->query_heads;
                const auto qbase = ((batch * d->sequence_length + query_pos) * d->query_heads + qh) * d->head_dim;
                double out_dot = 0;
                for (std::int64_t dim = 0; dim < d->head_dim; ++dim) out_dot += d->grad_output[qbase + dim] * d->output[qbase + dim];
                for (std::int64_t key_pos = first; key_pos <= last; ++key_pos) {
                    if (d->sequence_ids &&
                        d->sequence_ids[batch*d->sequence_length+key_pos] !=
                            d->sequence_ids[batch*d->sequence_length+query_pos]) continue;
                    const auto kbase = ((batch * d->sequence_length + key_pos) * d->kv_heads + kh) * d->head_dim;
                    double score = 0, dp = 0;
                    for (std::int64_t dim = 0; dim < d->head_dim; ++dim) {
                        score += d->query[qbase + dim] * d->key[kbase + dim];
                        dp += d->grad_output[qbase + dim] * d->value[kbase + dim];
                    }
                    const double p = std::exp(score * d->scale - d->logsumexp[(batch * d->sequence_length + query_pos) * d->query_heads + qh]);
                    const double ds = p * (dp - out_dot);
                    for (std::int64_t dim = 0; dim < d->head_dim; ++dim) {
                        d->grad_query[qbase + dim] += static_cast<float>(ds * d->key[kbase + dim] * d->scale);
                        d->grad_key[kbase + dim] += static_cast<float>(ds * d->query[qbase + dim] * d->scale);
                        d->grad_value[kbase + dim] += static_cast<float>(p * d->grad_output[qbase + dim]);
                    }
                }
            }
        }
    }
    return 0;
}
int nfn_native_tile_glimmer_rms_norm_backward_float32_v1(
    const NfnNativeTileGlimmerRmsNormBackwardDescriptorV1* d) {
    if (d == nullptr || d->input == nullptr || d->grad_output == nullptr || d->grad_input == nullptr) return 1;
    if (d->grad_weight) std::fill(d->grad_weight, d->grad_weight + d->width, 0.0f);
    for (std::int64_t row = 0; row < d->rows; ++row) {
        double sq = 0;
        for (std::int64_t col = 0; col < d->width; ++col) sq += d->input[row*d->width+col]*d->input[row*d->width+col];
        const double inv = 1.0 / std::sqrt(sq / d->width + d->eps);
        double dot = 0;
        for (std::int64_t col = 0; col < d->width; ++col) {
            const double scale = d->weight ? value(*d->weight, 0, col) + (d->centered ? 1.0 : 0.0) : 1.0;
            dot += d->grad_output[row*d->width+col] * scale * d->input[row*d->width+col];
            if (d->grad_weight) d->grad_weight[col] += static_cast<float>(d->grad_output[row*d->width+col] * d->input[row*d->width+col] * inv);
        }
        for (std::int64_t col = 0; col < d->width; ++col) {
            const double scale = d->weight ? value(*d->weight, 0, col) + (d->centered ? 1.0 : 0.0) : 1.0;
            d->grad_input[row*d->width+col] = static_cast<float>(d->grad_output[row*d->width+col]*scale*inv - d->input[row*d->width+col]*dot*inv*inv*inv/d->width);
        }
    }
    return 0;
}
int nfn_native_tile_glimmer_positioned_rope_batch_float32_v1(
    float* query, float* key, std::int64_t rows, std::int64_t qh, std::int64_t kh,
    std::int64_t hd, std::int64_t start, float theta, std::uint32_t layout,
    bool inverse, void*) {
    for (std::int64_t row = 0; row < rows; ++row) {
        const auto pos = inverse ? -(start + row) : start + row;
        if (nfn_native_tile_glimmer_positioned_rope_float32_v1(
                query + row*qh*hd, key + row*kh*hd, qh, kh, hd, pos, theta, layout, nullptr) != 0) return 1;
    }
    return 0;
}
int nfn_native_tile_glimmer_sigmoid_gate_backward_float32_v1(
    const float* values, const float* gate, const float* grad_output,
    float* grad_values, float* grad_gate, std::int64_t count, void*) {
    for (std::int64_t i = 0; i < count; ++i) {
        const float s = 1.0f / (1.0f + std::exp(-gate[i]));
        grad_values[i] = grad_output[i] * s;
        grad_gate[i] = grad_output[i] * values[i] * s * (1.0f-s);
    }
    return 0;
}
int nfn_native_tile_glimmer_logit_transform_backward_float32_v1(
    const float* transformed, const float* grad_transformed, float* grad_raw,
    std::int64_t count, float multiplier, float softcap, void*) {
    for (std::int64_t i = 0; i < count; ++i) {
        const float ratio = transformed[i] / softcap;
        grad_raw[i] = grad_transformed[i] * multiplier * (1.0f-ratio*ratio);
    }
    return 0;
}
int nfn_native_tile_glimmer_masked_cross_entropy_i32_float32_v1(
    const NfnNativeTileGlimmerMaskedCeDescriptorV1* d) {
    if (d == nullptr) return 1;
    for (std::int64_t row = 0; row < d->rows; ++row) {
        const auto target = d->targets[row];
        const float mask = d->loss_mask ? d->loss_mask[row] : 1.0f;
        if (target == d->ignore_index || !(mask > 0)) {
            d->row_loss[row] = 0;
            if (d->grad_transformed_logits) std::fill(d->grad_transformed_logits + row*d->vocab_size, d->grad_transformed_logits + (row+1)*d->vocab_size, 0.0f);
            continue;
        }
        const float* logits = d->transformed_logits + row*d->vocab_size;
        const float maximum = *std::max_element(logits, logits+d->vocab_size);
        double denom = 0;
        for (std::int64_t col = 0; col < d->vocab_size; ++col) denom += std::exp(logits[col]-maximum);
        const double logden = maximum + std::log(denom);
        d->row_loss[row] = static_cast<float>(mask*(logden-logits[target]));
        if (d->grad_transformed_logits) for (std::int64_t col = 0; col < d->vocab_size; ++col)
            d->grad_transformed_logits[row*d->vocab_size+col] = static_cast<float>(d->grad_scale*mask*(std::exp(logits[col]-logden)-(col==target)));
    }
    return 0;
}
int nfn_native_tile_token_embedding_backward_weight_i32_float32(
    const std::int32_t* ids, const float* grad_output, float* grad_weight,
    std::int64_t rows, std::int64_t vocab, std::int64_t dim, void*) {
    std::fill(grad_weight, grad_weight+vocab*dim, 0.0f);
    for (std::int64_t row = 0; row < rows; ++row)
        for (std::int64_t col = 0; col < dim; ++col)
            grad_weight[ids[row]*dim+col] += grad_output[row*dim+col];
    return 0;
}
int nfn_native_tile_glimmer_adamw_bf16_float32_v1(
    std::uint16_t* parameter, const float* gradient, float* avg, float* avg_sq,
    std::int64_t count, float lr, float beta1, float beta2, float eps,
    float weight_decay, std::int64_t step, float gradient_scale, void*) {
    const double bc1 = 1.0-std::pow(beta1, static_cast<double>(step));
    const double bc2 = 1.0-std::pow(beta2, static_cast<double>(step));
    for (std::int64_t i = 0; i < count; ++i) {
        const float g = gradient[i]*gradient_scale;
        avg[i] = beta1*avg[i]+(1-beta1)*g;
        avg_sq[i] = beta2*avg_sq[i]+(1-beta2)*g*g;
        float p = bf16(parameter[i]);
        p -= lr*(static_cast<float>((avg[i]/bc1)/(std::sqrt(avg_sq[i]/bc2)+eps))+weight_decay*p);
        parameter[i] = to_bf16(p);
    }
    return 0;
}
int nfn_native_tile_swiglu_float32(
    const float* gate, const float* up, float* output, std::int64_t count, void*) {
    for (std::int64_t i = 0; i < count; ++i) output[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
    return 0;
}
int nfn_native_tile_swiglu_backward_float32(
    const float* gate, const float* up, const float* grad_output,
    float* grad_gate, float* grad_up, std::int64_t count, void*) {
    for (std::int64_t i = 0; i < count; ++i) {
        const float sigmoid = 1.0f / (1.0f + std::exp(-gate[i]));
        const float silu = gate[i] * sigmoid;
        grad_gate[i] = grad_output[i] * up[i] *
            (sigmoid + gate[i] * sigmoid * (1.0f - sigmoid));
        grad_up[i] = grad_output[i] * silu;
    }
    return 0;
}
int nfn_native_tile_add_float32(
    const float* lhs, const float* rhs, float* output, std::int64_t count, void*) {
    for (std::int64_t i = 0; i < count; ++i) output[i] = lhs[i] + rhs[i];
    return 0;
}
int nfn_native_tile_scale_inplace_float32(float* values, std::int64_t count, float scale, void*) {
    for (std::int64_t i = 0; i < count; ++i) values[i] *= scale;
    return 0;
}
int nfn_native_tile_copy_float32(
    const float* source, float* destination, std::int64_t count, void*) {
    std::copy(source, source + count, destination);
    return 0;
}
int nfn_native_tile_dropout_forward_float32(
    const float* input, float* output, std::int64_t count, float dropout_p,
    std::int64_t seed, void*) {
    if (!input || !output || count <= 0 || dropout_p < 0.0f || dropout_p >= 1.0f) return 1;
    const double scale = 1.0 / (1.0 - dropout_p);
    for (std::int64_t i = 0; i < count; ++i) {
        std::uint64_t value = static_cast<std::uint64_t>(seed) ^ static_cast<std::uint64_t>(i);
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        value ^= value >> 31U;
        const double unit = static_cast<double>(value >> 11U) *
            (1.0 / 9007199254740992.0);
        output[i] = unit >= dropout_p ? static_cast<float>(input[i] * scale) : 0.0f;
    }
    return 0;
}
int nfn_native_tile_dropout_backward_float32(
    const float* grad_output, float* grad_input, std::int64_t count, float dropout_p,
    std::int64_t seed, void* stream) {
    return nfn_native_tile_dropout_forward_float32(
        grad_output, grad_input, count, dropout_p, seed, stream);
}

}
