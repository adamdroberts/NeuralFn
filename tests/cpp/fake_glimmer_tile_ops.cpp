#include "tile_ops.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

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
    if (weight->encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_F32) {
        row_bytes = weight->input_dim * 4;
    } else if (weight->encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_BF16) {
        row_bytes = weight->input_dim * 2;
    } else if (weight->encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K) {
        if (weight->input_dim % 256 != 0) return false;
        row_bytes = weight->input_dim / 256 * 144;
    } else if (weight->encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q5_K) {
        if (weight->input_dim % 256 != 0) return false;
        row_bytes = weight->input_dim / 256 * 176;
    } else if (weight->encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q6_K) {
        if (weight->input_dim % 256 != 0) return false;
        row_bytes = weight->input_dim / 256 * 210;
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
    if (weight.encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_F32) {
        float result = 0.0f;
        std::memcpy(&result, row_data + col * 4, sizeof(result));
        return result;
    }
    if (weight.encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_BF16) {
        const auto* values = reinterpret_cast<const std::uint16_t*>(row_data);
        return bf16(values[col]);
    }
    // Tiny K-Quant integration fixtures use canonical-sized all-zero blocks.
    // The production dequantizer is verified independently; returning zero
    // here keeps this fake runtime a compact orchestration oracle while still
    // requiring the exact Q4_K/Q5_K/Q6_K descriptor strides and dispatch.
    if (weight.encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K ||
        weight.encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q5_K ||
        weight.encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_Q6_K) {
        return 0.0f;
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
int nfn_native_tile_glimmer_vision_abi_version() { return 1; }
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
    const float* bias,
    float* output,
    std::int64_t rows,
    bool has_bias) {
    if (!valid(weight) || input == nullptr || output == nullptr || rows <= 0 ||
        (has_bias && bias == nullptr)) return 1;
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t out = 0; out < weight->output_dim; ++out) {
            double sum = has_bias ? bias[out] : 0.0;
            for (std::int64_t col = 0; col < weight->input_dim; ++col) {
                sum += input[row * weight->input_dim + col] * value(*weight, out, col);
            }
            output[row * weight->output_dim + out] = static_cast<float>(sum);
        }
    }
    return 0;
}
int nfn_native_tile_argmax_rows_float32_v1(
    const float* values,
    std::int64_t* output_indices,
    float* output_values,
    std::int64_t rows,
    std::int64_t width,
    void*) {
    if (values == nullptr || output_indices == nullptr || output_values == nullptr ||
        rows <= 0 || width <= 0) return 1;
    for (std::int64_t row = 0; row < rows; ++row) {
        std::int64_t best_index = 0;
        float best_value = values[row * width];
        for (std::int64_t column = 1; column < width; ++column) {
            const float candidate = values[row * width + column];
            if (candidate > best_value) {
                best_value = candidate;
                best_index = column;
            }
        }
        output_indices[row] = best_index;
        output_values[row] = best_value;
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
int nfn_native_tile_glimmer_rms_norm_affine_capture_residual_float32_v1(
    const float* input,
    const NfnNativeTilePackedWeightDescriptorV1* weight,
    float* output,
    float* residual_output,
    std::int64_t rows,
    std::int64_t width,
    float eps,
    bool centered,
    void* stream) {
    if (input == nullptr || residual_output == nullptr || rows <= 0 || width <= 0) return 1;
    std::copy(input, input + rows * width, residual_output);
    return nfn_native_tile_glimmer_rms_norm_affine_float32_v1(
        input, weight, output, rows, width, eps, centered, stream);
}
int nfn_native_tile_glimmer_rms_norm_affine_add_residual_float32_v1(
    const float* input,
    const NfnNativeTilePackedWeightDescriptorV1* weight,
    const float* residual_input,
    float* output,
    std::int64_t rows,
    std::int64_t width,
    float eps,
    bool centered,
    void* stream) {
    if (residual_input == nullptr) return 1;
    const int status = nfn_native_tile_glimmer_rms_norm_affine_float32_v1(
        input, weight, output, rows, width, eps, centered, stream);
    if (status != 0) return status;
    for (std::int64_t index = 0; index < rows * width; ++index) {
        output[index] += residual_input[index];
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
int nfn_native_tile_glimmer_cache_commit_layers_bf16_v1(
    const NfnNativeTileGlimmerCacheCommitLayersDescriptorV1* d) {
    if (d == nullptr || d->version != 1 || d->staged_keys == nullptr ||
        d->staged_values == nullptr || d->layers == nullptr ||
        d->layer_count <= 0 || d->layer_count > 64 || d->source_rows <= 0 ||
        d->rows <= 0 || d->rows > d->source_rows || d->kv_heads <= 0 ||
        d->head_dim <= 0 || d->position < 0) {
        return 1;
    }
    const std::int64_t width = d->kv_heads * d->head_dim;
    if (d->source_layer_stride < d->source_rows * width) return 1;
    for (std::int64_t layer_index = 0; layer_index < d->layer_count;
         ++layer_index) {
        const NfnNativeTileGlimmerCacheLayerV1& layer = d->layers[layer_index];
        if (layer.key_cache_bf16 == nullptr ||
            layer.value_cache_bf16 == nullptr || layer.cache_capacity <= 0 ||
            layer.cache_row_stride < width) {
            return 1;
        }
        for (std::int64_t row = 0; row < d->rows; ++row) {
            const std::int64_t source =
                layer_index * d->source_layer_stride + row * width;
            const std::int64_t target =
                ((d->position + row) % layer.cache_capacity) *
                layer.cache_row_stride;
            for (std::int64_t column = 0; column < width; ++column) {
                layer.key_cache_bf16[target + column] =
                    to_bf16(d->staged_keys[source + column]);
                layer.value_cache_bf16[target + column] =
                    to_bf16(d->staged_values[source + column]);
            }
        }
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
int nfn_native_tile_glimmer_vision_prepare_float32_v1(
    const NfnNativeTileGlimmerVisionPrepareDescriptorV1* d) {
    if (d == nullptr || d->version != 1 || d->projected == nullptr ||
        d->position_table == nullptr || d->corner_indices == nullptr ||
        d->corner_weights == nullptr || d->permutation == nullptr ||
        d->output == nullptr || d->rows <= 0 || d->width <= 0) return 1;
    for (std::int64_t row = 0; row < d->rows; ++row) {
        const std::int32_t source = d->permutation[row];
        if (source < 0 || source >= d->rows) return 1;
        for (std::int64_t dim = 0; dim < d->width; ++dim) {
            float result = d->projected[static_cast<std::int64_t>(source)*d->width+dim];
            for (int corner = 0; corner < 4; ++corner) {
                const auto metadata = static_cast<std::int64_t>(source)*4+corner;
                const std::int32_t table = d->corner_indices[metadata];
                if (table >= 0) {
                    if (table >= d->position_rows) return 1;
                    result += d->corner_weights[metadata]*
                        d->position_table[static_cast<std::int64_t>(table)*d->width+dim];
                }
            }
            d->output[row*d->width+dim] = result;
        }
    }
    return 0;
}
int nfn_native_tile_glimmer_vision_layer_norm_float32_v1(
    const float* input, const float* weight, const float* bias, float* output,
    std::int64_t rows, std::int64_t width, float eps, void*) {
    if (input == nullptr || weight == nullptr || bias == nullptr || output == nullptr ||
        rows <= 0 || width <= 0 || !(eps > 0.0f)) return 1;
    for (std::int64_t row = 0; row < rows; ++row) {
        double mean = 0.0;
        for (std::int64_t dim = 0; dim < width; ++dim) mean += input[row*width+dim];
        mean /= width;
        double variance = 0.0;
        for (std::int64_t dim = 0; dim < width; ++dim) {
            const double centered = input[row*width+dim]-mean;
            variance += centered*centered;
        }
        const double inverse = 1.0/std::sqrt(variance/width+eps);
        for (std::int64_t dim = 0; dim < width; ++dim) {
            output[row*width+dim] = static_cast<float>(
                (input[row*width+dim]-mean)*inverse*weight[dim]+bias[dim]);
        }
    }
    return 0;
}
int nfn_native_tile_glimmer_vision_attention_float32_v1(
    const NfnNativeTileGlimmerVisionAttentionDescriptorV1* d) {
    if (d == nullptr || d->version != 1 || d->query == nullptr ||
        d->key == nullptr || d->value == nullptr || d->output == nullptr ||
        d->position_width == nullptr || d->position_height == nullptr ||
        d->row_begin == nullptr || d->row_end == nullptr || d->rows <= 0 ||
        d->heads <= 0 || d->head_dim <= 0 || d->head_dim % 4 != 0) return 1;
    const auto rotated = [&](const float* values, std::int64_t row,
                             std::int64_t head, std::int64_t dim) {
        const auto* data = values+(row*d->heads+head)*d->head_dim;
        const std::int64_t spatial = d->head_dim/2;
        const std::int64_t frequencies = spatial/2;
        if (d->interleaved_rope) {
            const std::int64_t pair = dim/2;
            const float position = static_cast<float>(pair < frequencies
                ? d->position_width[row] : d->position_height[row]);
            const auto frequency = pair%frequencies;
            const double angle = position/std::pow(
                d->rope_theta, static_cast<double>(frequency*2)/spatial);
            const float first = data[pair*2], second = data[pair*2+1];
            return static_cast<float>((dim&1)
                ? second*std::cos(angle)+first*std::sin(angle)
                : first*std::cos(angle)-second*std::sin(angle));
        }
        const auto frequency = dim%frequencies;
        const bool width_axis = dim < frequencies ||
            (dim >= spatial && dim < spatial+frequencies);
        const float position = static_cast<float>(width_axis
            ? d->position_width[row] : d->position_height[row]);
        const double angle = position/std::pow(
            d->rope_theta, static_cast<double>(frequency*2)/spatial);
        const auto paired = dim < d->head_dim/2
            ? dim+d->head_dim/2 : dim-d->head_dim/2;
        const float other = dim < d->head_dim/2 ? -data[paired] : data[paired];
        return static_cast<float>(data[dim]*std::cos(angle)+other*std::sin(angle));
    };
    for (std::int64_t row = 0; row < d->rows; ++row) {
        const auto begin = d->row_begin[row], end = d->row_end[row];
        if (begin < 0 || end <= begin || end > d->rows) return 1;
        for (std::int64_t head = 0; head < d->heads; ++head) {
            std::vector<double> scores(static_cast<std::size_t>(end-begin));
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::int64_t key_row = begin; key_row < end; ++key_row) {
                double score = 0.0;
                for (std::int64_t dim = 0; dim < d->head_dim; ++dim) {
                    score += rotated(d->query,row,head,dim)*
                        rotated(d->key,key_row,head,dim);
                }
                score /= std::sqrt(static_cast<double>(d->head_dim));
                scores[static_cast<std::size_t>(key_row-begin)] = score;
                maximum = std::max(maximum, score);
            }
            double denominator = 0.0;
            for (double score : scores) denominator += std::exp(score-maximum);
            for (std::int64_t dim = 0; dim < d->head_dim; ++dim) {
                double result = 0.0;
                for (std::int64_t key_row = begin; key_row < end; ++key_row) {
                    result += std::exp(scores[static_cast<std::size_t>(key_row-begin)]-maximum)/
                        denominator*d->value[(key_row*d->heads+head)*d->head_dim+dim];
                }
                d->output[(row*d->heads+head)*d->head_dim+dim] = static_cast<float>(result);
            }
        }
    }
    return 0;
}
int nfn_native_tile_glimmer_vision_pixel_shuffle_float32_v1(
    const NfnNativeTileGlimmerVisionPixelShuffleDescriptorV1* d) {
    if (d == nullptr || d->version != 1 || d->reordered_hidden == nullptr ||
        d->source_rows == nullptr || d->output == nullptr || d->merged_rows <= 0 ||
        d->hidden_size <= 0 || d->merge_area <= 0) return 1;
    for (std::int64_t row = 0; row < d->merged_rows; ++row) {
        for (std::int64_t dim = 0; dim < d->hidden_size; ++dim) {
            for (std::int64_t slot = 0; slot < d->merge_area; ++slot) {
                const auto source = d->source_rows[row*d->merge_area+slot];
                if (source < 0) return 1;
                d->output[(row*d->hidden_size+dim)*d->merge_area+slot] =
                    d->reordered_hidden[static_cast<std::int64_t>(source)*d->hidden_size+dim];
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
int nfn_native_tile_sequence_logp_i32_float32_forward_v1(
    const NfnNativeTileSequenceLogpDescriptorV1* d) {
    if (d == nullptr || d->transformed_logits == nullptr || d->targets == nullptr ||
        d->loss_mask == nullptr || d->sequence_logp == nullptr) return 1;
    for (std::int64_t example = 0; example < d->batch_size; ++example) {
        double total = 0.0;
        for (std::int64_t step = 0; step < d->sequence_length; ++step) {
            const std::int64_t row = example*d->sequence_length+step;
            const auto target = d->targets[row];
            const float mask = d->loss_mask[row];
            if (target == d->ignore_index || !(mask > 0.0f)) continue;
            if (target < 0 || target >= d->vocab_size) return 1;
            const float* logits = d->transformed_logits+row*d->vocab_size;
            const float maximum = *std::max_element(logits, logits+d->vocab_size);
            double denominator = 0.0;
            for (std::int64_t col = 0; col < d->vocab_size; ++col)
                denominator += std::exp(static_cast<double>(logits[col]-maximum));
            total += mask*(static_cast<double>(logits[target]-maximum)-std::log(denominator));
        }
        d->sequence_logp[example] = static_cast<float>(total);
    }
    return 0;
}
int nfn_native_tile_sequence_logp_i32_float32_backward_v1(
    const NfnNativeTileSequenceLogpDescriptorV1* d) {
    if (d == nullptr || d->transformed_logits == nullptr || d->targets == nullptr ||
        d->loss_mask == nullptr || d->grad_sequence_logp == nullptr ||
        d->grad_transformed_logits == nullptr) return 1;
    const std::int64_t rows = d->batch_size*d->sequence_length;
    for (std::int64_t row = 0; row < rows; ++row) {
        const auto target = d->targets[row];
        const float mask = d->loss_mask[row];
        float* grad = d->grad_transformed_logits+row*d->vocab_size;
        if (target == d->ignore_index || !(mask > 0.0f)) {
            std::fill(grad, grad+d->vocab_size, 0.0f);
            continue;
        }
        if (target < 0 || target >= d->vocab_size) return 1;
        const float* logits = d->transformed_logits+row*d->vocab_size;
        const float maximum = *std::max_element(logits, logits+d->vocab_size);
        double denominator = 0.0;
        for (std::int64_t col = 0; col < d->vocab_size; ++col)
            denominator += std::exp(static_cast<double>(logits[col]-maximum));
        const float upstream = d->grad_sequence_logp[row/d->sequence_length]*mask;
        for (std::int64_t col = 0; col < d->vocab_size; ++col) {
            const float probability = static_cast<float>(
                std::exp(static_cast<double>(logits[col]-maximum))/denominator);
            grad[col] = upstream*((col == target ? 1.0f : 0.0f)-probability);
        }
    }
    return 0;
}
int nfn_native_tile_dpo_pairwise_loss_float32_forward_v1(
    const NfnNativeTileDpoPairwiseDescriptorV1* d) {
    if (d == nullptr || d->row_loss == nullptr || d->chosen_reward == nullptr ||
        d->rejected_reward == nullptr) return 1;
    const auto softplus = [](float value) {
        return std::max(value, 0.0f)+std::log1p(std::exp(-std::abs(value)));
    };
    for (std::int64_t i = 0; i < d->examples; ++i) {
        const float chosen = d->policy_logp_chosen[i]-d->reference_logp_chosen[i];
        const float rejected = d->policy_logp_rejected[i]-d->reference_logp_rejected[i];
        const float logit = d->beta*(chosen-rejected);
        if (d->loss_type == NFN_NATIVE_TILE_DPO_LOSS_HINGE) {
            d->row_loss[i] = std::max(0.0f, 1.0f-logit);
        } else if (d->loss_type == NFN_NATIVE_TILE_DPO_LOSS_IPO) {
            const float delta = logit-1.0f/(2.0f*std::max(d->beta, 1.0e-8f));
            d->row_loss[i] = delta*delta;
        } else {
            d->row_loss[i] = (1.0f-d->label_smoothing)*softplus(-logit)+
                d->label_smoothing*softplus(logit);
        }
        d->chosen_reward[i] = d->beta*chosen;
        d->rejected_reward[i] = d->beta*rejected;
    }
    return 0;
}
int nfn_native_tile_dpo_pairwise_loss_float32_backward_v1(
    const NfnNativeTileDpoPairwiseDescriptorV1* d) {
    if (d == nullptr || d->grad_policy_logp_chosen == nullptr ||
        d->grad_policy_logp_rejected == nullptr) return 1;
    for (std::int64_t i = 0; i < d->examples; ++i) {
        const float chosen = d->policy_logp_chosen[i]-d->reference_logp_chosen[i];
        const float rejected = d->policy_logp_rejected[i]-d->reference_logp_rejected[i];
        const float logit = d->beta*(chosen-rejected);
        float derivative = 0.0f;
        if (d->loss_type == NFN_NATIVE_TILE_DPO_LOSS_HINGE) {
            derivative = logit < 1.0f ? -1.0f : 0.0f;
        } else if (d->loss_type == NFN_NATIVE_TILE_DPO_LOSS_IPO) {
            derivative = 2.0f*(logit-1.0f/(2.0f*std::max(d->beta, 1.0e-8f)));
        } else {
            derivative = 1.0f/(1.0f+std::exp(-logit))-
                (1.0f-d->label_smoothing);
        }
        const float gradient = d->grad_scale*d->beta*derivative;
        d->grad_policy_logp_chosen[i] = gradient;
        d->grad_policy_logp_rejected[i] = -gradient;
    }
    return 0;
}
int nfn_native_tile_masked_reward_head_float32_forward_v1(
    const NfnNativeTileMaskedRewardHeadDescriptorV1* d) {
    if (d == nullptr || d->hidden == nullptr || d->sequence_mask == nullptr ||
        !valid(d->weight) || d->reward == nullptr || d->selected_positions == nullptr)
        return 1;
    for (std::int64_t example = 0; example < d->batch_size; ++example) {
        std::int32_t selected = -1;
        for (std::int64_t position = 0; position < d->sequence_length; ++position) {
            const float mask = d->sequence_mask[example*d->sequence_length+position];
            if (!std::isfinite(mask) || mask < 0.0f) return 1;
            if (mask > 0.0f) selected = static_cast<std::int32_t>(position);
        }
        if (selected < 0) return 1;
        d->selected_positions[example] = selected;
        double reward = 0.0;
        for (std::int64_t col = 0; col < d->hidden_size; ++col) {
            reward += d->hidden[
                (example*d->sequence_length+selected)*d->hidden_size+col]*
                value(*d->weight, 0, col);
        }
        d->reward[example] = static_cast<float>(reward);
    }
    return 0;
}
int nfn_native_tile_masked_reward_head_float32_backward_v1(
    const NfnNativeTileMaskedRewardHeadDescriptorV1* d) {
    if (d == nullptr || d->grad_reward == nullptr || d->grad_hidden == nullptr ||
        d->grad_weight == nullptr || d->selected_positions == nullptr || !valid(d->weight))
        return 1;
    std::fill(
        d->grad_hidden,
        d->grad_hidden+d->batch_size*d->sequence_length*d->hidden_size,
        0.0f);
    std::fill(d->grad_weight, d->grad_weight+d->hidden_size, 0.0f);
    for (std::int64_t example = 0; example < d->batch_size; ++example) {
        const std::int32_t selected = d->selected_positions[example];
        if (selected < 0 || selected >= d->sequence_length) return 1;
        for (std::int64_t col = 0; col < d->hidden_size; ++col) {
            const std::int64_t index =
                (example*d->sequence_length+selected)*d->hidden_size+col;
            d->grad_hidden[index] = d->grad_reward[example]*value(*d->weight, 0, col);
            d->grad_weight[col] += d->grad_reward[example]*d->hidden[index];
        }
    }
    return 0;
}
int nfn_native_tile_preference_bce_loss_float32_forward_v1(
    const NfnNativeTilePreferenceBceDescriptorV1* d) {
    if (d == nullptr || d->row_loss == nullptr) return 1;
    for (std::int64_t i = 0; i < d->examples; ++i) {
        const float difference = d->reward_chosen[i]-d->reward_rejected[i];
        d->row_loss[i] = std::max(-difference, 0.0f)+
            std::log1p(std::exp(-std::abs(difference)));
    }
    return 0;
}
int nfn_native_tile_preference_bce_loss_float32_backward_v1(
    const NfnNativeTilePreferenceBceDescriptorV1* d) {
    if (d == nullptr || d->grad_reward_chosen == nullptr ||
        d->grad_reward_rejected == nullptr) return 1;
    for (std::int64_t i = 0; i < d->examples; ++i) {
        const float difference = d->reward_chosen[i]-d->reward_rejected[i];
        const float gradient = d->grad_scale*(
            1.0f/(1.0f+std::exp(-difference))-1.0f);
        d->grad_reward_chosen[i] = gradient;
        d->grad_reward_rejected[i] = -gradient;
    }
    return 0;
}
int nfn_native_tile_token_logp_entropy_i32_float32_forward_v1(
    const NfnNativeTileTokenLogpEntropyDescriptorV1* d) {
    if (d == nullptr || d->transformed_logits == nullptr || d->targets == nullptr ||
        d->loss_mask == nullptr || d->token_logp == nullptr ||
        d->token_entropy == nullptr) return 1;
    for (std::int64_t row = 0; row < d->rows; ++row) {
        const auto target = d->targets[row];
        const float mask = d->loss_mask[row];
        if (target == d->ignore_index || !(mask > 0.0f)) {
            d->token_logp[row] = 0.0f;
            d->token_entropy[row] = 0.0f;
            continue;
        }
        if (target < 0 || target >= d->vocab_size || !std::isfinite(mask)) return 1;
        const float* logits = d->transformed_logits+row*d->vocab_size;
        const float maximum = *std::max_element(logits, logits+d->vocab_size);
        double denominator = 0.0;
        for (std::int64_t col = 0; col < d->vocab_size; ++col)
            denominator += std::exp(static_cast<double>(logits[col]-maximum));
        const double logden = std::log(denominator);
        double entropy = 0.0;
        for (std::int64_t col = 0; col < d->vocab_size; ++col) {
            const double logp = logits[col]-maximum-logden;
            entropy -= std::exp(logp)*logp;
        }
        d->token_logp[row] = static_cast<float>(mask*(logits[target]-maximum-logden));
        d->token_entropy[row] = static_cast<float>(mask*entropy);
    }
    return 0;
}
int nfn_native_tile_token_logp_entropy_i32_float32_backward_v1(
    const NfnNativeTileTokenLogpEntropyDescriptorV1* d) {
    if (d == nullptr || d->grad_token_logp == nullptr ||
        d->grad_token_entropy == nullptr || d->grad_transformed_logits == nullptr)
        return 1;
    for (std::int64_t row = 0; row < d->rows; ++row) {
        const auto target = d->targets[row];
        const float mask = d->loss_mask[row];
        float* gradient = d->grad_transformed_logits+row*d->vocab_size;
        if (target == d->ignore_index || !(mask > 0.0f)) {
            std::fill(gradient, gradient+d->vocab_size, 0.0f);
            continue;
        }
        if (target < 0 || target >= d->vocab_size || !std::isfinite(mask)) return 1;
        const float* logits = d->transformed_logits+row*d->vocab_size;
        const float maximum = *std::max_element(logits, logits+d->vocab_size);
        double denominator = 0.0;
        for (std::int64_t col = 0; col < d->vocab_size; ++col)
            denominator += std::exp(static_cast<double>(logits[col]-maximum));
        const double logden = std::log(denominator);
        double entropy = 0.0;
        for (std::int64_t col = 0; col < d->vocab_size; ++col) {
            const double logp = logits[col]-maximum-logden;
            entropy -= std::exp(logp)*logp;
        }
        for (std::int64_t col = 0; col < d->vocab_size; ++col) {
            const double logp = logits[col]-maximum-logden;
            const float probability = static_cast<float>(std::exp(logp));
            gradient[col] = d->grad_token_logp[row]*mask*
                    ((col == target ? 1.0f : 0.0f)-probability) +
                d->grad_token_entropy[row]*mask*(-probability)*
                    (static_cast<float>(logp)+static_cast<float>(entropy));
        }
    }
    return 0;
}
static bool fake_masked_ppo_stats(
    const NfnNativeTileMaskedPpoLossDescriptorV1* d,
    double* denominator, double* mean, double* variance) {
    double count = 0.0, sum = 0.0;
    for (std::int64_t row = 0; row < d->rows; ++row) {
        if (!std::isfinite(d->loss_mask[row]) || d->loss_mask[row] < 0.0f ||
            !std::isfinite(d->advantages[row])) return false;
        count += d->loss_mask[row];
        sum += d->loss_mask[row]*d->advantages[row];
    }
    if (!(count > 0.0)) return false;
    *mean = sum/count;
    *variance = 0.0;
    if (d->flags & NFN_NATIVE_TILE_PPO_NORMALIZE_ADVANTAGES) {
        for (std::int64_t row = 0; row < d->rows; ++row) {
            const double delta = d->advantages[row]-*mean;
            *variance += d->loss_mask[row]*delta*delta;
        }
        *variance /= count;
    }
    *denominator = count;
    return true;
}
int nfn_native_tile_masked_ppo_loss_float32_forward_v1(
    const NfnNativeTileMaskedPpoLossDescriptorV1* d) {
    if (d == nullptr || d->policy_loss == nullptr || d->value_loss == nullptr ||
        d->entropy_bonus == nullptr || d->total_loss == nullptr) return 1;
    double denominator = 0.0, mean = 0.0, variance = 0.0;
    if (!fake_masked_ppo_stats(d, &denominator, &mean, &variance)) return 1;
    const double inv_std = d->flags & NFN_NATIVE_TILE_PPO_NORMALIZE_ADVANTAGES
        ? 1.0/std::sqrt(variance+d->epsilon) : 1.0;
    double policy = 0.0, value_loss = 0.0, entropy = 0.0;
    for (std::int64_t row = 0; row < d->rows; ++row) {
        const double mask = d->loss_mask[row];
        if (!(mask > 0.0)) continue;
        const double advantage = (d->advantages[row]-
            ((d->flags & NFN_NATIVE_TILE_PPO_NORMALIZE_ADVANTAGES) ? mean : 0.0))*inv_std;
        const double ratio = std::exp(d->logp_new[row]-d->logp_old[row]);
        const double clipped = std::clamp(
            ratio, 1.0-static_cast<double>(d->clip_range),
            1.0+static_cast<double>(d->clip_range));
        policy -= mask*std::min(ratio*advantage, clipped*advantage);
        const double delta = d->value_new[row]-d->value_old[row];
        const double value_clipped = d->value_old[row]+std::clamp(
            delta, -static_cast<double>(d->clip_range),
            static_cast<double>(d->clip_range));
        const double raw_error = d->value_new[row]-d->returns[row];
        const double clipped_error = value_clipped-d->returns[row];
        value_loss += 0.5*mask*std::max(
            raw_error*raw_error, clipped_error*clipped_error);
        entropy += mask*d->entropy[row];
    }
    *d->policy_loss = static_cast<float>(policy/denominator);
    *d->value_loss = static_cast<float>(value_loss/denominator);
    *d->entropy_bonus = static_cast<float>(entropy/denominator);
    *d->total_loss = *d->policy_loss+d->value_coefficient**d->value_loss-
        d->entropy_coefficient**d->entropy_bonus;
    return 0;
}
int nfn_native_tile_masked_ppo_loss_float32_backward_v1(
    const NfnNativeTileMaskedPpoLossDescriptorV1* d) {
    if (d == nullptr || d->grad_logp_new == nullptr ||
        d->grad_value_new == nullptr || d->grad_entropy == nullptr) return 1;
    double denominator = 0.0, mean = 0.0, variance = 0.0;
    if (!fake_masked_ppo_stats(d, &denominator, &mean, &variance)) return 1;
    const double inv_std = d->flags & NFN_NATIVE_TILE_PPO_NORMALIZE_ADVANTAGES
        ? 1.0/std::sqrt(variance+d->epsilon) : 1.0;
    for (std::int64_t row = 0; row < d->rows; ++row) {
        const double mask = d->loss_mask[row];
        if (!(mask > 0.0)) {
            d->grad_logp_new[row] = d->grad_value_new[row] =
                d->grad_entropy[row] = 0.0f;
            continue;
        }
        const double advantage = (d->advantages[row]-
            ((d->flags & NFN_NATIVE_TILE_PPO_NORMALIZE_ADVANTAGES) ? mean : 0.0))*inv_std;
        const double ratio = std::exp(d->logp_new[row]-d->logp_old[row]);
        const bool active = advantage >= 0.0
            ? ratio <= 1.0+d->clip_range : ratio >= 1.0-d->clip_range;
        d->grad_logp_new[row] = active
            ? static_cast<float>(-mask*advantage*ratio/denominator) : 0.0f;
        const double delta = d->value_new[row]-d->value_old[row];
        const double value_clipped = d->value_old[row]+std::clamp(
            delta, -static_cast<double>(d->clip_range),
            static_cast<double>(d->clip_range));
        const double raw_error = d->value_new[row]-d->returns[row];
        const double clipped_error = value_clipped-d->returns[row];
        double value_gradient = 0.0;
        if (raw_error*raw_error >= clipped_error*clipped_error)
            value_gradient = raw_error;
        else if (delta >= -d->clip_range && delta <= d->clip_range)
            value_gradient = clipped_error;
        d->grad_value_new[row] = static_cast<float>(
            d->value_coefficient*mask*value_gradient/denominator);
        d->grad_entropy[row] = static_cast<float>(
            -d->entropy_coefficient*mask/denominator);
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
int nfn_native_tile_gelu_float32(
    const float* input, float* output, std::int64_t count, void*) {
    if (input == nullptr || output == nullptr || count <= 0) return 1;
    for (std::int64_t i = 0; i < count; ++i) {
        output[i] = static_cast<float>(
            0.5 * input[i] *
            (1.0 + std::erf(input[i] / std::sqrt(2.0f))));
    }
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
