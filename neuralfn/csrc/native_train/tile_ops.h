#pragma once

#include <cstdint>

enum : std::uint32_t {
    NFN_NATIVE_TILE_TURBOQUANT_ATTENTION_V1 = 1,
    NFN_NATIVE_TILE_TURBOQUANT_PROFILE_MSE_3_5 = 1,
    NFN_NATIVE_TILE_TURBOQUANT_PROFILE_QJL_3_5 = 2,
};

enum : std::uint32_t {
    NFN_NATIVE_TILE_PACKED_WEIGHT_V1 = 1,
    NFN_NATIVE_TILE_K_QUANT_MMQ_V1 = 1,
    NFN_NATIVE_TILE_PACKED_WEIGHT_F32 = 0,
    NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K = 12,
    NFN_NATIVE_TILE_PACKED_WEIGHT_Q5_K = 13,
    NFN_NATIVE_TILE_PACKED_WEIGHT_Q6_K = 14,
    NFN_NATIVE_TILE_PACKED_WEIGHT_BF16 = 30,
    // Training-only NF4 row layout. Each 64-value group is encoded as one
    // little-endian FP32 absolute-maximum followed by 32 low-nibble-first NF4
    // codes. The final group is zero-padded when input_dim is not divisible by
    // 64. This
    // self-contained layout deliberately avoids auxiliary scale pointers and
    // can therefore use the authenticated v1 typed-weight descriptor.
    NFN_NATIVE_TILE_PACKED_WEIGHT_NF4_GROUP64 = 31,
};

// Muse Glimmer whole-model CUDA feature ABI.  It is deliberately separate
// from the generic SDPA ABI: the latter has a 1,024-key implementation limit
// and cannot represent Glimmer's 32-query/2-KV-head, 128-wide, hybrid
// local/global decode contract.  All cache payloads in this ABI are BF16 bits;
// projections and normalization accumulators are float32.
enum : std::uint32_t {
    NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1 = 1,
    NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT = 0,
    NFN_NATIVE_TILE_GLIMMER_ROPE_INTERLEAVED = 1,
    NFN_NATIVE_TILE_BLOCK_ATTENTION_CAUSAL = 1,
};

// Training feature ABI for the exact Muse Glimmer decoder.  This stays
// separate from the inference ABI so adding saved-LSE/backward state cannot
// change the resident loader's contract.  Activations and gradients are
// float32 in v1; immutable/trainable BF16 weights use the typed packed-weight
// descriptor above.  `window == 0` means full attention, otherwise it is the
// exact causal left window (2,048 for local Glimmer layers).
enum : std::uint32_t {
    NFN_NATIVE_TILE_GLIMMER_TRAINING_V1 = 1,
    NFN_NATIVE_TILE_GLIMMER_TRAIN_CAUSAL = 1,
    NFN_NATIVE_TILE_DPO_LOSS_SIGMOID = 0,
    NFN_NATIVE_TILE_DPO_LOSS_HINGE = 1,
    NFN_NATIVE_TILE_DPO_LOSS_IPO = 2,
    NFN_NATIVE_TILE_PPO_NORMALIZE_ADVANTAGES = 1,
};

struct NfnNativeTilePackedWeightDescriptorV1;

struct NfnNativeTileGlimmerAttentionTrainingDescriptorV1 {
    std::uint32_t struct_size;
    std::uint32_t version;
    std::uint32_t flags;
    std::uint32_t reserved0;

    const float* query;
    const float* key;
    const float* value;
    float* output;
    float* logsumexp;

    // Backward-only fields.  They may be null for forward.
    const float* grad_output;
    float* grad_query;
    float* grad_key;
    float* grad_value;

    std::int64_t batch_size;
    std::int64_t sequence_length;
    std::int64_t query_heads;
    std::int64_t kv_heads;
    std::int64_t head_dim;
    std::int64_t window;
    float scale;
    std::uint32_t reserved1;
    void* cuda_stream;

    // Optional packed-example segment IDs, shaped [batch, sequence].  When
    // non-null, query/key pairs with different IDs are masked in both forward
    // and backward.  This tail field is part of the Glimmer training-v1
    // descriptor size and prevents response-only SFT packing from leaking
    // attention across examples.
    const std::int32_t* sequence_ids;
    std::uint32_t reserved2;
    std::uint32_t reserved3;
};

struct NfnNativeTileGlimmerRmsNormBackwardDescriptorV1 {
    std::uint32_t struct_size;
    std::uint32_t version;
    std::uint32_t flags;
    std::uint32_t reserved0;

    const float* input;
    const NfnNativeTilePackedWeightDescriptorV1* weight;
    const float* grad_output;
    float* grad_input;
    float* grad_weight;

    std::int64_t rows;
    std::int64_t width;
    float eps;
    std::uint32_t centered;
    void* cuda_stream;
};

struct NfnNativeTileGlimmerMaskedCeDescriptorV1 {
    std::uint32_t struct_size;
    std::uint32_t version;
    std::uint32_t flags;
    std::uint32_t reserved0;

    const float* transformed_logits;
    const std::int32_t* targets;
    const float* loss_mask;
    float* row_loss;
    float* grad_transformed_logits;

    std::int64_t rows;
    std::int64_t vocab_size;
    std::int32_t ignore_index;
    std::uint32_t reserved1;
    float grad_scale;
    std::uint32_t reserved2;
    void* cuda_stream;
};

struct NfnNativeTileSequenceLogpDescriptorV1 {
    std::uint32_t struct_size;
    std::uint32_t version;
    std::uint32_t flags;
    std::uint32_t reserved0;

    const float* transformed_logits;
    const std::int32_t* targets;
    const float* loss_mask;
    float* sequence_logp;

    // Backward-only fields. `grad_sequence_logp` has one value per example;
    // `grad_transformed_logits` has batch * sequence * vocab values.
    const float* grad_sequence_logp;
    float* grad_transformed_logits;

    std::int64_t batch_size;
    std::int64_t sequence_length;
    std::int64_t vocab_size;
    std::int32_t ignore_index;
    std::uint32_t reserved1;
    void* cuda_stream;
};

struct NfnNativeTileDpoPairwiseDescriptorV1 {
    std::uint32_t struct_size;
    std::uint32_t version;
    std::uint32_t loss_type;
    std::uint32_t flags;

    const float* policy_logp_chosen;
    const float* policy_logp_rejected;
    const float* reference_logp_chosen;
    const float* reference_logp_rejected;
    float* row_loss;
    float* chosen_reward;
    float* rejected_reward;

    // Backward-only outputs.  The reference is immutable, so this ABI only
    // emits policy gradients.
    float* grad_policy_logp_chosen;
    float* grad_policy_logp_rejected;

    std::int64_t examples;
    float beta;
    float label_smoothing;
    float grad_scale;
    std::uint32_t reserved0;
    void* cuda_stream;
};

struct NfnNativeTileMaskedRewardHeadDescriptorV1 {
    std::uint32_t struct_size;
    std::uint32_t version;
    std::uint32_t flags;
    std::uint32_t reserved0;

    const float* hidden;
    const float* sequence_mask;
    const NfnNativeTilePackedWeightDescriptorV1* weight;
    float* reward;
    std::int32_t* selected_positions;

    // Backward-only fields.
    const float* grad_reward;
    float* grad_hidden;
    float* grad_weight;

    std::int64_t batch_size;
    std::int64_t sequence_length;
    std::int64_t hidden_size;
    void* cuda_stream;
};

struct NfnNativeTilePreferenceBceDescriptorV1 {
    std::uint32_t struct_size;
    std::uint32_t version;
    std::uint32_t flags;
    std::uint32_t reserved0;

    const float* reward_chosen;
    const float* reward_rejected;
    float* row_loss;

    // Backward-only outputs.
    float* grad_reward_chosen;
    float* grad_reward_rejected;

    std::int64_t examples;
    float grad_scale;
    std::uint32_t reserved1;
    void* cuda_stream;
};

struct NfnNativeTileTokenLogpEntropyDescriptorV1 {
    std::uint32_t struct_size;
    std::uint32_t version;
    std::uint32_t flags;
    std::uint32_t reserved0;

    const float* transformed_logits;
    const std::int32_t* targets;
    const float* loss_mask;
    float* token_logp;
    float* token_entropy;

    // Backward-only inputs/outputs.
    const float* grad_token_logp;
    const float* grad_token_entropy;
    float* grad_transformed_logits;

    std::int64_t rows;
    std::int64_t vocab_size;
    std::int32_t ignore_index;
    std::uint32_t reserved1;
    void* cuda_stream;
};

struct NfnNativeTileMaskedPpoLossDescriptorV1 {
    std::uint32_t struct_size;
    std::uint32_t version;
    std::uint32_t flags;
    std::uint32_t reserved0;

    const float* logp_new;
    const float* logp_old;
    const float* advantages;
    const float* value_new;
    const float* value_old;
    const float* returns;
    const float* loss_mask;
    const float* entropy;

    // Four scalar outputs.
    float* policy_loss;
    float* value_loss;
    float* entropy_bonus;
    float* total_loss;

    // Backward-only outputs. Inputs other than logp_new/value_new/entropy are
    // immutable rollout data and intentionally receive no gradient.
    float* grad_logp_new;
    float* grad_value_new;
    float* grad_entropy;

    std::int64_t rows;
    float clip_range;
    float value_coefficient;
    float entropy_coefficient;
    float epsilon;
    void* cuda_stream;
};

// Typed, immutable packed-weight ABI.  This is intentionally distinct from
// nfn_native_tile_linear_quantized_float32, whose `const float*` input is a
// fake-quant training oracle rather than packed checkpoint storage.  Every v1
// call validates the exact canonical row stride and byte extent; unknown
// encoding IDs fail instead of falling through to float reads.
struct NfnNativeTilePackedWeightDescriptorV1 {
    std::uint32_t struct_size;
    std::uint32_t version;
    std::uint32_t encoding;
    std::uint32_t flags;

    const std::uint8_t* data;
    std::int64_t data_nbytes;
    std::int64_t output_dim;
    std::int64_t input_dim;
    std::int64_t row_stride_bytes;
    std::uint32_t reserved0;
    std::uint32_t reserved1;
    void* cuda_stream;
};

// One causal one-token GQA attention operation.  `position` is the absolute
// position of current_key/current_value and is not yet present in the cache.
// Historical rows occupy `key_cache`/`value_cache`; local caches use
// absolute_position % cache_capacity, while global caches use the absolute
// position directly.  `first_key_position` makes the local 2,048-token window
// explicit and must be in [0, position].  The operation is read-only and thus
// safe to abandon before the separate transactional cache commit call.
struct NfnNativeTileGlimmerGqaDecodeDescriptorV1 {
    std::uint32_t struct_size;
    std::uint32_t version;
    std::uint32_t flags;
    std::uint32_t reserved0;

    const float* query;
    const float* current_key;
    const float* current_value;
    const std::uint16_t* key_cache_bf16;
    const std::uint16_t* value_cache_bf16;
    float* output;

    std::int64_t query_heads;
    std::int64_t kv_heads;
    std::int64_t head_dim;
    std::int64_t position;
    std::int64_t first_key_position;
    std::int64_t cache_capacity;
    std::int64_t cache_row_stride;
    float scale;
    std::uint32_t reserved1;
    void* cuda_stream;
};

struct NfnNativeTileGlimmerCacheCommitDescriptorV1 {
    std::uint32_t struct_size;
    std::uint32_t version;
    std::uint32_t flags;
    std::uint32_t reserved0;

    const float* current_key;
    const float* current_value;
    std::uint16_t* key_cache_bf16;
    std::uint16_t* value_cache_bf16;

    std::int64_t kv_heads;
    std::int64_t head_dim;
    std::int64_t position;
    std::int64_t cache_capacity;
    std::int64_t cache_row_stride;
    std::uint32_t reserved1;
    std::uint32_t reserved2;
    void* cuda_stream;
};

// Decode-only composition of the three dependent per-layer operations used by
// the resident Glimmer target: per-head Q/K RMS normalization (plus query
// scale and optional positioned RoPE), one-token GQA, and the transactional
// BF16 cache-row write.  The cache's logical length is still committed by the
// resident only after the complete token succeeds; this operation merely
// writes the otherwise-invisible row selected by `position`.
//
// The packed norm descriptors are embedded by value so the CUDA kernel never
// dereferences host descriptor memory.  `has_*_norm_weight == 0` makes the
// corresponding descriptor payload inert.
struct NfnNativeTileGlimmerFusedDecodeAttentionDescriptorV1 {
    std::uint32_t struct_size;
    std::uint32_t version;
    std::uint32_t flags;
    std::uint32_t reserved0;

    float* query;
    float* key;
    const float* current_value;
    std::uint16_t* key_cache_bf16;
    std::uint16_t* value_cache_bf16;
    float* output;

    NfnNativeTilePackedWeightDescriptorV1 query_norm_weight;
    NfnNativeTilePackedWeightDescriptorV1 key_norm_weight;

    std::int64_t query_heads;
    std::int64_t kv_heads;
    std::int64_t head_dim;
    std::int64_t position;
    std::int64_t first_key_position;
    std::int64_t cache_capacity;
    std::int64_t cache_row_stride;

    float norm_eps;
    float query_scale;
    float rope_theta;
    float attention_scale;
    std::uint32_t rope_layout;
    std::uint32_t has_query_norm_weight;
    std::uint32_t has_key_norm_weight;
    std::uint32_t query_norm_centered;
    std::uint32_t key_norm_centered;
    std::uint32_t apply_rope;
    std::uint32_t reserved1;
    void* cuda_stream;
};

// One immutable cache-layer entry used by the all-layer verification commit.
// The containing descriptor's `layers` pointer addresses host memory; each
// cache pointer stored here addresses device memory.  Keeping this table on the
// host lets the C ABI validate every capacity/stride while the CUDA launcher
// copies the small, bounded table into kernel parameters without a transient
// device allocation.
struct NfnNativeTileGlimmerCacheLayerV1 {
    std::uint16_t* key_cache_bf16;
    std::uint16_t* value_cache_bf16;
    std::int64_t cache_capacity;
    std::int64_t cache_row_stride;
};

// Commits a prefix of a transactionally staged verification block to every
// target cache layer in one CUDA launch.  Staged K/V are laid out as
// [layer, source_rows, kv_heads * head_dim].  `layers` is a host array with
// `layer_count` entries and must remain valid only for the duration of the
// synchronous C call; the cache payload pointers inside its entries are device
// pointers.  The fixed layer limit keeps the CUDA parameter block below the
// portable 4 KiB kernel-argument limit.
struct NfnNativeTileGlimmerCacheCommitLayersDescriptorV1 {
    std::uint32_t struct_size;
    std::uint32_t version;
    std::uint32_t flags;
    std::uint32_t reserved0;

    const float* staged_keys;
    const float* staged_values;
    const NfnNativeTileGlimmerCacheLayerV1* layers;

    std::int64_t layer_count;
    std::int64_t source_rows;
    std::int64_t rows;
    std::int64_t kv_heads;
    std::int64_t head_dim;
    std::int64_t position;
    std::int64_t source_layer_stride;
    std::uint32_t reserved1;
    std::uint32_t reserved2;
    void* cuda_stream;
};

// Block attention evaluates current rows against accepted-context K/V.  With
// flags=0 the current block is bidirectional for DFlash.  With
// NFN_NATIVE_TILE_BLOCK_ATTENTION_CAUSAL it is the target verifier's causal
// block.  Both modes obey the configured absolute sliding window.
// Historical cache rows are BF16 ring-buffer entries. In causal verifier mode,
// earlier current-block rows are BF16-rounded on read to reproduce sequential
// cache semantics while the query's own K/V stays float32. In non-causal
// DFlash mode the current block remains float32. The output is always float32
// so a proposal may be abandoned transactionally.
struct NfnNativeTileDFlashBlockAttentionDescriptorV1 {
    std::uint32_t struct_size;
    std::uint32_t version;
    std::uint32_t flags;
    std::uint32_t reserved0;

    const float* query;
    const float* block_key;
    const float* block_value;
    const std::uint16_t* key_cache_bf16;
    const std::uint16_t* value_cache_bf16;
    float* output;

    std::int64_t query_rows;
    std::int64_t block_rows;
    std::int64_t query_heads;
    std::int64_t kv_heads;
    std::int64_t head_dim;
    std::int64_t context_length;
    std::int64_t sliding_window;
    std::int64_t cache_capacity;
    std::int64_t cache_row_stride;
    float scale;
    std::uint32_t reserved1;
    void* cuda_stream;
};

#define NFN_NATIVE_TILE_GLIMMER_VISION_V1 1u

// Whole-model Muse Glimmer vision operations. All tensor and layout pointers
// are device pointers. The host may construct index/mask metadata because it
// is independent of learned weights, but patch projection, learned-position
// interpolation, normalization, RoPE, attention and pixel shuffle execute on
// the selected CUDA stream.
struct NfnNativeTileGlimmerVisionPrepareDescriptorV1 {
    std::uint32_t struct_size;
    std::uint32_t version;
    const float* projected;
    const float* position_table;
    const std::int32_t* corner_indices;
    const float* corner_weights;
    const std::int32_t* permutation;
    float* output;
    std::int64_t rows;
    std::int64_t width;
    std::int64_t position_rows;
    void* cuda_stream;
};

struct NfnNativeTileGlimmerVisionAttentionDescriptorV1 {
    std::uint32_t struct_size;
    std::uint32_t version;
    std::uint32_t interleaved_rope;
    std::uint32_t reserved0;
    const float* query;
    const float* key;
    const float* value;
    const std::int32_t* position_width;
    const std::int32_t* position_height;
    const std::int32_t* row_begin;
    const std::int32_t* row_end;
    float* output;
    std::int64_t rows;
    std::int64_t heads;
    std::int64_t head_dim;
    float rope_theta;
    std::uint32_t reserved1;
    void* cuda_stream;
};

struct NfnNativeTileGlimmerVisionPixelShuffleDescriptorV1 {
    std::uint32_t struct_size;
    std::uint32_t version;
    const float* reordered_hidden;
    const std::int32_t* source_rows;
    float* output;
    std::int64_t merged_rows;
    std::int64_t hidden_size;
    std::int64_t merge_area;
    void* cuda_stream;
};

// Additive TurboQuant attention ABI.  struct_size is first so callers may pass
// a larger future descriptor while v1 validates and consumes this prefix.
//
// key_records/value_records use the resident CPU-v1 byte layout, batched as
// [batch][layer][position][kv_head][record].  An MSE key record is
// [f32 norm][mixed-bit indices]; a QJL key record is
// [f32 norm][f32 residual_norm][mixed-bit indices][ceil(head_dim/8) signs].
// Value records are always [f32 norm][mixed-bit indices].  Mixed-bit fields are
// packed least-significant bit first, with canonical even/odd channel widths
// 4/3 for MSE values and keys, and 3/2 for QJL keys.  The current row remains
// exact float32 and participates in the same stable softmax as historical
// compressed rows.  Matrix/table pointers and all tensor pointers are device
// pointers; matrices are row-major float64, matching the CPU-v1 tables.
struct NfnNativeTileTurboQuantAttentionDescriptorV1 {
    std::uint32_t struct_size;
    std::uint32_t version;
    std::uint32_t profile;
    std::uint32_t flags;

    const float* query;
    const std::uint8_t* key_records;
    const std::uint8_t* value_records;
    const float* current_key;
    const float* current_value;
    float* output;

    const double* rotation;
    const double* qjl_projection;
    const double* centroids_2bit;
    const double* centroids_3bit;
    const double* centroids_4bit;

    std::int64_t batch_size;
    std::int64_t layer_index;
    std::int64_t num_layers;
    std::int64_t query_heads;
    std::int64_t kv_heads;
    std::int64_t head_dim;
    std::int64_t past_sequence_length;
    std::int64_t cache_capacity;
    std::int64_t key_record_bytes;
    std::int64_t value_record_bytes;

    // Zero selects the canonical contiguous span for that tensor/cache.
    std::int64_t key_cache_batch_stride_bytes;
    std::int64_t value_cache_batch_stride_bytes;
    std::int64_t query_batch_stride;
    std::int64_t current_key_batch_stride;
    std::int64_t current_value_batch_stride;
    std::int64_t output_batch_stride;

    float scale;
    std::uint32_t reserved0;
    void* cuda_stream;
};

extern "C" {

int nfn_native_tile_ops_abi_version();
int nfn_native_tile_strict_math_abi_version();
int nfn_native_tile_turboquant_attention_abi_version();
int nfn_native_tile_packed_weight_abi_version();
int nfn_native_tile_k_quant_mmq_abi_version();
int nfn_native_tile_glimmer_inference_abi_version();
int nfn_native_tile_glimmer_vision_abi_version();
int nfn_native_tile_glimmer_training_abi_version();
int nfn_native_tile_packed_weight_validate_v1(
    const NfnNativeTilePackedWeightDescriptorV1* descriptor);
int nfn_native_tile_packed_weight_dequantize_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* descriptor,
    float* output);
int nfn_native_tile_linear_packed_weight_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* descriptor,
    const float* input,
    const float* bias,
    float* output,
    std::int64_t rows,
    bool has_bias);
// Optional K-quant inference fast path. The activation is quantized in
// independent 32-value blocks; q8_values has rows*width int8 entries and
// q8_scales/q8_sums each have rows*(width/32) float entries.
int nfn_native_tile_quantize_q8_1_float32_v1(
    const float* input,
    std::int8_t* q8_values,
    float* q8_scales,
    float* q8_sums,
    std::int64_t rows,
    std::int64_t width,
    void* cuda_stream);
int nfn_native_tile_linear_packed_weight_q8_1_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* descriptor,
    const std::int8_t* q8_values,
    const float* q8_scales,
    const float* q8_sums,
    const float* bias,
    float* output,
    std::int64_t rows,
    bool has_bias);
// Small-batch K-quant MMQ used by Muse Glimmer prefill and DFlash.
// The descriptor/output arrays are host arrays containing CUDA device
// pointers. All descriptors must have the same input_dim and use Q4_K, Q5_K,
// or Q6_K. The implementation quantizes the shared FP32 activation once per
// encoding present, executes one to four projections, and uses only the
// caller-owned workspace on cuda_stream.
std::int64_t nfn_native_tile_k_quant_mmq_workspace_bytes_v1(
    std::int64_t rows,
    std::int64_t input_dim);
int nfn_native_tile_k_quant_mmq_multi_linear_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
    const float* input,
    float* const* outputs,
    std::int64_t operation_count,
    std::int64_t rows,
    void* workspace,
    std::int64_t workspace_nbytes,
    void* cuda_stream);
// Exact inference-only fusion of Glimmer's attention sigmoid gate with the
// MMQ activation quantizer. It is numerically identical to materializing
// `input / (1 + exp(-gate))` in FP32 before the ordinary MMQ call, while
// eliminating that intermediate kernel and device-memory round trip.
int nfn_native_tile_k_quant_mmq_multi_linear_gated_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
    const float* input,
    const float* gate,
    float* const* outputs,
    std::int64_t operation_count,
    std::int64_t rows,
    void* workspace,
    std::int64_t workspace_nbytes,
    void* cuda_stream);
// Exact inference-only fusion of SwiGLU (`up * gate * sigmoid(gate)`) with
// the MMQ activation quantizer for the following down projection.
int nfn_native_tile_k_quant_mmq_multi_linear_swiglu_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
    const float* gate,
    const float* up,
    float* const* outputs,
    std::int64_t operation_count,
    std::int64_t rows,
    void* workspace,
    std::int64_t workspace_nbytes,
    void* cuda_stream);
// Target-verifier variants preserve the one-row MMVQ activation layout and
// warp accumulation order independently for every row, while co-scheduling up
// to 16 rows per launch to reuse packed-weight cache lines. These must remain
// distinct from prompt MMQ: switching prompt arithmetic changes greedy output.
int nfn_native_tile_k_quant_mmvq_multi_linear_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
    const float* input,
    float* const* outputs,
    std::int64_t operation_count,
    std::int64_t rows,
    void* workspace,
    std::int64_t workspace_nbytes,
    void* cuda_stream);
// One-row counterpart for a Q8_1 activation already prepared in `workspace`
// by the fused dual-RMS handoff below. It skips only activation quantization;
// packed-weight dot products and reductions are identical to the ordinary
// MMVQ entry point.
int nfn_native_tile_k_quant_mmvq_multi_linear_prequantized_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
    float* const* outputs,
    std::int64_t operation_count,
    std::int64_t rows,
    void* workspace,
    std::int64_t workspace_nbytes,
    void* cuda_stream);
int nfn_native_tile_k_quant_mmvq_multi_linear_gated_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
    const float* input,
    const float* gate,
    float* const* outputs,
    std::int64_t operation_count,
    std::int64_t rows,
    void* workspace,
    std::int64_t workspace_nbytes,
    void* cuda_stream);
int nfn_native_tile_k_quant_mmvq_multi_linear_swiglu_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* const* descriptors,
    const float* gate,
    const float* up,
    float* const* outputs,
    std::int64_t operation_count,
    std::int64_t rows,
    void* workspace,
    std::int64_t workspace_nbytes,
    void* cuda_stream);
// Exact one-row decode path using the pinned four-warp cooperative llama.cpp
// MMVQ reduction. `workspace` needs at least input_dim/32 Q8_1 blocks and may
// reuse the larger MMQ workspace returned above.
int nfn_native_tile_k_quant_mmvq_linear_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* descriptor,
    const float* input,
    float* output,
    void* workspace,
    std::int64_t workspace_nbytes,
    void* cuda_stream);
// Decode-only fused launch for two to four independent projections that share
// one pre-quantized Q8_1 activation row. Unused descriptor/output slots must be
// null. Each projection preserves the single-linear warp accumulation order.
int nfn_native_tile_linear_packed_weight_q8_1_multi_decode_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* descriptor0,
    const NfnNativeTilePackedWeightDescriptorV1* descriptor1,
    const NfnNativeTilePackedWeightDescriptorV1* descriptor2,
    const NfnNativeTilePackedWeightDescriptorV1* descriptor3,
    const std::int8_t* q8_values,
    const float* q8_scales,
    const float* q8_sums,
    float* output0,
    float* output1,
    float* output2,
    float* output3,
    std::int64_t projection_count,
    void* cuda_stream);
// Deterministic row-wise argmax. Ties select the lowest column index.
int nfn_native_tile_argmax_rows_float32_v1(
    const float* values,
    std::int64_t* output_indices,
    float* output_values,
    std::int64_t rows,
    std::int64_t width,
    void* cuda_stream);
int nfn_native_tile_linear_backward_input_packed_weight_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* descriptor,
    const float* grad_output,
    float* grad_input,
    std::int64_t rows);
int nfn_native_tile_glimmer_embedding_gather_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* descriptor,
    std::int64_t token_id,
    float* output);
// Device-token counterpart used by a captured greedy decode graph. The token
// ID is read when the graph executes, so one graph instance can serve every
// generated token without patching a kernel node between launches.
int nfn_native_tile_glimmer_embedding_gather_device_i64_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* descriptor,
    const std::int64_t* token_id,
    float* output);
int nfn_native_tile_glimmer_embedding_batch_i32_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* descriptor,
    const std::int32_t* token_ids,
    float* output,
    std::int64_t rows);
int nfn_native_tile_glimmer_rms_norm_affine_float32_v1(
    const float* input,
    const NfnNativeTilePackedWeightDescriptorV1* weight,
    float* output,
    std::int64_t rows,
    std::int64_t width,
    float eps,
    bool centered,
    void* cuda_stream);
int nfn_native_tile_glimmer_rms_norm_affine_capture_residual_float32_v1(
    const float* input,
    const NfnNativeTilePackedWeightDescriptorV1* weight,
    float* output,
    float* residual_output,
    std::int64_t rows,
    std::int64_t width,
    float eps,
    bool centered,
    void* cuda_stream);
// Fuses the decode-path Q8_1 activation quantizer with the residual-capturing
// wide RMSNorm. The FP32 output/residual and Q8 metadata are bit-identical to
// calling the two standalone operations in stream order.
int nfn_native_tile_glimmer_rms_norm_affine_capture_residual_q8_1_float32_v1(
    const float* input,
    const NfnNativeTilePackedWeightDescriptorV1* weight,
    float* output,
    float* residual_output,
    std::int8_t* q8_values,
    float* q8_scales,
    float* q8_sums,
    std::int64_t rows,
    std::int64_t width,
    float eps,
    bool centered,
    void* cuda_stream);
int nfn_native_tile_glimmer_rms_norm_affine_add_residual_float32_v1(
    const float* input,
    const NfnNativeTilePackedWeightDescriptorV1* weight,
    const float* residual_input,
    float* output,
    std::int64_t rows,
    std::int64_t width,
    float eps,
    bool centered,
    void* cuda_stream);
// Decode composition for two adjacent Glimmer normalization stages:
//
//   hidden = residual_input + rms(input, first_weight, first_eps)
//   residual_output = hidden
//   normalized_output = rms(hidden, second_weight, second_eps)
//
// The reduction and FP32 operation order match the standalone add-residual
// followed by capture-residual kernels, while removing one launch.
int nfn_native_tile_glimmer_dual_rms_add_capture_float32_v1(
    const float* input,
    const NfnNativeTilePackedWeightDescriptorV1* first_weight,
    const float* residual_input,
    float* hidden_output,
    const NfnNativeTilePackedWeightDescriptorV1* second_weight,
    float* normalized_output,
    float* residual_output,
    std::int64_t rows,
    std::int64_t width,
    float first_eps,
    bool first_centered,
    float second_eps,
    bool second_centered,
    void* cuda_stream);
// Multi-row verifier specialization. Each row is partitioned across several
// cooperative blocks while preserving the exact 256-lane FP32 reduction and
// operation order of the ordinary dual-RMS path. The implementation currently
// accepts only F32 affine norm descriptors; callers must retain the ordinary
// kernel as the fallback for BF16 or other norm encodings.
int nfn_native_tile_glimmer_dual_rms_add_capture_cooperative_batch_float32_v1(
    const float* input,
    const NfnNativeTilePackedWeightDescriptorV1* first_weight,
    const float* residual_input,
    float* hidden_output,
    const NfnNativeTilePackedWeightDescriptorV1* second_weight,
    float* normalized_output,
    float* residual_output,
    std::int64_t rows,
    std::int64_t width,
    float first_eps,
    bool first_centered,
    float second_eps,
    bool second_centered,
    void* cuda_stream);
// One-row decode megakernel variant. In addition to the ordinary dual-RMS
// outputs, it writes the exact llama Q8_1 activation layout into the caller's
// MMVQ workspace so the next packed projection can skip a standalone
// quantization launch.
int nfn_native_tile_glimmer_dual_rms_add_capture_mmvq_q8_float32_v1(
    const float* input,
    const NfnNativeTilePackedWeightDescriptorV1* first_weight,
    const float* residual_input,
    float* hidden_output,
    const NfnNativeTilePackedWeightDescriptorV1* second_weight,
    float* normalized_output,
    float* residual_output,
    std::int64_t rows,
    std::int64_t width,
    float first_eps,
    bool first_centered,
    float second_eps,
    bool second_centered,
    void* mmvq_workspace,
    std::int64_t mmvq_workspace_nbytes,
    void* cuda_stream);
int nfn_native_tile_glimmer_positioned_rope_float32_v1(
    float* query,
    float* key,
    std::int64_t query_heads,
    std::int64_t kv_heads,
    std::int64_t head_dim,
    std::int64_t position,
    float theta,
    std::uint32_t layout,
    void* cuda_stream);
// Decode-only composition of per-head Q/K RMS normalization, query scaling,
// and optional positioned RoPE. Global NoPE layers set apply_rope=false.
int nfn_native_tile_glimmer_qk_norm_scale_rope_float32_v1(
    float* query,
    float* key,
    const NfnNativeTilePackedWeightDescriptorV1* query_norm_weight,
    const NfnNativeTilePackedWeightDescriptorV1* key_norm_weight,
    std::int64_t query_heads,
    std::int64_t kv_heads,
    std::int64_t head_dim,
    float eps,
    bool query_norm_centered,
    bool key_norm_centered,
    float query_scale,
    std::int64_t position,
    float theta,
    std::uint32_t layout,
    bool apply_rope,
    void* cuda_stream);
// Batched counterpart used by target verification and DFlash proposal
// blocks. Each row uses absolute position `position + row`.
int nfn_native_tile_glimmer_qk_norm_scale_rope_batch_float32_v1(
    float* query,
    float* key,
    const NfnNativeTilePackedWeightDescriptorV1* query_norm_weight,
    const NfnNativeTilePackedWeightDescriptorV1* key_norm_weight,
    std::int64_t rows,
    std::int64_t query_heads,
    std::int64_t kv_heads,
    std::int64_t head_dim,
    float eps,
    bool query_norm_centered,
    bool key_norm_centered,
    float query_scale,
    std::int64_t position,
    float theta,
    std::uint32_t layout,
    bool apply_rope,
    void* cuda_stream);
int nfn_native_tile_glimmer_gqa_decode_float32_v1(
    const NfnNativeTileGlimmerGqaDecodeDescriptorV1* descriptor);
int nfn_native_tile_glimmer_fused_decode_attention_float32_v1(
    const NfnNativeTileGlimmerFusedDecodeAttentionDescriptorV1* descriptor);
// Captured-graph counterpart of the fused decode operation. `device_position`
// is read on device at execution time. Local layers pass their sliding window;
// global NoPE layers pass max context and still derive first_key_position=0.
int nfn_native_tile_glimmer_fused_decode_attention_device_position_float32_v1(
    const NfnNativeTileGlimmerFusedDecodeAttentionDescriptorV1* descriptor,
    const std::int64_t* device_position,
    std::int64_t sliding_window);
int nfn_native_tile_glimmer_cache_commit_bf16_v1(
    const NfnNativeTileGlimmerCacheCommitDescriptorV1* descriptor);
int nfn_native_tile_glimmer_cache_commit_rows_bf16_v1(
    const NfnNativeTileGlimmerCacheCommitDescriptorV1* descriptor,
    std::int64_t rows);
int nfn_native_tile_glimmer_cache_commit_layers_bf16_v1(
    const NfnNativeTileGlimmerCacheCommitLayersDescriptorV1* descriptor);
// Packs target verification taps from [tap, source_row, hidden] to
// [row, tap, hidden] without a host transpose. The selected source range is
// contiguous and may be shorter than source_rows.
int nfn_native_tile_glimmer_pack_target_taps_float32_v1(
    const float* tap_major,
    float* row_major,
    std::int64_t source_rows,
    std::int64_t source_row_offset,
    std::int64_t rows,
    std::int64_t tap_count,
    std::int64_t hidden_width,
    void* cuda_stream);
int nfn_native_tile_dflash_block_attention_float32_v1(
    const NfnNativeTileDFlashBlockAttentionDescriptorV1* descriptor);
// Short-context DFlash verifier specialization. The caller supplies the FP32
// score workspace; unsupported shapes/lengths fail closed so the resident
// runtime can retain the general attention kernel as its fallback.
int nfn_native_tile_dflash_block_attention_short_split_float32_v1(
    const NfnNativeTileDFlashBlockAttentionDescriptorV1* descriptor,
    float* score_workspace,
    std::int64_t score_workspace_nbytes);
int nfn_native_tile_glimmer_vision_prepare_float32_v1(
    const NfnNativeTileGlimmerVisionPrepareDescriptorV1* descriptor);
int nfn_native_tile_glimmer_vision_layer_norm_float32_v1(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::int64_t rows,
    std::int64_t width,
    float eps,
    void* cuda_stream);
int nfn_native_tile_glimmer_vision_attention_float32_v1(
    const NfnNativeTileGlimmerVisionAttentionDescriptorV1* descriptor);
int nfn_native_tile_glimmer_vision_pixel_shuffle_float32_v1(
    const NfnNativeTileGlimmerVisionPixelShuffleDescriptorV1* descriptor);
int nfn_native_tile_glimmer_sigmoid_gate_float32_v1(
    const float* values,
    const float* gate,
    float* output,
    std::int64_t count,
    void* cuda_stream);
int nfn_native_tile_glimmer_logit_transform_float32_v1(
    float* logits,
    std::int64_t count,
    float multiplier,
    float softcap,
    void* cuda_stream);
int nfn_native_tile_glimmer_attention_forward_float32_v1(
    const NfnNativeTileGlimmerAttentionTrainingDescriptorV1* descriptor);
int nfn_native_tile_glimmer_attention_backward_float32_v1(
    const NfnNativeTileGlimmerAttentionTrainingDescriptorV1* descriptor);
int nfn_native_tile_glimmer_rms_norm_backward_float32_v1(
    const NfnNativeTileGlimmerRmsNormBackwardDescriptorV1* descriptor);
int nfn_native_tile_glimmer_positioned_rope_batch_float32_v1(
    float* query,
    float* key,
    std::int64_t rows,
    std::int64_t query_heads,
    std::int64_t kv_heads,
    std::int64_t head_dim,
    std::int64_t start_position,
    float theta,
    std::uint32_t layout,
    bool inverse,
    void* cuda_stream);
int nfn_native_tile_glimmer_sigmoid_gate_backward_float32_v1(
    const float* values,
    const float* gate,
    const float* grad_output,
    float* grad_values,
    float* grad_gate,
    std::int64_t count,
    void* cuda_stream);
int nfn_native_tile_glimmer_logit_transform_backward_float32_v1(
    const float* transformed_logits,
    const float* grad_transformed_logits,
    float* grad_raw_logits,
    std::int64_t count,
    float multiplier,
    float softcap,
    void* cuda_stream);
int nfn_native_tile_glimmer_masked_cross_entropy_i32_float32_v1(
    const NfnNativeTileGlimmerMaskedCeDescriptorV1* descriptor);
int nfn_native_tile_sequence_logp_i32_float32_forward_v1(
    const NfnNativeTileSequenceLogpDescriptorV1* descriptor);
int nfn_native_tile_sequence_logp_i32_float32_backward_v1(
    const NfnNativeTileSequenceLogpDescriptorV1* descriptor);
int nfn_native_tile_dpo_pairwise_loss_float32_forward_v1(
    const NfnNativeTileDpoPairwiseDescriptorV1* descriptor);
int nfn_native_tile_dpo_pairwise_loss_float32_backward_v1(
    const NfnNativeTileDpoPairwiseDescriptorV1* descriptor);
int nfn_native_tile_masked_reward_head_float32_forward_v1(
    const NfnNativeTileMaskedRewardHeadDescriptorV1* descriptor);
int nfn_native_tile_masked_reward_head_float32_backward_v1(
    const NfnNativeTileMaskedRewardHeadDescriptorV1* descriptor);
int nfn_native_tile_preference_bce_loss_float32_forward_v1(
    const NfnNativeTilePreferenceBceDescriptorV1* descriptor);
int nfn_native_tile_preference_bce_loss_float32_backward_v1(
    const NfnNativeTilePreferenceBceDescriptorV1* descriptor);
int nfn_native_tile_token_logp_entropy_i32_float32_forward_v1(
    const NfnNativeTileTokenLogpEntropyDescriptorV1* descriptor);
int nfn_native_tile_token_logp_entropy_i32_float32_backward_v1(
    const NfnNativeTileTokenLogpEntropyDescriptorV1* descriptor);
int nfn_native_tile_masked_ppo_loss_float32_forward_v1(
    const NfnNativeTileMaskedPpoLossDescriptorV1* descriptor);
int nfn_native_tile_masked_ppo_loss_float32_backward_v1(
    const NfnNativeTileMaskedPpoLossDescriptorV1* descriptor);
int nfn_native_tile_token_embedding_backward_weight_i32_float32(
    const std::int32_t* token_ids,
    const float* grad_output,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t vocab_size,
    std::int64_t embedding_dim,
    void* cuda_stream);
int nfn_native_tile_glimmer_adamw_bf16_float32_v1(
    std::uint16_t* parameter_bf16,
    const float* gradient,
    float* exp_avg,
    float* exp_avg_sq,
    std::int64_t count,
    float learning_rate,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    std::int64_t step,
    float gradient_scale,
    void* cuda_stream);
int nfn_native_tile_turboquant_attention_forward_v1(
    const NfnNativeTileTurboQuantAttentionDescriptorV1* descriptor);
void nfn_native_tile_turboquant_attention_stats_reset();
std::int64_t nfn_native_tile_turboquant_attention_launch_count();
const char* nfn_native_tile_ops_error_string(int code);
void nfn_native_tile_attention_forward_stats_reset();
std::int64_t nfn_native_tile_attention_forward_row_launch_count();
std::int64_t nfn_native_tile_attention_forward_tk_launch_count();
std::int64_t nfn_native_tile_attention_backward_tk_launch_count();
std::int64_t nfn_native_tile_attention_backward_tk_batch_cap();
std::int64_t nfn_native_tile_attention_backward_tk_chunk_batch_total();
std::int64_t nfn_native_tile_attention_backward_tk_chunk_batch_max();
std::int64_t nfn_native_tile_attention_backward_tk_chunk_batch_min();
std::int64_t nfn_native_tile_attention_backward_tk_chunk_batch_last();
int nfn_native_tile_attention_backward_tk_block_size();
std::int64_t nfn_native_tile_attention_backward_dprep_default_warps_per_block();
std::int64_t nfn_native_tile_sm120_memory_block_size();
std::int64_t nfn_native_tile_sm120_layernorm_bwd_blocks_per_sm();
std::int64_t nfn_native_tile_attention_backward_float_hd64_dprep_launch_count();
std::int64_t nfn_native_tile_attention_backward_dprep_timing_us();
std::int64_t nfn_native_tile_attention_backward_dprep_timing_count();
std::int64_t nfn_native_tile_attention_backward_tk_timing_us();
std::int64_t nfn_native_tile_attention_backward_tk_timing_count();
std::int64_t nfn_native_tile_attention_tk_workspace_allocation_count();
std::int64_t nfn_native_tile_attention_tk_workspace_element_capacity();
std::int64_t nfn_native_tile_attention_tk_workspace_row_capacity();
std::int64_t nfn_native_tile_token_cross_entropy_workspace_allocation_count();
std::int64_t nfn_native_tile_token_cross_entropy_workspace_row_capacity();
std::int64_t nfn_native_tile_token_cross_entropy_bf16_threads_per_row();
std::int64_t nfn_native_tile_lm_head_true_fused_mat_tile();
std::int64_t nfn_native_tile_lm_head_true_fused_required_threads();
std::int64_t nfn_native_tile_lm_head_prob_only_target_correction_threads();
void nfn_native_tile_lm_head_classifier_stats_reset();
std::int64_t nfn_native_tile_lm_head_classifier_chunk_launch_count();
std::int64_t nfn_native_tile_lm_head_classifier_last_rows();
std::int64_t nfn_native_tile_lm_head_classifier_last_vocab();
std::int64_t nfn_native_tile_lm_head_classifier_last_row_stride();
std::int64_t nfn_native_tile_lm_head_classifier_loss_bin_launch_count();
std::int64_t nfn_native_tile_lm_head_classifier_true_fused_launch_count();
std::int64_t nfn_native_tile_attention_forward_row_fallback_count();
std::int64_t nfn_native_tile_attention_forward_scalar_launch_count();
int nfn_native_tile_attention_forward_row_last_error();
int nfn_native_tile_attention_forward_row_prelaunch_clear_error();
int nfn_native_tile_attention_forward_row_prelaunch_peek_error();
std::int64_t nfn_native_tile_attention_forward_row_grid_x();
std::int64_t nfn_native_tile_attention_forward_row_grid_y();
std::int64_t nfn_native_tile_attention_forward_row_grid_z();
std::int64_t nfn_native_tile_attention_forward_row_block_x();
int nfn_native_tile_attention_forward_row_attr_status();
int nfn_native_tile_attention_forward_row_attr_max_threads_per_block();
int nfn_native_tile_attention_forward_row_attr_num_regs();
std::int64_t nfn_native_tile_attention_forward_row_attr_shared_size_bytes();
std::int64_t nfn_native_tile_attention_forward_row_attr_const_size_bytes();
std::int64_t nfn_native_tile_attention_forward_row_attr_local_size_bytes();
void nfn_native_tile_trainer_linear_stats_reset();
void nfn_native_tile_trainer_linear_bf16_cache_reset();
std::int64_t nfn_native_tile_trainer_linear_bf16_gemm_count();
std::int64_t nfn_native_tile_trainer_linear_bf16_gemm_fast16bf_request_count();
std::int64_t nfn_native_tile_trainer_linear_tk_gemm_count();
std::int64_t nfn_native_tile_trainer_linear_tk_float_out_gemm_count();
std::int64_t nfn_native_tile_trainer_linear_tk_dweight_gemm_count();
std::int64_t nfn_native_tile_trainer_linear_tk_dgelu_dinput_gemm_count();
int nfn_native_tile_trainer_linear_tk_sm120_k_tile();
int nfn_native_tile_trainer_linear_tk_sm120_grad_k_tile();
int nfn_native_tile_trainer_linear_tk_sm120_super_m();
int nfn_native_tile_trainer_linear_tk_sm120_dinput_super_m();
int nfn_native_tile_trainer_linear_tk_sm120_dweight_super_m();
int nfn_native_tile_trainer_linear_tk_sm120_huge_n_k_tile();
int nfn_native_tile_trainer_linear_tk_sm120_fast_dgelu_enabled();
int nfn_native_tile_trainer_linear_tk_sm120_approx_dgelu_tanh_enabled();
std::int64_t nfn_native_tile_trainer_linear_cublaslt_gemm_count();
std::int64_t nfn_native_tile_trainer_linear_cublaslt_bgrad_gemm_count();
std::int64_t nfn_native_tile_trainer_linear_cublaslt_bgrad_direct_write_count();
std::int64_t nfn_native_tile_trainer_linear_cublaslt_bgrad_accumulate_count();
int nfn_native_tile_linear_backward_bias_threads_per_block();
std::int64_t nfn_native_tile_trainer_linear_sgemm_count();
std::int64_t nfn_native_tile_trainer_bf16_to_f32_vec4_count();
std::int64_t nfn_native_tile_trainer_linear_bf16_a_pack_count();
std::int64_t nfn_native_tile_trainer_linear_bf16_cached_a_pack_count();
std::int64_t nfn_native_tile_trainer_linear_bf16_cached_b_pack_count();
std::int64_t nfn_native_tile_trainer_linear_bf16_transient_a_pack_count();
std::int64_t nfn_native_tile_trainer_linear_bf16_transient_b_pack_count();
std::int64_t nfn_native_tile_trainer_linear_bf16_a_cache_hit_count();
std::int64_t nfn_native_tile_trainer_linear_bf16_cache_reset_count();
std::int64_t nfn_native_tile_trainer_linear_bf16_workspace_allocation_count();
std::int64_t nfn_native_tile_trainer_linear_bf16_workspace_a_capacity();
std::int64_t nfn_native_tile_trainer_linear_bf16_workspace_b_capacity();
std::int64_t nfn_native_tile_trainer_linear_bf16_cached_a_capacity();
std::int64_t nfn_native_tile_trainer_linear_bf16_cache_entry_count();
int nfn_native_tile_trainer_linear_cublaslt_grouped_layout_probe_status();
int nfn_native_tile_trainer_linear_cublaslt_grouped_matmul_probe_status();
int nfn_native_tile_trainer_linear_cublas_grouped_bf16_gemm_probe_status();
int nfn_native_tile_trainer_linear_cublas_prewarm(void* stream);
int nfn_native_tile_trainer_linear_bf16_workspace_prewarm(
    std::int64_t a_elements,
    std::int64_t b_elements,
    std::int64_t c_elements);
int nfn_native_tile_trainer_linear_cublaslt_prewarm_bf16_plan(
    int m,
    int n,
    int k,
    int op_a,
    int op_b,
    int lda,
    int ldb,
    int ldc,
    int bgrad_epilogue);
std::int64_t nfn_native_tile_trainer_linear_shape_stats_count();
bool nfn_native_tile_trainer_linear_shape_stats_entry(
    std::int64_t index,
    int* path,
    int* m,
    int* n,
    int* k,
    int* op_a,
    int* op_b,
    std::int64_t* calls,
    std::int64_t* total_us);
bool nfn_native_tile_trainer_linear_shape_stats_entry_v2(
    std::int64_t index,
    int* path,
    int* m,
    int* n,
    int* k,
    int* op_a,
    int* op_b,
    std::int64_t* calls,
    std::int64_t* total_us,
    int* cublaslt_selected_heuristic,
    int* cublaslt_returned_heuristics,
    std::int64_t* cublaslt_workspace_bytes);
std::int64_t nfn_native_tile_trainer_linear_cublaslt_plan_cache_count();
bool nfn_native_tile_trainer_linear_cublaslt_plan_cache_entry(
    std::int64_t index,
    int* m,
    int* n,
    int* k,
    int* op_a,
    int* op_b,
    int* selected_heuristic,
    int* returned_heuristics,
    std::int64_t* workspace_bytes,
    int* epilogue);

int nfn_native_tile_gradient_accumulate_float32(
    float* buffer,
    const float* grad,
    std::int64_t n,
    float scale,
    void* cuda_stream);

int nfn_native_tile_fill_float32(
    float* values,
    std::int64_t n,
    float value,
    void* cuda_stream);

int nfn_native_tile_tanh_float32(
    const float* x,
    float* out,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_tanh_backward_float32(
    const float* grad_out,
    const float* tanh_out,
    float* grad_x,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_add_float32(
    const float* lhs,
    const float* rhs,
    float* out,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_vector_binary_float32(
    const float* lhs,
    const float* rhs,
    const float* scale0,
    const float* scale1,
    float* out,
    std::int64_t n,
    std::int64_t dim,
    std::int64_t op,
    void* cuda_stream);

int nfn_native_tile_mhc_beta_gradient_float32(
    const float* beta_logit,
    const float* input,
    const float* attention_proj,
    const float* residual1,
    const float* ffn_out,
    const float* grad_second,
    const float* grad_first,
    float* grad_beta_logit,
    std::int64_t rows,
    std::int64_t model_dim,
    float scale,
    void* cuda_stream);

int nfn_native_tile_fill_many_float32(
    float* const* buffers,
    const std::int64_t* elements,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    float value,
    void* cuda_stream);

int nfn_native_tile_fill_many_values_float32(
    float* const* buffers,
    const std::int64_t* elements,
    const float* values,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    void* cuda_stream);

int nfn_native_tile_fill_many_values_bf16_bits_float32(
    std::uint16_t* const* buffers,
    const std::int64_t* elements,
    const float* values,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    void* cuda_stream);

int nfn_native_tile_fill_many_values_mixed_float32_bf16_bits(
    float* const* float_buffers,
    const std::int64_t* float_elements,
    const float* float_values,
    std::int64_t float_buffer_count,
    std::int64_t float_max_elements,
    std::uint16_t* const* bf16_buffers,
    const std::int64_t* bf16_elements,
    const float* bf16_values,
    std::int64_t bf16_buffer_count,
    std::int64_t bf16_max_elements,
    void* cuda_stream);

int nfn_native_tile_init_gpt2_token_weight_float32(
    float* values,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_seeded_normal_float32(
    float* values,
    std::uint16_t* shadow_bf16_bits,
    std::int64_t n,
    std::uint64_t seed,
    std::uint64_t offset,
    float stddev,
    void* cuda_stream);

int nfn_native_tile_init_gpt2_token_weight_with_bf16_shadow_float32(
    float* values,
    std::uint16_t* shadow_bf16_bits,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_copy_float32(
    const float* source,
    float* dest,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_evo_mutate_candidates_float32(
    const float* base,
    float* candidates,
    std::int64_t elements,
    std::int64_t candidate_count,
    float mutation_scale,
    std::int64_t seed,
    void* cuda_stream);

int nfn_native_tile_evo_select_best_loss_float32(
    const float* losses,
    std::int64_t candidate_count,
    std::int64_t* best_index,
    float* best_loss,
    void* cuda_stream);

int nfn_native_tile_evo_adopt_candidate_float32(
    const float* candidates,
    const std::int64_t* best_index,
    float* target,
    std::int64_t elements,
    std::int64_t candidate_count,
    void* cuda_stream);

int nfn_native_tile_lm_head_classifier_backward_loss_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* loss_total,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream);

int nfn_native_tile_lm_head_classifier_backward_row_losses_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* row_losses,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream);

int nfn_native_tile_lm_head_classifier_backward_loss_bins_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* loss_bins,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    std::int64_t loss_bin_count,
    float loss_scale,
    void* cuda_stream);

int nfn_native_tile_lm_head_classifier_backward_prob_only_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream);

int nfn_native_tile_lm_head_classifier_backward_prob_only_ce_target_correction_bf16_bits(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    const std::uint16_t* token_weight_bf16,
    const std::uint16_t* hidden_bf16,
    float* grad_hidden,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    std::int64_t hidden_dim,
    std::int64_t token_weight_row_stride,
    std::int64_t grad_weight_row_stride,
    float loss_scale,
    void* cuda_stream);

int nfn_native_tile_lm_head_prob_only_dhidden_target_correction_bf16_bits(
    const std::uint16_t* targets,
    const std::uint16_t* token_weight_bf16,
    float* grad_hidden,
    std::int64_t rows,
    std::int64_t hidden_dim,
    std::int64_t token_weight_row_stride,
    float loss_scale,
    void* cuda_stream);

int nfn_native_tile_lm_head_prob_only_dweight_target_correction_bf16_bits(
    const std::uint16_t* targets,
    const std::uint16_t* hidden_bf16,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t hidden_dim,
    std::int64_t grad_weight_row_stride,
    float loss_scale,
    void* cuda_stream);

int nfn_native_tile_lm_head_prob_only_combined_target_correction_bf16_bits(
    const std::uint16_t* targets,
    const std::uint16_t* token_weight_bf16,
    const std::uint16_t* hidden_bf16,
    float* grad_hidden,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t hidden_dim,
    std::int64_t token_weight_row_stride,
    std::int64_t grad_weight_row_stride,
    float loss_scale,
    void* cuda_stream);

int nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_float32_beta(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    std::int64_t grad_out_row_stride,
    float beta,
    void* cuda_stream);

std::int64_t nfn_native_tile_lm_head_cooperative_sequence_launch_count();
std::int64_t nfn_native_tile_lm_head_cooperative_sequence_ce_launch_count();
std::int64_t nfn_native_tile_lm_head_cooperative_sequence_dhidden_launch_count();
std::int64_t nfn_native_tile_lm_head_cooperative_sequence_dweight_launch_count();
std::int64_t nfn_native_tile_lm_head_cooperative_sequence_concurrent_count();
std::int64_t nfn_native_tile_lm_head_cooperative_sequence_legacy_count();
std::int64_t nfn_native_tile_lm_head_cooperative_sequence_loss_bin_count();
std::int64_t nfn_native_tile_lm_head_fused_graph_capture_attempt_count();
std::int64_t nfn_native_tile_lm_head_fused_graph_capture_success_count();
std::int64_t nfn_native_tile_lm_head_fused_graph_upload_success_count();
std::int64_t nfn_native_tile_lm_head_fused_graph_upload_failure_count();
std::int64_t nfn_native_tile_lm_head_fused_graph_cache_hit_count();
std::int64_t nfn_native_tile_lm_head_fused_graph_thread_cache_hit_count();
std::int64_t nfn_native_tile_lm_head_fused_graph_cache_entry_count();
std::int64_t nfn_native_tile_lm_head_fused_graph_replay_count();
std::int64_t nfn_native_tile_lm_head_fused_graph_replay_success_count();
std::int64_t nfn_native_tile_lm_head_fused_graph_fallback_count();
std::int64_t nfn_native_tile_lm_head_graph_body_cublaslt_dhidden_launch_count();
std::int64_t nfn_native_tile_lm_head_graph_body_cublaslt_dweight_launch_count();
std::int64_t nfn_native_tile_lm_head_graph_body_tile_dhidden_fallback_count();
std::int64_t nfn_native_tile_lm_head_graph_body_tile_dweight_fallback_count();

int nfn_native_tile_lm_head_classifier_backward_fused_graph_prewarm_bf16_u16(
    std::uint16_t* logits_bf16,
    const std::uint16_t* targets_u16,
    float* row_losses,
    const std::uint16_t* hidden_bf16,
    const float* hidden_float,
    const std::uint16_t* token_weight_bf16,
    const float* token_weight_float,
    float* grad_hidden,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t hidden_dim,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    float dweight_beta,
    int flags,
    void* cuda_stream);

int nfn_native_tile_lm_head_classifier_backward_cooperative_bf16_u16(
    std::uint16_t* logits_bf16,
    const std::uint16_t* targets_u16,
    float* row_losses,
    const std::uint16_t* hidden_bf16,
    const float* hidden_float,
    const std::uint16_t* token_weight_bf16,
    const float* token_weight_float,
    float* grad_hidden,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t hidden_dim,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    float dweight_beta,
    int flags,
    void* cuda_stream);

int nfn_native_tile_lm_head_classifier_backward_cooperative_fused_bf16_u16(
    std::uint16_t* logits_bf16,
    const std::uint16_t* targets_u16,
    float* row_losses,
    const std::uint16_t* hidden_bf16,
    const float* hidden_float,
    const std::uint16_t* token_weight_bf16,
    const float* token_weight_float,
    float* grad_hidden,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t hidden_dim,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    float dweight_beta,
    int flags,
    void* cuda_stream);

int nfn_native_tile_lm_head_classifier_backward_cooperative_cublaslt_bf16_u16(
    std::uint16_t* logits_bf16,
    const std::uint16_t* targets_u16,
    float* row_losses,
    const std::uint16_t* hidden_bf16,
    const float* hidden_float,
    const std::uint16_t* token_weight_bf16,
    const float* token_weight_float,
    float* grad_hidden,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t hidden_dim,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    float dweight_beta,
    int flags,
    void* cuda_stream);

int nfn_native_tile_lm_head_classifier_backward_fused_kernel_bf16_u16(
    std::uint16_t* logits_bf16,
    const std::uint16_t* targets_u16,
    float* row_losses,
    const std::uint16_t* hidden_bf16,
    const float* hidden_float,
    const std::uint16_t* token_weight_bf16,
    const float* token_weight_float,
    float* grad_hidden,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t hidden_dim,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    float dweight_beta,
    int flags,
    void* cuda_stream);

int nfn_native_tile_lm_head_classifier_backward_fused_kernel_is_true_fused();
const char* nfn_native_tile_lm_head_classifier_backward_fused_kernel_path_class();
const char* nfn_native_tile_lm_head_classifier_backward_fused_kernel_implementation_class();
int nfn_native_tile_lm_head_classifier_backward_fused_kernel_graph_body_node_count();
int nfn_native_tile_lm_head_classifier_backward_fused_kernel_graph_body_ce_node_count();
int nfn_native_tile_lm_head_classifier_backward_fused_kernel_graph_body_dhidden_node_count();
int nfn_native_tile_lm_head_classifier_backward_fused_kernel_graph_body_dweight_node_count();
int nfn_native_tile_lm_head_classifier_backward_llmk_classifier_matmul_parity();
std::int64_t nfn_native_tile_lm_head_true_fused_ce_cycles();
std::int64_t nfn_native_tile_lm_head_true_fused_dhidden_cycles();
std::int64_t nfn_native_tile_lm_head_true_fused_dweight_cycles();
std::int64_t nfn_native_tile_lm_head_true_fused_ce_blocks();
std::int64_t nfn_native_tile_lm_head_true_fused_dhidden_blocks();
std::int64_t nfn_native_tile_lm_head_true_fused_dweight_blocks();

int nfn_native_tile_lm_head_classifier_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* row_max_workspace,
    float* row_denom_workspace,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream);

int nfn_native_tile_uint16_to_int64(
    const std::uint16_t* source,
    std::int64_t* dest,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_uint32_to_int64(
    const std::uint32_t* source,
    std::int64_t* dest,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_uint8_to_int64(
    const std::uint8_t* source,
    std::int64_t* dest,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_diffusion_mask_u16_int64(
    const std::uint16_t* source_tokens,
    std::uint16_t* masked_tokens,
    std::int64_t* targets,
    std::int64_t rows,
    std::int64_t seq_len,
    std::int64_t vocab,
    void* cuda_stream);

int nfn_native_tile_float32_to_bf16_bits(
    const float* source,
    std::uint16_t* dest,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_bf16_bits_to_float32(
    const std::uint16_t* source,
    float* dest,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_float32_to_nvfp4_packed(
    const float* source,
    std::uint8_t* packed,
    std::uint8_t* block_scales_e4m3,
    float tensor_scale,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_nvfp4_packed_to_float32(
    const std::uint8_t* packed,
    const std::uint8_t* block_scales_e4m3,
    float tensor_scale,
    float* dest,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_linear_nvfp4_input_weight_bf16_float32(
    const std::uint8_t* x_nvfp4_packed,
    const std::uint8_t* x_block_scales_e4m3,
    float x_tensor_scale,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    void* cuda_stream);

int nfn_native_tile_linear_nvfp4_input_weight_bf16_output_float32(
    const std::uint8_t* x_nvfp4_packed,
    const std::uint8_t* x_block_scales_e4m3,
    float x_tensor_scale,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    std::uint16_t* out_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    void* cuda_stream);

int nfn_native_tile_linear_backward_weight_accumulate_nvfp4_input_float32_beta(
    const std::uint8_t* x_nvfp4_packed,
    const std::uint8_t* x_block_scales_e4m3,
    float x_tensor_scale,
    const float* grad_out,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    void* cuda_stream);

int nfn_native_tile_linear_backward_weight_accumulate_nvfp4_input_bf16_grad_float32_beta(
    const std::uint8_t* x_nvfp4_packed,
    const std::uint8_t* x_block_scales_e4m3,
    float x_tensor_scale,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    void* cuda_stream);

int nfn_native_tile_store_mlp_activations_bf16_float32(
    const float* ln2_out,
    const float* fc_out,
    const float* act,
    std::uint16_t* dest,
    std::int64_t activation_elements,
    std::int64_t hidden_elements,
    void* cuda_stream);

int nfn_native_tile_restore_mlp_activations_bf16_float32(
    const std::uint16_t* source,
    float* ln2_out,
    float* fc_out,
    float* act,
    std::int64_t activation_elements,
    std::int64_t hidden_elements,
    void* cuda_stream);

int nfn_native_tile_float32_to_bf16_bits_many(
    const float* const* sources,
    const std::int64_t* elements,
    const std::int64_t* offsets,
    std::uint16_t* dest,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    void* cuda_stream);

int nfn_native_tile_init_gpt2_token_weight_fast_float32(
    float* values,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_init_gpt2_token_weight_fast_with_bf16_shadow_float32(
    float* values,
    std::uint16_t* shadow_bf16_bits,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_init_gpt2_token_weight_fast_with_bf16_shadow_padded_float32(
    float* values,
    std::uint16_t* shadow_bf16_bits,
    std::int64_t public_n,
    std::int64_t total_n,
    void* cuda_stream);

int nfn_native_tile_sumsq_partials_float32(
    const float* values,
    float* partials,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_sumsq_partials_many_float32(
    const float* const* buffers,
    const std::int64_t* elements,
    const std::int64_t* partial_offsets,
    float* partials,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    void* cuda_stream);

int nfn_native_tile_sumsq_partials_many_bf16_bits_float32(
    const std::uint16_t* const* buffers,
    const std::int64_t* elements,
    const std::int64_t* partial_offsets,
    float* partials,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    void* cuda_stream);

int nfn_native_tile_optimizer_tile_size();

int nfn_native_tile_sum_partials_float32(
    const float* values,
    float* partials,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_sum_accumulate_float32(
    const float* values,
    float* total,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_extract_diagonal_float32(
    const float* matrix,
    float* diagonal,
    std::int64_t dim,
    void* cuda_stream);

int nfn_native_tile_scale_inplace_float32(
    float* values,
    std::int64_t n,
    float scale,
    void* cuda_stream);

int nfn_native_tile_global_norm_clip_scale_float32(
    const float* sumsq_partials,
    float* clip_scale,
    std::int64_t partial_count,
    float max_norm,
    float eps,
    void* cuda_stream);

int nfn_native_tile_scale_inplace_by_device_float32(
    float* values,
    const float* scale,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_scaled_residual_add_float32(
    const float* lhs,
    const float* rhs,
    const float* scale,
    float* out,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_split_qkv_float32(
    const float* qkv,
    float* q,
    float* k,
    float* v,
    std::int64_t rows,
    std::int64_t dim,
    void* cuda_stream);

int nfn_native_tile_split_qkv_to_heads_float32(
    const float* qkv,
    float* q_heads,
    float* k_heads,
    float* v_heads,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t heads,
    std::int64_t head_dim,
    void* cuda_stream);

int nfn_native_tile_split_qkv_to_heads_add_bias_float32(
    const float* qkv,
    const float* bias,
    float* q_heads,
    float* k_heads,
    float* v_heads,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t heads,
    std::int64_t head_dim,
    void* cuda_stream);

int nfn_native_tile_merge_qkv_float32(
    const float* q,
    const float* k,
    const float* v,
    float* qkv,
    std::int64_t rows,
    std::int64_t dim,
    void* cuda_stream);

int nfn_native_tile_merge_heads_to_qkv_float32(
    const float* q_heads,
    const float* k_heads,
    const float* v_heads,
    float* qkv,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t heads,
    std::int64_t head_dim,
    void* cuda_stream);

int nfn_native_tile_reshape_heads_float32(
    const float* x,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t heads,
    std::int64_t head_dim,
    void* cuda_stream);

int nfn_native_tile_merge_heads_float32(
    const float* x,
    float* out,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    void* cuda_stream);

int nfn_native_tile_repeat_kv_float32(
    const float* input,
    float* output,
    std::int64_t batch,
    std::int64_t kv_heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    std::int64_t repeats,
    void* cuda_stream);

int nfn_native_tile_repeat_kv_backward_float32(
    const float* grad_output,
    float* grad_input,
    std::int64_t batch,
    std::int64_t kv_heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    std::int64_t repeats,
    void* cuda_stream);

int nfn_native_tile_byte_patch_embed_float32(
    const std::int64_t* tokens,
    const float* embedding,
    const float* proj,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    std::int64_t patch_size,
    std::int64_t stride,
    std::int64_t out_len,
    std::int64_t vocab_size,
    void* cuda_stream);

int nfn_native_tile_byte_patch_merge_float32(
    const float* x,
    float* out,
    std::int64_t batch,
    std::int64_t source_len,
    std::int64_t target_len,
    std::int64_t dim,
    void* cuda_stream);

int nfn_native_tile_byte_patch_merge_backward_float32(
    const float* grad_out,
    float* grad_x,
    std::int64_t batch,
    std::int64_t source_len,
    std::int64_t target_len,
    std::int64_t dim,
    void* cuda_stream);

int nfn_native_tile_byte_patch_embed_backward_float32(
    const std::int64_t* tokens,
    const float* embedding,
    const float* proj,
    const float* grad_out,
    float* grad_embedding,
    float* grad_proj,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    std::int64_t patch_size,
    std::int64_t stride,
    std::int64_t out_len,
    std::int64_t vocab_size,
    void* cuda_stream);

int nfn_native_tile_causal_chunk_state_float32(
    const float* hidden,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t dim,
    std::int64_t chunk_size,
    std::int64_t chunks,
    std::int64_t mode,
    void* cuda_stream);

int nfn_native_tile_causal_chunk_state_backward_float32(
    const float* grad_out,
    float* grad_hidden,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t dim,
    std::int64_t chunk_size,
    std::int64_t chunks,
    std::int64_t mode,
    void* cuda_stream);

int nfn_native_tile_topk_route_float32(
    const float* logits,
    float* weights,
    std::int64_t* indices,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t top_k,
    void* cuda_stream);

int nfn_native_tile_topk_route_sqrt_softplus_float32(
    const float* logits,
    float* weights,
    std::int64_t* indices,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t top_k,
    void* cuda_stream);

int nfn_native_tile_topk_route_backward_float32(
    const float* weights,
    const std::int64_t* indices,
    const float* grad_weights,
    float* grad_logits,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t top_k,
    float route_scale,
    void* cuda_stream);

int nfn_native_tile_semantic_shared_topk_route_float32(
    const float* logits,
    float* weights,
    std::int64_t* indices,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t shared_experts,
    std::int64_t top_k,
    void* cuda_stream);

int nfn_native_tile_semantic_shared_forced_topk_route_float32(
    const float* logits,
    const std::int64_t* semantic_target_matrix,
    float* weights,
    std::int64_t* indices,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t semantic_vocab_dims,
    std::int64_t shared_experts,
    std::int64_t top_k,
    std::int64_t ignore_index,
    void* cuda_stream);

int nfn_native_tile_semantic_shared_topk_route_backward_float32(
    const float* weights,
    const std::int64_t* indices,
    const float* grad_weights,
    float* grad_logits,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t shared_experts,
    std::int64_t top_k,
    float route_scale,
    void* cuda_stream);

int nfn_native_tile_topk_route_sqrt_softplus_backward_float32(
    const float* logits,
    const float* weights,
    const std::int64_t* indices,
    const float* grad_weights,
    float* grad_logits,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t top_k,
    float route_scale,
    void* cuda_stream);

int nfn_native_tile_broadcast_expert_routes_float32(
    const float* weights,
    const std::int64_t* indices,
    float* out_weights,
    std::int64_t* out_indices,
    std::int64_t batch,
    std::int64_t route_seq,
    std::int64_t seq_len,
    std::int64_t route_width,
    void* cuda_stream);

int nfn_native_tile_broadcast_chunk_routes_float32(
    const float* weights,
    const std::int64_t* indices,
    float* out_weights,
    std::int64_t* out_indices,
    std::int64_t batch,
    std::int64_t chunks,
    std::int64_t seq_len,
    std::int64_t route_width,
    std::int64_t chunk_size,
    void* cuda_stream);

int nfn_native_tile_compact_chunk_routes_float32_int64(
    const float* weights,
    const std::int64_t* indices,
    float* chunk_weights,
    std::int64_t* chunk_indices,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t chunks,
    std::int64_t route_width,
    std::int64_t chunk_size,
    void* cuda_stream);

int nfn_native_tile_aggregate_chunk_route_gradients_float32(
    const float* grad_weights,
    float* aggregated_grad_weights,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t route_width,
    std::int64_t chunk_size,
    void* cuda_stream);

int nfn_native_tile_semantic_route_distillation_backward_float32(
    const float* route_logits,
    const std::int64_t* semantic_targets,
    const std::uint8_t* semantic_target_valid,
    float* grad_route_logits,
    float* loss_items,
    std::int64_t rows,
    std::int64_t seq_len,
    std::int64_t experts,
    std::int64_t semantic_vocab_dims,
    std::int64_t shared_experts,
    std::int64_t route_chunk_size,
    float distill_weight,
    float teacher_target,
    void* cuda_stream);

int nfn_native_tile_semantic_target_topic_distillation_backward_float32(
    const float* route_logits,
    const float* target_topic_logits,
    float* grad_route_logits,
    float* loss_items,
    std::int64_t rows,
    std::int64_t seq_len,
    std::int64_t experts,
    std::int64_t semantic_vocab_dims,
    std::int64_t shared_experts,
    std::int64_t route_chunk_size,
    float distill_weight,
    void* cuda_stream);

int nfn_native_tile_semantic_target_topic_packed_distillation_backward_float32(
    const float* route_logits,
    const float* target_topic_logits,
    const std::int64_t* term_counts,
    const std::int64_t* term_offsets,
    float* grad_route_logits,
    float* loss_items,
    std::int64_t rows,
    std::int64_t seq_len,
    std::int64_t experts,
    std::int64_t semantic_vocab_dims,
    std::int64_t total_terms,
    std::int64_t shared_experts,
    std::int64_t route_chunk_size,
    float distill_weight,
    void* cuda_stream);

int nfn_native_tile_semantic_hash_table_backward_float32(
    const std::int64_t* hash_indices,
    const float* hash_embedding,
    const float* table_gate_logits,
    const float* grad_route_logits,
    float* grad_hash_embedding,
    float* grad_table_gate,
    float* grad_dimension_bias,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t semantic_vocab_dims,
    std::int64_t shared_experts,
    std::int64_t tables,
    std::int64_t buckets,
    void* cuda_stream);

int nfn_native_tile_semantic_route_policy_float32(
    float* route_logits,
    const std::int64_t* hash_indices,
    const float* hash_embedding,
    const float* table_gate_logits,
    const float* dimension_bias,
    const std::int64_t* semantic_targets,
    const std::uint8_t* semantic_target_valid,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t semantic_vocab_dims,
    std::int64_t shared_experts,
    std::int64_t tables,
    std::int64_t buckets,
    std::int64_t top_k,
    float target_boost,
    void* cuda_stream);

int nfn_native_tile_semantic_route_policy_packed_topic_float32(
    float* route_logits,
    const std::int64_t* hash_indices,
    const float* hash_embedding,
    const float* table_gate_logits,
    const float* dimension_bias,
    const float* topic_logits,
    const std::int64_t* term_counts,
    const std::int64_t* term_offsets,
    const std::int64_t* semantic_targets,
    const std::uint8_t* semantic_target_valid,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t semantic_vocab_dims,
    std::int64_t total_terms,
    std::int64_t shared_experts,
    std::int64_t tables,
    std::int64_t buckets,
    std::int64_t top_k,
    float target_boost,
    void* cuda_stream);

int nfn_native_tile_semantic_route_policy_packed_topic_matrix_float32(
    float* route_logits,
    const std::int64_t* hash_indices,
    const float* hash_embedding,
    const float* table_gate_logits,
    const float* dimension_bias,
    const float* topic_logits,
    const std::int64_t* term_counts,
    const std::int64_t* term_offsets,
    const std::int64_t* semantic_target_matrix,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t semantic_vocab_dims,
    std::int64_t total_terms,
    std::int64_t shared_experts,
    std::int64_t tables,
    std::int64_t buckets,
    std::int64_t top_k,
    float target_boost,
    std::int64_t ignore_index,
    void* cuda_stream);

int nfn_native_tile_semantic_vec_from_packed_topic_float32(
    const float* topic_logits,
    const std::int64_t* term_counts,
    const std::int64_t* term_offsets,
    float* semantic_vec,
    std::int64_t rows,
    std::int64_t semantic_vocab_dims,
    std::int64_t total_terms,
    void* cuda_stream);

int nfn_native_tile_semantic_packed_topic_to_padded_float32(
    const float* packed_logits,
    const std::int64_t* term_counts,
    const std::int64_t* term_offsets,
    float* padded_logits,
    std::int64_t rows,
    std::int64_t semantic_vocab_dims,
    std::int64_t total_terms,
    std::int64_t max_terms,
    void* cuda_stream);

int nfn_native_tile_semantic_signature_scalar_float32(
    const float* sig_logits,
    float* signature_scalar,
    std::int64_t rows,
    std::int64_t buckets,
    void* cuda_stream);

int nfn_native_tile_semantic_vec_append_signature_float32(
    const float* topic_vec,
    const float* signature_scalar,
    float* semantic_vec,
    std::int64_t rows,
    std::int64_t topic_dims,
    void* cuda_stream);

int nfn_native_tile_semantic_vec_split_signature_grad_float32(
    const float* grad_semantic_vec,
    float* grad_topic_vec,
    float* grad_signature_scalar,
    std::int64_t rows,
    std::int64_t topic_dims,
    void* cuda_stream);

int nfn_native_tile_semantic_signature_scalar_backward_float32(
    const float* sig_logits,
    const float* signature_scalar,
    const float* grad_signature_scalar,
    float* grad_sig_logits,
    std::int64_t rows,
    std::int64_t buckets,
    void* cuda_stream);

int nfn_native_tile_semantic_free_expert_projection_float32(
    const float* semantic_vec,
    const float* free_weight,
    float* route_logits,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t semantic_vocab_dims,
    std::int64_t semantic_vec_dims,
    std::int64_t semantic_shared_experts,
    std::int64_t semantic_free_experts,
    std::int64_t weight_stride,
    void* cuda_stream);

int nfn_native_tile_semantic_shared_expert_projection_float32(
    const float* semantic_vec,
    const float* shared_weight,
    float* route_logits,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t semantic_vocab_dims,
    std::int64_t semantic_shared_experts,
    std::int64_t weight_stride,
    void* cuda_stream);

int nfn_native_tile_semantic_free_expert_projection_backward_float32(
    const float* semantic_vec,
    const float* free_weight,
    const float* grad_route_logits,
    float* grad_semantic_vec,
    float* grad_free_weight,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t semantic_vocab_dims,
    std::int64_t semantic_vec_dims,
    std::int64_t semantic_shared_experts,
    std::int64_t semantic_free_experts,
    std::int64_t weight_stride,
    void* cuda_stream);

int nfn_native_tile_semantic_shared_expert_projection_backward_float32(
    const float* semantic_vec,
    const float* shared_weight,
    const float* grad_route_logits,
    float* grad_semantic_vec,
    float* grad_shared_weight,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t semantic_vocab_dims,
    std::int64_t semantic_shared_experts,
    std::int64_t weight_stride,
    void* cuda_stream);

int nfn_native_tile_semantic_router_bias_add_float32(
    float* route_logits,
    const float* shared_logits,
    const float* free_bias,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t semantic_vocab_dims,
    std::int64_t semantic_shared_experts,
    std::int64_t semantic_free_experts,
    void* cuda_stream);

int nfn_native_tile_semantic_router_bias_backward_float32(
    const float* grad_route_logits,
    float* grad_shared_logits,
    float* grad_free_bias,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t semantic_vocab_dims,
    std::int64_t semantic_shared_experts,
    std::int64_t semantic_free_experts,
    void* cuda_stream);

int nfn_native_tile_semantic_targets_from_matrix_int64(
    const std::int64_t* semantic_matrix,
    const std::int64_t* lm_targets,
    std::int64_t* semantic_targets,
    std::uint8_t* semantic_target_valid,
    std::int64_t rows,
    std::int64_t semantic_dims,
    std::int64_t semantic_vocab_dims,
    void* cuda_stream);

int nfn_native_tile_semantic_targets_from_tokens_u16_int64(
    const std::uint16_t* tokens,
    const std::int64_t* lm_targets,
    std::int64_t* semantic_targets,
    std::uint8_t* semantic_target_valid,
    std::int64_t rows,
    std::int64_t semantic_dims,
    std::int64_t semantic_terms,
    std::int64_t semantic_vocab_dims,
    void* cuda_stream);

int nfn_native_tile_semantic_target_matrix_from_tokens_u16_int64(
    const std::uint16_t* tokens,
    std::int64_t* semantic_matrix,
    const std::int64_t* term_counts,
    std::int64_t rows,
    std::int64_t semantic_dims,
    std::int64_t ignore_index,
    void* cuda_stream);

int nfn_native_tile_moe_swiglu_forward_float32(
    const float* x,
    const float* route_weights,
    const std::int64_t* route_indices,
    const float* w1,
    const float* w2,
    const float* w3,
    float* out,
    std::int64_t tokens,
    std::int64_t dim,
    std::int64_t hidden_dim,
    std::int64_t experts,
    std::int64_t top_k,
    void* cuda_stream);

int nfn_native_tile_moe_swiglu_backward_float32(
    const float* x,
    const float* route_weights,
    const std::int64_t* route_indices,
    const float* w1,
    const float* w2,
    const float* w3,
    const float* grad_out,
    float* grad_x,
    float* grad_w1,
    float* grad_w2,
    float* grad_w3,
    std::int64_t tokens,
    std::int64_t dim,
    std::int64_t hidden_dim,
    std::int64_t experts,
    std::int64_t top_k,
    void* cuda_stream);

int nfn_native_tile_moe_swiglu_backward_with_route_grad_float32(
    const float* x,
    const float* route_weights,
    const std::int64_t* route_indices,
    const float* w1,
    const float* w2,
    const float* w3,
    const float* grad_out,
    float* grad_x,
    float* grad_w1,
    float* grad_w2,
    float* grad_w3,
    float* grad_route_weights,
    std::int64_t tokens,
    std::int64_t dim,
    std::int64_t hidden_dim,
    std::int64_t experts,
    std::int64_t top_k,
    void* cuda_stream);

int nfn_native_tile_moe_swiglu_forward_quantized_float32(
    const float* x,
    const float* route_weights,
    const std::int64_t* route_indices,
    const float* w1,
    const float* w2,
    const float* w3,
    float* out,
    std::int64_t tokens,
    std::int64_t dim,
    std::int64_t hidden_dim,
    std::int64_t experts,
    std::int64_t top_k,
    std::int64_t quantization_kind,
    void* cuda_stream);

int nfn_native_tile_moe_swiglu_backward_quantized_float32(
    const float* x,
    const float* route_weights,
    const std::int64_t* route_indices,
    const float* w1,
    const float* w2,
    const float* w3,
    const float* grad_out,
    float* grad_x,
    float* grad_w1,
    float* grad_w2,
    float* grad_w3,
    float* grad_route_weights,
    std::int64_t tokens,
    std::int64_t dim,
    std::int64_t hidden_dim,
    std::int64_t experts,
    std::int64_t top_k,
    std::int64_t quantization_kind,
    void* cuda_stream);

int nfn_native_tile_semantic_hash_int64(
    const float* sem_vec,
    const float* proj,
    std::int64_t* out,
    std::int64_t batch,
    std::int64_t dim,
    std::int64_t tables,
    std::int64_t planes,
    void* cuda_stream);

int nfn_native_tile_attentionless_decoder_float32(
    const std::int64_t* bucket_indices,
    const float* expert_output,
    const float* bucket_embed,
    const float* out_weight,
    float* out,
    std::int64_t batch,
    std::int64_t residual_dim,
    std::int64_t vocab_size,
    std::int64_t n_buckets,
    void* cuda_stream);

int nfn_native_tile_expert_bias_add_float32(
    const float* logits,
    const float* bias,
    float* out,
    std::int64_t n,
    std::int64_t experts,
    void* cuda_stream);

int nfn_native_tile_adamw_step_float32(
    float* param,
    const float* grad,
    float* exp_avg,
    float* exp_avg_sq,
    std::int64_t n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float sqrt_bias_correction2,
    void* cuda_stream);

int nfn_native_tile_adamw_step_with_device_scale_float32(
    float* param,
    const float* grad,
    const float* grad_scale,
    float* exp_avg,
    float* exp_avg_sq,
    std::int64_t n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float sqrt_bias_correction2,
    void* cuda_stream);

int nfn_native_tile_adamw_step_many_with_device_scale_float32(
    float* const* params,
    const float* const* grads,
    const float* grad_scale,
    float* const* exp_avgs,
    float* const* exp_avg_sqs,
    const std::int64_t* elements,
    const float* weight_decays,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bias_correction1,
    float sqrt_bias_correction2,
    void* cuda_stream);

int nfn_native_tile_adamw_step_many_with_device_scale_hyper_float32(
    float* const* params,
    const float* const* grads,
    const float* grad_scale,
    float* const* exp_avgs,
    float* const* exp_avg_sqs,
    const std::int64_t* elements,
    const float* weight_decays,
    const float* hyper,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    void* cuda_stream);

int nfn_native_tile_adamw_step_many_with_device_scale_bf16_shadow_float32(
    float* const* params,
    const float* const* grads,
    const float* grad_scale,
    float* const* exp_avgs,
    float* const* exp_avg_sqs,
    const std::int64_t* elements,
    const float* weight_decays,
    const std::int64_t* bf16_shadow_offsets,
    std::uint16_t* bf16_shadow_bits,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bias_correction1,
    float sqrt_bias_correction2,
    void* cuda_stream);

int nfn_native_tile_adamw_step_many_with_device_scale_bf16_shadow_hyper_float32(
    float* const* params,
    const float* const* grads,
    const float* grad_scale,
    float* const* exp_avgs,
    float* const* exp_avg_sqs,
    const std::int64_t* elements,
    const float* weight_decays,
    const std::int64_t* bf16_shadow_offsets,
    std::uint16_t* bf16_shadow_bits,
    const float* hyper,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    void* cuda_stream);

int nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_float32(
    std::uint16_t* const* params_bf16_bits,
    const float* const* grads,
    const float* grad_scale,
    float* const* exp_avgs,
    float* const* exp_avg_sqs,
    const std::int64_t* elements,
    const float* weight_decays,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bias_correction1,
    float sqrt_bias_correction2,
    void* cuda_stream);

int nfn_native_tile_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32(
    std::uint16_t* const* params_bf16_bits,
    const std::uint16_t* const* grads_bf16_bits,
    const float* grad_scale,
    float* const* exp_avgs,
    float* const* exp_avg_sqs,
    const std::int64_t* elements,
    const float* weight_decays,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bias_correction1,
    float sqrt_bias_correction2,
    void* cuda_stream);

int nfn_native_tile_linear_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    void* cuda_stream);

// kind: 1=ternary, 2=FP8 E4M3, 3=MXFP4 E2M1 with 32-value blocks.
int nfn_native_tile_linear_quantized_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    int kind,
    void* cuda_stream);

int nfn_native_tile_linear_backward_input_quantized_float32(
    const float* grad_out,
    const float* weight,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    int kind,
    void* cuda_stream);

int nfn_native_tile_fused_causal_attention_forward_float32(
    const float* x,
    const float* q_weight,
    const float* k_weight,
    const float* v_weight,
    const float* out_weight,
    const float* inv_freq,
    float* q_projection,
    float* k_projection,
    float* v_projection,
    float* q,
    float* k,
    float* v,
    float* q_rope,
    float* k_rope,
    float* attention,
    float* attention_flat,
    float* output,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    std::int64_t heads,
    std::int64_t kv_heads,
    std::int64_t head_dim,
    float scale,
    void* cuda_stream);

int nfn_native_tile_fused_causal_attention_backward_float32(
    const float* x,
    const float* q_weight,
    const float* k_weight,
    const float* v_weight,
    const float* out_weight,
    const float* inv_freq,
    const float* q_rope,
    const float* k_rope,
    const float* v,
    const float* attention_flat,
    const float* grad_output,
    float* grad_attention_flat,
    float* grad_attention,
    float* grad_q_rope,
    float* grad_k_rope,
    float* grad_v,
    float* grad_q,
    float* grad_k,
    float* grad_q_projection,
    float* grad_k_projection,
    float* grad_v_projection,
    float* grad_q_input,
    float* grad_k_input,
    float* grad_v_input,
    float* grad_q_weight,
    float* grad_k_weight,
    float* grad_v_weight,
    float* grad_out_weight,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    std::int64_t heads,
    std::int64_t kv_heads,
    std::int64_t head_dim,
    float scale,
    void* cuda_stream);

int nfn_native_tile_split_last_dim_float32(
    const float* input,
    float* first,
    float* second,
    std::int64_t rows,
    std::int64_t dim,
    void* cuda_stream);

int nfn_native_tile_merge_last_dim_float32(
    const float* first,
    const float* second,
    float* output,
    std::int64_t rows,
    std::int64_t half_dim,
    void* cuda_stream);

int nfn_native_tile_split_at_last_dim_float32(
    const float* input,
    float* first,
    float* second,
    std::int64_t rows,
    std::int64_t first_dim,
    std::int64_t second_dim,
    void* cuda_stream);

int nfn_native_tile_concat_last_dim_float32(
    const float* first,
    const float* second,
    float* output,
    std::int64_t rows,
    std::int64_t first_dim,
    std::int64_t second_dim,
    void* cuda_stream);

int nfn_native_tile_differential_combine_float32(
    const float* first,
    const float* second,
    float* output,
    std::int64_t elements,
    float lambda,
    float output_scale,
    void* cuda_stream);

int nfn_native_tile_differential_backward_float32(
    const float* grad_output,
    float* grad_first,
    float* grad_second,
    std::int64_t elements,
    float lambda,
    float output_scale,
    void* cuda_stream);

int nfn_native_tile_linear_bf16_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    void* cuda_stream);

int nfn_native_tile_linear_weight_bf16_float32(
    const float* x,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    void* cuda_stream);

int nfn_native_tile_linear_bf16_output_float32(
    const float* x,
    const float* weight,
    const float* bias,
    std::uint16_t* out_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    void* cuda_stream);

int nfn_native_tile_linear_weight_bf16_output_float32(
    const float* x,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    std::uint16_t* out_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    void* cuda_stream);

int nfn_native_tile_linear_bf16_input_weight_bf16_output_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    std::uint16_t* out_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    void* cuda_stream);

int nfn_native_tile_linear_bf16_input_float_weight_bf16_output_float32(
    const std::uint16_t* x_bf16_bits,
    const float* weight,
    const float* bias,
    std::uint16_t* out_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    void* cuda_stream);

int nfn_native_tile_bf16_bits_add_bias_inplace_float32(
    std::uint16_t* values,
    const float* bias,
    std::int64_t rows,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_bf16_input_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    void* cuda_stream);

int nfn_native_tile_linear_bf16_input_weight_bf16_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    void* cuda_stream);

int nfn_native_tile_linear_backward_input_float32(
    const float* grad_out,
    const float* weight,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_backward_input_bf16_float32(
    const float* grad_out,
    const float* weight,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_backward_input_weight_bf16_float32(
    const float* grad_out,
    const std::uint16_t* weight_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_float32(
    const std::uint16_t* grad_out_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_strided_float32(
    const std::uint16_t* grad_out_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    std::int64_t grad_out_row_stride,
    void* cuda_stream);

int nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_strided_cublaslt_float32(
    const std::uint16_t* grad_out_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    std::int64_t grad_out_row_stride,
    void* cuda_stream);

int nfn_native_tile_linear_backward_input_bf16_bits_float32(
    const std::uint16_t* grad_out_bf16_bits,
    const float* weight,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_backward_input_dgelu_bf16_bits_float32(
    const float* grad_out,
    const float* weight,
    const std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* grad_x_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_backward_input_dgelu_weight_bf16_bits_float32(
    const float* grad_out,
    const std::uint16_t* weight_bf16_bits,
    const std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* grad_x_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_backward_input_dgelu_weight_bf16_bits_only_float32(
    const float* grad_out,
    const std::uint16_t* weight_bf16_bits,
    const std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* grad_x_bf16_bits,
    float* grad_x_fallback,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_backward_input_dgelu_bf16_bits_weight_bf16_bits_only_float32(
    const std::uint16_t* grad_out_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    const std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* grad_x_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_backward_input_weight_bf16_to_bf16_bits_float32(
    const float* grad_out,
    const std::uint16_t* weight_bf16_bits,
    std::uint16_t* grad_x_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_backward_weight_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_backward_weight_accumulate_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_backward_weight_accumulate_bf16_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const float* grad_out,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const float* grad_out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_float32_beta(
    const std::uint16_t* x_bf16_bits,
    const float* grad_out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    void* cuda_stream);

int nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32_beta(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    void* cuda_stream);

int nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_to_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    std::uint16_t* grad_weight_bf16_bits,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    void* cuda_stream);

int nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_cublaslt_float32_beta(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    std::int64_t grad_out_row_stride,
    float beta,
    void* cuda_stream);

int nfn_native_tile_linear_backward_weight_bias_accumulate_float32_bf16_bits(
    const float* x,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_backward_weight_bias_accumulate_float32_bf16_bits_beta(
    const float* x,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    void* cuda_stream);

int nfn_native_tile_linear_backward_weight_accumulate_float32_bf16_bits(
    const float* x,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_backward_bias_float32(
    const float* grad_out,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_backward_bias_accumulate_float32(
    const float* grad_out,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_backward_bias_accumulate_bf16_bits_float32(
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_gelu_float32(
    const float* x,
    float* out,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_gelu_add_bias_float32(
    const float* x,
    const float* bias,
    float* biased_out,
    float* gelu_out,
    std::int64_t rows,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_gelu_add_bias_bf16_act_float32(
    const float* x,
    const float* bias,
    float* biased_out,
    float* gelu_out,
    std::uint16_t* gelu_bf16_bits,
    std::int64_t rows,
    std::int64_t output_dim,
    void* cuda_stream);

// Shared-backbone MoA activation dispatch: kind 0=GELU, 1=ReLU, 2=SiLU, 3=ReLU^2.
int nfn_native_tile_moa_add_bias_float32(
    const float* x,
    const float* bias,
    float* biased_out,
    float* activation_out,
    std::int64_t rows,
    std::int64_t output_dim,
    int activation_kind,
    void* cuda_stream);

int nfn_native_tile_moa_add_bias_bf16_act_float32(
    const float* x,
    const float* bias,
    float* biased_out,
    float* activation_out,
    std::uint16_t* activation_bf16_bits,
    std::int64_t rows,
    std::int64_t output_dim,
    int activation_kind,
    void* cuda_stream);

int nfn_native_tile_swiglu_float32(
    const float* gate,
    const float* up,
    float* out,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_linear_bf16_gelu_bf16_float32(
    const float* x,
    const float* weight,
    const float* bias,
    std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* gelu_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_weight_bf16_gelu_bf16_float32(
    const float* x,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* gelu_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_bf16_input_weight_bf16_gelu_bf16_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* gelu_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_bias_residual_add_float32(
    const float* residual,
    const float* linear_out,
    const float* bias,
    const float* residual_scale,
    float* out,
    std::int64_t rows,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_bias_residual_add_bf16_linear_float32(
    const float* residual,
    const std::uint16_t* linear_out_bf16_bits,
    const float* bias,
    const float* residual_scale,
    float* out,
    std::int64_t rows,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_bias_residual_add_bf16_linear_bf16_residual_float32(
    const float* residual,
    const std::uint16_t* linear_out_bf16_bits,
    const float* bias,
    const float* residual_scale,
    float* out,
    std::uint16_t* residual_bf16_out,
    std::int64_t rows,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_linear_bias_residual_layer_norm_float32(
    const float* residual,
    const float* linear_out,
    const float* linear_bias,
    const float* residual_scale,
    const float* norm_weight,
    const float* norm_bias,
    float* residual_out,
    float* norm_out,
    std::int64_t rows,
    std::int64_t output_dim,
    float eps,
    void* cuda_stream);

int nfn_native_tile_linear_bias_residual_layer_norm_with_stats_float32(
    const float* residual,
    const float* linear_out,
    const float* linear_bias,
    const float* residual_scale,
    const float* norm_weight,
    const float* norm_bias,
    float* residual_out,
    float* norm_out,
    float* mean_out,
    float* rstd_out,
    std::int64_t rows,
    std::int64_t output_dim,
    float eps,
    void* cuda_stream);

int nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_linear_float32(
    const float* residual,
    const std::uint16_t* linear_out_bf16_bits,
    const float* linear_bias,
    const float* residual_scale,
    const float* norm_weight,
    const float* norm_bias,
    float* residual_out,
    float* norm_out,
    float* mean_out,
    float* rstd_out,
    std::int64_t rows,
    std::int64_t output_dim,
    float eps,
    void* cuda_stream);

int nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_residual_float32(
    const float* residual,
    const float* linear_out,
    const float* linear_bias,
    const float* residual_scale,
    const float* norm_weight,
    const float* norm_bias,
    float* residual_out,
    float* norm_out,
    float* mean_out,
    float* rstd_out,
    std::uint16_t* residual_bf16_out,
    std::int64_t rows,
    std::int64_t output_dim,
    float eps,
    void* cuda_stream);

int nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_float32(
    const float* residual,
    const std::uint16_t* linear_out_bf16_bits,
    const float* linear_bias,
    const float* residual_scale,
    const float* norm_weight,
    const float* norm_bias,
    float* residual_out,
    float* norm_out,
    float* mean_out,
    float* rstd_out,
    std::uint16_t* residual_bf16_out,
    std::int64_t rows,
    std::int64_t output_dim,
    float eps,
    void* cuda_stream);

int nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_residual_bf16_norm_float32(
    const float* residual,
    const float* linear_out,
    const float* linear_bias,
    const float* residual_scale,
    const float* norm_weight,
    const float* norm_bias,
    float* residual_out,
    float* norm_out,
    float* mean_out,
    float* rstd_out,
    std::uint16_t* residual_bf16_out,
    std::uint16_t* norm_bf16_out,
    std::int64_t rows,
    std::int64_t output_dim,
    float eps,
    void* cuda_stream);

int nfn_native_tile_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_bf16_norm_float32(
    const float* residual,
    const std::uint16_t* linear_out_bf16_bits,
    const float* linear_bias,
    const float* residual_scale,
    const float* norm_weight,
    const float* norm_bias,
    float* residual_out,
    float* norm_out,
    float* mean_out,
    float* rstd_out,
    std::uint16_t* residual_bf16_out,
    std::uint16_t* norm_bf16_out,
    std::int64_t rows,
    std::int64_t output_dim,
    float eps,
    void* cuda_stream);

int nfn_native_tile_gelu_backward_float32(
    const float* x,
    const float* grad_out,
    float* grad_x,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_swiglu_backward_float32(
    const float* gate,
    const float* up,
    const float* grad_out,
    float* grad_gate,
    float* grad_up,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_gelu_backward_inplace_float32(
    const float* x,
    float* grad,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_gelu_backward_inplace_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    float* grad,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_moa_backward_inplace_float32(
    const float* x,
    float* grad,
    std::int64_t n,
    int activation_kind,
    void* cuda_stream);

int nfn_native_tile_moa_backward_inplace_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    float* grad,
    std::int64_t n,
    int activation_kind,
    void* cuda_stream);

int nfn_native_tile_dropout_forward_float32(
    const float* x,
    float* out,
    std::int64_t n,
    float dropout_p,
    std::int64_t seed,
    void* cuda_stream);

int nfn_native_tile_dropout_backward_float32(
    const float* grad_out,
    float* grad_x,
    std::int64_t n,
    float dropout_p,
    std::int64_t seed,
    void* cuda_stream);

int nfn_native_tile_absolute_position_embedding_float32(
    const float* weight,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    void* cuda_stream);

int nfn_native_tile_absolute_position_embedding_backward_float32(
    const float* grad_out,
    float* grad_weight,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    void* cuda_stream);

int nfn_native_tile_absolute_position_embedding_backward_accumulate_float32(
    const float* grad_out,
    float* grad_weight,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    void* cuda_stream);

int nfn_native_tile_token_embedding_float32(
    const float* weight,
    const std::int64_t* token_ids,
    float* out,
    std::int64_t tokens,
    std::int64_t model_dim,
    void* cuda_stream);

int nfn_native_tile_token_embedding_u16_float32(
    const float* weight,
    const std::uint16_t* token_ids,
    float* out,
    std::int64_t tokens,
    std::int64_t model_dim,
    void* cuda_stream);

int nfn_native_tile_token_embedding_backward_weight_float32(
    const std::int64_t* token_ids,
    const float* grad_out,
    float* grad_weight,
    std::int64_t tokens,
    std::int64_t model_dim,
    void* cuda_stream);

int nfn_native_tile_token_embedding_backward_weight_u16_float32(
    const std::uint16_t* token_ids,
    const float* grad_out,
    float* grad_weight,
    std::int64_t tokens,
    std::int64_t model_dim,
    void* cuda_stream);

int nfn_native_tile_random_timesteps_float32(
    float* out,
    std::int64_t batch,
    std::int64_t counter,
    void* cuda_stream);

int nfn_native_tile_mask_scheduler_int64(
    const std::int64_t* tokens,
    const float* timesteps,
    std::int64_t* out,
    std::int64_t n,
    std::int64_t seq_len,
    std::int64_t mask_token_id,
    std::int64_t counter,
    void* cuda_stream);

int nfn_native_tile_rotary_embedding_float32(
    const float* x,
    const float* inv_freq,
    float* out,
    std::int64_t n,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    void* cuda_stream);

int nfn_native_tile_rotary_embedding_backward_float32(
    const float* grad_out,
    const float* inv_freq,
    float* grad_x,
    std::int64_t n,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    void* cuda_stream);

int nfn_native_tile_rms_norm_float32(
    const float* x,
    float* out,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    void* cuda_stream);

int nfn_native_tile_rms_norm_backward_input_float32(
    const float* x,
    const float* grad_out,
    float* grad_x,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    void* cuda_stream);

int nfn_native_tile_layer_norm_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    void* cuda_stream);

int nfn_native_tile_layer_norm_with_stats_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    float* mean,
    float* rstd,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    void* cuda_stream);

int nfn_native_tile_layer_norm_with_stats_bf16_out_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    float* mean,
    float* rstd,
    std::uint16_t* out_bf16_bits,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    void* cuda_stream);

int nfn_native_tile_layer_norm_apply_stats_bf16_out_float32(
    const float* x,
    const float* weight,
    const float* bias,
    const float* mean,
    const float* rstd,
    std::uint16_t* out_bf16_bits,
    std::int64_t rows,
    std::int64_t dim,
    void* cuda_stream);

int nfn_native_tile_layer_norm_backward_input_float32(
    const float* x,
    const float* grad_out,
    const float* weight,
    float* grad_x,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    void* cuda_stream);

int nfn_native_tile_layer_norm_backward_input_with_stats_float32(
    const float* x,
    const float* grad_out,
    const float* weight,
    const float* mean,
    const float* rstd,
    float* grad_x,
    std::int64_t rows,
    std::int64_t dim,
    void* cuda_stream);

int nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_float32(
    const float* x,
    const float* grad_out,
    const float* weight,
    const float* mean,
    const float* rstd,
    const float* residual_grad,
    const float* residual_scale,
    float* out,
    std::int64_t rows,
    std::int64_t dim,
    void* cuda_stream);

int nfn_native_tile_layer_norm_backward_input_residual_add_with_stats_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const float* grad_out,
    const float* weight,
    const float* mean,
    const float* rstd,
    const float* residual_grad,
    const float* residual_scale,
    float* out,
    std::int64_t rows,
    std::int64_t dim,
    void* cuda_stream);

int nfn_native_tile_layer_norm_backward_affine_residual_add_accumulate_with_stats_float32(
    const float* x,
    const float* grad_out,
    const float* weight,
    const float* mean,
    const float* rstd,
    const float* residual_grad,
    const float* residual_scale,
    float* out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    void* cuda_stream);

int nfn_native_tile_layer_norm_backward_affine_residual_add_accumulate_with_stats_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const float* grad_out,
    const float* weight,
    const float* mean,
    const float* rstd,
    const float* residual_grad,
    const float* residual_scale,
    float* out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    void* cuda_stream);

int nfn_native_tile_layer_norm_backward_affine_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    void* cuda_stream);

int nfn_native_tile_layer_norm_backward_affine_accumulate_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    void* cuda_stream);

int nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_float32(
    const float* x,
    const float* grad_out,
    const float* mean,
    const float* rstd,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    void* cuda_stream);

int nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const float* grad_out,
    const float* mean,
    const float* rstd,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    void* cuda_stream);

int nfn_native_tile_softmax_lastdim_float32(
    const float* x,
    float* out,
    std::int64_t rows,
    std::int64_t dim,
    void* cuda_stream);

int nfn_native_tile_token_cross_entropy_partials_float32(
    const float* logits,
    const std::int64_t* targets,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    void* cuda_stream);

int nfn_native_tile_token_cross_entropy_partials_bf16_bits(
    const std::uint16_t* logits_bf16_bits,
    const std::int64_t* targets,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    void* cuda_stream);

int nfn_native_tile_token_cross_entropy_partials_strided_float32(
    const float* logits,
    const std::int64_t* targets,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    void* cuda_stream);

int nfn_native_tile_token_cross_entropy_partials_strided_bf16_bits(
    const std::uint16_t* logits_bf16_bits,
    const std::int64_t* targets,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    void* cuda_stream);

int nfn_native_tile_token_cross_entropy_partials_strided_bf16_bits_u16_targets(
    const std::uint16_t* logits_bf16_bits,
    const std::uint16_t* targets,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    void* cuda_stream);

// H0302 / MK-C: cross-entropy partials plus, when z_partials != nullptr, the
// per-block sum of squared log-partitions for the z-loss term.
int nfn_native_tile_token_cross_entropy_z_partials_strided_bf16_bits_u16_targets(
    const std::uint16_t* logits_bf16_bits,
    const std::uint16_t* targets,
    float* partials,
    float* z_partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    void* cuda_stream);

int nfn_native_tile_token_cross_entropy_variant_bf16_u16(
    std::uint16_t* logits_bf16_bits,
    const std::uint16_t* targets,
    float* row_losses,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    float z_loss_coef,
    float logit_softcap,
    bool write_gradient,
    void* cuda_stream);

int nfn_native_tile_qk_rms_norm_packed_bf16_forward(
    std::uint16_t* packed_qkv_bits,
    float* rstd,
    std::int64_t rows,
    std::int64_t heads,
    std::int64_t head_dim,
    float eps,
    void* cuda_stream);

int nfn_native_tile_qk_rms_norm_packed_bf16_backward(
    const std::uint16_t* normalized_qkv_bits,
    const float* rstd,
    float* grad_qkv_float,
    std::uint16_t* grad_qkv_bf16_bits,
    std::int64_t rows,
    std::int64_t heads,
    std::int64_t head_dim,
    void* cuda_stream);

int nfn_native_tile_differential_packed_attention_forward_bf16(
    const std::uint16_t* qkv_bf16_bits,
    std::uint16_t* out_bf16_bits,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    float lambda,
    float output_scale,
    float eps,
    void* cuda_stream);

int nfn_native_tile_differential_packed_attention_backward_bf16(
    const std::uint16_t* out_bf16_bits,
    const float* grad_out,
    std::uint16_t* grad_qkv_bf16_bits,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    float lambda,
    float output_scale,
    void* cuda_stream);

int nfn_native_tile_differential_packed_attention_forward_learned_lambda_bf16(
    const std::uint16_t* qkv_bf16_bits,
    std::uint16_t* out_bf16_bits,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    const float* lambda,
    float output_scale,
    float eps,
    void* cuda_stream);

int nfn_native_tile_differential_packed_attention_backward_learned_lambda_bf16(
    const std::uint16_t* qkv_bf16_bits,
    const std::uint16_t* out_bf16_bits,
    const float* grad_out,
    std::uint16_t* grad_qkv_bf16_bits,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    const float* lambda,
    float output_scale,
    float eps,
    float* grad_lambda,
    void* cuda_stream);

// Drains every stream that has used differential packed attention, then frees
// its stream-owned scratch. Safe to call while prior launches are still queued.
int nfn_native_tile_differential_packed_attention_release_workspaces();

int nfn_native_tile_masked_token_cross_entropy_partials_float32(
    const float* logits,
    const std::int64_t* targets,
    const float* loss_mask,
    float* loss_partials,
    float* mask_partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t ignore_index,
    void* cuda_stream);

int nfn_native_tile_latent_mse_loss_float32(
    const float* pred,
    const float* target,
    float* partials,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_latent_pool_float32(
    const float* x,
    const float* mask_values,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t dim,
    void* cuda_stream);

int nfn_native_tile_latent_pool_backward_float32(
    const float* grad_pooled,
    const float* mask_values,
    float* grad_x,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t dim,
    void* cuda_stream);

int nfn_native_tile_native_family_jepa_mask_float32(
    float* mask_values,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t masked_span,
    float mask_ratio,
    int strategy,
    void* cuda_stream);

int nfn_native_tile_native_family_jepa_mask_u16_float32(
    const std::uint16_t* tokens,
    std::uint16_t* masked_tokens,
    float* mask_values,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t masked_span,
    float mask_ratio,
    int strategy,
    void* cuda_stream);

int nfn_native_tile_act_weighted_sum_float32(
    const float* states,
    const float* weights,
    float* out,
    std::int64_t batch,
    std::int64_t steps,
    std::int64_t inner,
    void* cuda_stream);

int nfn_native_tile_act_pack_step_float32(
    const float* state_step,
    const float* halt_logits_step,
    float* state_stack,
    float* halt_logits_stack,
    std::int64_t rows,
    std::int64_t steps,
    std::int64_t inner,
    std::int64_t step,
    void* cuda_stream);

int nfn_native_tile_act_prepare_weights_float32(
    const float* halt_logits_stack,
    const std::int64_t* targets,
    float* halt_targets,
    float* halt_weights,
    std::int64_t rows,
    std::int64_t steps,
    float halt_epsilon,
    void* cuda_stream);

int nfn_native_tile_act_unpack_step_grad_float32(
    const float* grad_act,
    const float* halt_weights,
    const float* grad_halt_stack,
    float* grad_state_step,
    float* grad_halt_step,
    std::int64_t rows,
    std::int64_t steps,
    std::int64_t inner,
    std::int64_t step,
    void* cuda_stream);

int nfn_native_tile_act_halting_bce_grad_float32(
    const float* logits,
    const float* targets,
    float* partials,
    float* grad_logits,
    float* probs_out,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_semantic_alignment_loss_items_float32(
    const float* logits,
    const std::int64_t* targets,
    const std::int64_t* term_counts,
    float* losses,
    float* counts,
    std::int64_t n,
    std::int64_t dims,
    std::int64_t terms,
    std::int64_t ignore_index,
    void* cuda_stream);

int nfn_native_tile_semantic_alignment_packed_loss_backward_float32(
    const float* logits,
    const std::int64_t* targets,
    const std::int64_t* term_counts,
    const std::int64_t* term_offsets,
    float* losses,
    float* counts,
    float* grad_logits,
    std::int64_t n,
    std::int64_t dims,
    std::int64_t total_terms,
    std::int64_t ignore_index,
    float grad_scale,
    void* cuda_stream);

int nfn_native_tile_route_balance_density_float32(
    const float* route_logits,
    float* density,
    std::int64_t rows,
    std::int64_t experts,
    void* cuda_stream);

int nfn_native_tile_route_selection_loss_partials_float32(
    const float* route_logits,
    const std::int64_t* sem_targets,
    float* loss_partials,
    float* count_partials,
    std::int64_t rows,
    std::int64_t seq_len,
    std::int64_t experts,
    std::int64_t num_vocab_dims,
    std::int64_t shared_experts,
    std::int64_t ignore_index,
    void* cuda_stream);

int nfn_native_tile_route_balance_loss_float32(
    const float* density,
    float* out,
    std::int64_t experts,
    void* cuda_stream);

// Computes the shipped standard-MoE graph auxiliary loss and adds its exact
// all-expert softmax-Jacobian gradient to grad_router_logits. The weighted loss
// is accumulated across layer calls. A zero coefficient is an exact no-op.
int nfn_native_tile_moe_router_aux_loss_backward_float32(
    const float* router_logits,
    float* density_workspace,
    float* weighted_loss_accumulator,
    float* grad_router_logits,
    std::int64_t rows,
    std::int64_t experts,
    float coefficient,
    void* cuda_stream);

int nfn_native_tile_softmax_distillation_partials_float32(
    const float* teacher_logits,
    const float* student_logits,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    void* cuda_stream);

int nfn_native_tile_token_cross_entropy_backward_float32(
    const float* logits,
    const std::int64_t* targets,
    float* grad_logits,
    std::int64_t rows,
    std::int64_t vocab,
    float loss_scale,
    void* cuda_stream);

int nfn_native_tile_token_cross_entropy_backward_with_workspace_float32(
    const float* logits,
    const std::int64_t* targets,
    float* row_max_workspace,
    float* row_denom_workspace,
    float* grad_logits,
    std::int64_t rows,
    std::int64_t vocab,
    float loss_scale,
    void* cuda_stream);

int nfn_native_tile_token_cross_entropy_backward_inplace_with_workspace_float32(
    float* logits,
    const std::int64_t* targets,
    float* row_max_workspace,
    float* row_denom_workspace,
    std::int64_t rows,
    std::int64_t vocab,
    float loss_scale,
    void* cuda_stream);

int nfn_native_tile_token_cross_entropy_backward_inplace_bf16_bits_with_workspace(
    std::uint16_t* logits,
    const std::int64_t* targets,
    float* row_max_workspace,
    float* row_denom_workspace,
    std::int64_t rows,
    std::int64_t vocab,
    float loss_scale,
    void* cuda_stream);

int nfn_native_tile_token_cross_entropy_backward_inplace_strided_with_workspace_float32(
    float* logits,
    const std::int64_t* targets,
    float* row_max_workspace,
    float* row_denom_workspace,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream);

int nfn_native_tile_token_cross_entropy_backward_inplace_strided_no_pad_zero_with_workspace_float32(
    float* logits,
    const std::int64_t* targets,
    float* row_max_workspace,
    float* row_denom_workspace,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream);

int nfn_native_tile_token_cross_entropy_backward_inplace_strided_bf16_bits_with_workspace(
    std::uint16_t* logits,
    const std::int64_t* targets,
    float* row_max_workspace,
    float* row_denom_workspace,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream);

int nfn_native_tile_token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_with_workspace(
    std::uint16_t* logits,
    const std::int64_t* targets,
    float* row_max_workspace,
    float* row_denom_workspace,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream);

int nfn_native_tile_token_cross_entropy_backward_inplace_strided_bf16_bits_u16_targets_with_workspace(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* row_max_workspace,
    float* row_denom_workspace,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream);

int nfn_native_tile_token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* row_max_workspace,
    float* row_denom_workspace,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream);

int nfn_native_tile_token_cross_entropy_backward_loss_inplace_strided_bf16_bits_u16_targets(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* loss_total,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream);

int nfn_native_tile_token_cross_entropy_backward_loss_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* loss_total,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream);

int nfn_native_tile_masked_token_cross_entropy_backward_float32(
    const float* logits,
    const std::int64_t* targets,
    const float* loss_mask,
    float* grad_logits,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t ignore_index,
    float loss_scale,
    void* cuda_stream);

int nfn_native_tile_masked_token_cross_entropy_backward_with_workspace_float32(
    const float* logits,
    const std::int64_t* targets,
    const float* loss_mask,
    float* row_max_workspace,
    float* row_denom_workspace,
    float* grad_logits,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t ignore_index,
    float loss_scale,
    void* cuda_stream);

// Sparse-rule execution is bounded to seq_k <= 1024. A larger key sequence
// returns cudaErrorInvalidValue before any kernel launch.
int nfn_native_tile_scaled_dot_product_attention_float32(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    std::int64_t n,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    void* cuda_stream);

// Sparse-rule execution is bounded to seq_k <= 1024. A larger key sequence
// returns cudaErrorInvalidValue before any kernel launch.
int nfn_native_tile_scaled_dot_product_attention_backward_float32(
    const float* q,
    const float* k,
    const float* v,
    const float* grad_out,
    float* grad_q,
    float* grad_k,
    float* grad_v,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    void* cuda_stream);

// Sparse-rule execution is bounded to seq_k <= 1024. A larger key sequence
// returns cudaErrorInvalidValue before any kernel launch.
int nfn_native_tile_scaled_dot_product_attention_backward_from_merged_grad_float32(
    const float* q,
    const float* k,
    const float* v,
    const float* grad_out,
    float* grad_q,
    float* grad_k,
    float* grad_v,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    void* cuda_stream);

int nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_merged_grad_float32(
    const float* q,
    const float* k,
    const float* v,
    const float* grad_out,
    float* grad_qkv,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    void* cuda_stream);

int nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_reuse_forward_from_merged_grad_float32(
    const float* grad_out,
    float* grad_qkv,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    void* cuda_stream);

int nfn_native_tile_scaled_dot_product_attention_packed_qkv_bf16_float32(
    const std::uint16_t* qkv_bf16_bits,
    std::uint16_t* out_bf16_bits,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    void* cuda_stream);

int nfn_native_tile_scaled_dot_product_attention_packed_qkv_store_lse_bf16_float32(
    const std::uint16_t* qkv_bf16_bits,
    std::uint16_t* out_bf16_bits,
    float* saved_lse,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    void* cuda_stream);

int nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_merged_grad_float32(
    const std::uint16_t* qkv_bf16_bits,
    const std::uint16_t* out_bf16_bits,
    const float* grad_out,
    float* grad_qkv,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    void* cuda_stream);

int nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_saved_lse_bf16_from_merged_grad_float32(
    const std::uint16_t* qkv_bf16_bits,
    const std::uint16_t* out_bf16_bits,
    const float* saved_lse,
    const float* grad_out,
    float* grad_qkv,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    void* cuda_stream);

int nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_merged_grad_float32(
    const std::uint16_t* qkv_bf16_bits,
    const std::uint16_t* out_bf16_bits,
    const float* grad_out,
    std::uint16_t* grad_qkv_bf16_bits,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    void* cuda_stream);

int nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_merged_grad_float32(
    const std::uint16_t* qkv_bf16_bits,
    const std::uint16_t* out_bf16_bits,
    const float* saved_lse,
    const float* grad_out,
    std::uint16_t* grad_qkv_bf16_bits,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    void* cuda_stream);

int nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_bf16_merged_grad_float32(
    const std::uint16_t* qkv_bf16_bits,
    const std::uint16_t* out_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    std::uint16_t* grad_qkv_bf16_bits,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    void* cuda_stream);

int nfn_native_tile_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_bf16_merged_grad_float32(
    const std::uint16_t* qkv_bf16_bits,
    const std::uint16_t* out_bf16_bits,
    const float* saved_lse,
    const std::uint16_t* grad_out_bf16_bits,
    std::uint16_t* grad_qkv_bf16_bits,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    void* cuda_stream);

int nfn_native_tile_scaled_dot_product_attention_store_tk_bf16_float32(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    std::uint16_t* saved_q_bf16_bits,
    std::uint16_t* saved_k_bf16_bits,
    std::uint16_t* saved_v_bf16_bits,
    std::uint16_t* saved_o_bf16_bits,
    float* saved_lse,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    void* cuda_stream);

int nfn_native_tile_attention_tk_store_forward_workspace_bf16(
    std::uint16_t* saved_q_bf16_bits,
    std::uint16_t* saved_k_bf16_bits,
    std::uint16_t* saved_v_bf16_bits,
    std::uint16_t* saved_o_bf16_bits,
    float* saved_lse,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    void* cuda_stream);

int nfn_native_tile_scaled_dot_product_attention_backward_to_qkv_from_saved_tk_bf16_from_merged_grad_float32(
    const std::uint16_t* saved_q_bf16_bits,
    const std::uint16_t* saved_k_bf16_bits,
    const std::uint16_t* saved_v_bf16_bits,
    const std::uint16_t* saved_o_bf16_bits,
    const float* saved_lse,
    const float* grad_out,
    float* grad_qkv,
    std::int64_t batch,
    std::int64_t query_heads,
    std::int64_t key_heads,
    std::int64_t seq_q,
    std::int64_t seq_k,
    std::int64_t qk_dim,
    std::int64_t value_dim,
    float scale,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride,
    void* cuda_stream);

}
