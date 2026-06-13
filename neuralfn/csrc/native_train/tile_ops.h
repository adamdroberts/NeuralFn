#pragma once

#include <cstdint>

extern "C" {

int nfn_native_tile_ops_abi_version();
const char* nfn_native_tile_ops_error_string(int code);
void nfn_native_tile_attention_forward_stats_reset();
std::int64_t nfn_native_tile_attention_forward_row_launch_count();
std::int64_t nfn_native_tile_attention_forward_tk_launch_count();
std::int64_t nfn_native_tile_attention_backward_tk_launch_count();
std::int64_t nfn_native_tile_attention_tk_workspace_allocation_count();
std::int64_t nfn_native_tile_attention_tk_workspace_element_capacity();
std::int64_t nfn_native_tile_attention_tk_workspace_row_capacity();
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
std::int64_t nfn_native_tile_trainer_linear_sgemm_count();
std::int64_t nfn_native_tile_trainer_linear_bf16_a_pack_count();
std::int64_t nfn_native_tile_trainer_linear_bf16_a_cache_hit_count();
std::int64_t nfn_native_tile_trainer_linear_bf16_cache_reset_count();
std::int64_t nfn_native_tile_trainer_linear_bf16_workspace_allocation_count();
std::int64_t nfn_native_tile_trainer_linear_bf16_workspace_a_capacity();
std::int64_t nfn_native_tile_trainer_linear_bf16_workspace_b_capacity();
std::int64_t nfn_native_tile_trainer_linear_bf16_cached_a_capacity();
std::int64_t nfn_native_tile_trainer_linear_bf16_cache_entry_count();

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

int nfn_native_tile_init_gpt2_token_weight_float32(
    float* values,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_copy_float32(
    const float* source,
    float* dest,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_uint16_to_int64(
    const std::uint16_t* source,
    std::int64_t* dest,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_float32_to_bf16_bits(
    const float* source,
    std::uint16_t* dest,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_float32_to_bf16_bits_many(
    const float* const* sources,
    const std::int64_t* elements,
    const std::int64_t* offsets,
    std::uint16_t* dest,
    std::int64_t buffer_count,
    std::int64_t max_elements,
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

int nfn_native_tile_sum_partials_float32(
    const float* values,
    float* partials,
    std::int64_t n,
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

int nfn_native_tile_linear_bias_residual_add_float32(
    const float* residual,
    const float* linear_out,
    const float* bias,
    const float* residual_scale,
    float* out,
    std::int64_t rows,
    std::int64_t output_dim,
    void* cuda_stream);

int nfn_native_tile_gelu_backward_float32(
    const float* x,
    const float* grad_out,
    float* grad_x,
    std::int64_t n,
    void* cuda_stream);

int nfn_native_tile_gelu_backward_inplace_float32(
    const float* x,
    float* grad,
    std::int64_t n,
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

int nfn_native_tile_token_embedding_backward_weight_float32(
    const std::int64_t* token_ids,
    const float* grad_out,
    float* grad_weight,
    std::int64_t tokens,
    std::int64_t model_dim,
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

int nfn_native_tile_layer_norm_backward_input_float32(
    const float* x,
    const float* grad_out,
    const float* weight,
    float* grad_x,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
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

}
