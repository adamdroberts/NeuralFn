#include "tile_ops.h"

#include <cuda_runtime_api.h>

namespace neuralfn::tile_cuda {

void reset_attention_forward_launch_stats();
std::int64_t attention_forward_row_launch_count();
std::int64_t attention_forward_tk_launch_count();
std::int64_t attention_backward_tk_launch_count();
std::int64_t attention_backward_dprep_timing_us();
std::int64_t attention_backward_dprep_timing_count();
std::int64_t attention_backward_tk_timing_us();
std::int64_t attention_backward_tk_timing_count();
std::int64_t attention_tk_workspace_allocation_count();
std::int64_t attention_tk_workspace_element_capacity();
std::int64_t attention_tk_workspace_row_capacity();
std::int64_t token_cross_entropy_workspace_allocation_count();
std::int64_t token_cross_entropy_workspace_row_capacity();
std::int64_t attention_forward_row_fallback_count();
std::int64_t attention_forward_scalar_launch_count();
int attention_forward_row_last_error();
int attention_forward_row_prelaunch_clear_error();
int attention_forward_row_prelaunch_peek_error();
std::int64_t attention_forward_row_grid_x();
std::int64_t attention_forward_row_grid_y();
std::int64_t attention_forward_row_grid_z();
std::int64_t attention_forward_row_block_x();
int attention_forward_row_attr_status();
int attention_forward_row_attr_max_threads_per_block();
int attention_forward_row_attr_num_regs();
std::int64_t attention_forward_row_attr_shared_size_bytes();
std::int64_t attention_forward_row_attr_const_size_bytes();
std::int64_t attention_forward_row_attr_local_size_bytes();
void reset_trainer_linear_launch_stats();
void reset_trainer_linear_bf16_cache();
std::int64_t trainer_linear_bf16_gemm_count();
std::int64_t trainer_linear_tk_gemm_count();
std::int64_t trainer_linear_tk_float_out_gemm_count();
std::int64_t trainer_linear_cublaslt_gemm_count();
std::int64_t trainer_linear_sgemm_count();
std::int64_t trainer_linear_bf16_a_pack_count();
std::int64_t trainer_linear_bf16_a_cache_hit_count();
std::int64_t trainer_linear_bf16_cache_reset_count();
std::int64_t trainer_linear_bf16_workspace_allocation_count();
std::int64_t trainer_linear_bf16_workspace_a_capacity();
std::int64_t trainer_linear_bf16_workspace_b_capacity();
std::int64_t trainer_linear_bf16_cached_a_capacity();
std::int64_t trainer_linear_bf16_cache_entry_count();
std::int64_t trainer_linear_shape_stats_count();
bool trainer_linear_shape_stats_entry(
    std::int64_t index,
    int* path,
    int* m,
    int* n,
    int* k,
    int* op_a,
    int* op_b,
    std::int64_t* calls,
    std::int64_t* total_us);
bool trainer_linear_shape_stats_entry_v2(
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
void launch_gradient_accumulate_float32(float* buffer, const float* grad, std::int64_t n, float scale, cudaStream_t stream);
void launch_copy_float32(const float* source, float* dest, std::int64_t n, cudaStream_t stream);
void launch_evo_mutate_candidates_float32(
    const float* base,
    float* candidates,
    std::int64_t elements,
    std::int64_t candidate_count,
    float mutation_scale,
    std::int64_t seed,
    cudaStream_t stream);
void launch_evo_select_best_loss_float32(
    const float* losses,
    std::int64_t candidate_count,
    std::int64_t* best_index,
    float* best_loss,
    cudaStream_t stream);
void launch_evo_adopt_candidate_float32(
    const float* candidates,
    const std::int64_t* best_index,
    float* target,
    std::int64_t elements,
    std::int64_t candidate_count,
    cudaStream_t stream);
void launch_uint16_to_int64(const std::uint16_t* source, std::int64_t* dest, std::int64_t n, cudaStream_t stream);
void launch_float32_to_bf16_bits(const float* source, std::uint16_t* dest, std::int64_t n, cudaStream_t stream);
void launch_bf16_bits_to_float32(const std::uint16_t* source, float* dest, std::int64_t n, cudaStream_t stream);
void launch_store_mlp_activations_bf16_float32(
    const float* ln2_out,
    const float* fc_out,
    const float* act,
    std::uint16_t* dest,
    std::int64_t activation_elements,
    std::int64_t hidden_elements,
    cudaStream_t stream);
void launch_restore_mlp_activations_bf16_float32(
    const std::uint16_t* source,
    float* ln2_out,
    float* fc_out,
    float* act,
    std::int64_t activation_elements,
    std::int64_t hidden_elements,
    cudaStream_t stream);
void launch_float32_to_bf16_bits_many(
    const float* const* sources,
    const std::int64_t* elements,
    const std::int64_t* offsets,
    std::uint16_t* dest,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    cudaStream_t stream);
void launch_fill_float32(float* values, std::int64_t n, float value, cudaStream_t stream);
void launch_fill_many_float32(
    float* const* buffers,
    const std::int64_t* elements,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    float value,
    cudaStream_t stream);
void launch_fill_many_values_float32(
    float* const* buffers,
    const std::int64_t* elements,
    const float* values,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    cudaStream_t stream);
void launch_fill_many_values_bf16_bits_float32(
    std::uint16_t* const* buffers,
    const std::int64_t* elements,
    const float* values,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    cudaStream_t stream);
void launch_init_gpt2_token_weight_float32(float* values, std::int64_t n, cudaStream_t stream);
void launch_init_gpt2_token_weight_fast_float32(float* values, std::int64_t n, cudaStream_t stream);
void launch_init_gpt2_token_weight_with_bf16_shadow_float32(
    float* values,
    std::uint16_t* shadow_bf16_bits,
    std::int64_t n,
    cudaStream_t stream);
void launch_init_gpt2_token_weight_fast_with_bf16_shadow_float32(
    float* values,
    std::uint16_t* shadow_bf16_bits,
    std::int64_t n,
    cudaStream_t stream);
void launch_sumsq_partials_float32(const float* values, float* partials, std::int64_t n, cudaStream_t stream);
void launch_sumsq_partials_many_float32(
    const float* const* buffers,
    const std::int64_t* elements,
    const std::int64_t* partial_offsets,
    float* partials,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    cudaStream_t stream);
void launch_sumsq_partials_many_bf16_bits_float32(
    const std::uint16_t* const* buffers,
    const std::int64_t* elements,
    const std::int64_t* partial_offsets,
    float* partials,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    cudaStream_t stream);
void launch_sum_partials_float32(const float* values, float* partials, std::int64_t n, cudaStream_t stream);
void launch_scale_inplace_float32(float* values, std::int64_t n, float scale, cudaStream_t stream);
void launch_global_norm_clip_scale_float32(
    const float* sumsq_partials,
    float* clip_scale,
    std::int64_t partial_count,
    float max_norm,
    float eps,
    cudaStream_t stream);
void launch_scale_inplace_by_device_float32(
    float* values,
    const float* scale,
    std::int64_t n,
    cudaStream_t stream);
void launch_scaled_residual_add_float32(
    const float* lhs,
    const float* rhs,
    const float* scale,
    float* out,
    std::int64_t n,
    cudaStream_t stream);
void launch_split_qkv_float32(
    const float* qkv,
    float* q,
    float* k,
    float* v,
    std::int64_t rows,
    std::int64_t dim,
    cudaStream_t stream);
void launch_split_qkv_to_heads_float32(
    const float* qkv,
    float* q_heads,
    float* k_heads,
    float* v_heads,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t heads,
    std::int64_t head_dim,
    cudaStream_t stream);
void launch_split_qkv_to_heads_add_bias_float32(
    const float* qkv,
    const float* bias,
    float* q_heads,
    float* k_heads,
    float* v_heads,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t heads,
    std::int64_t head_dim,
    cudaStream_t stream);
void launch_merge_qkv_float32(
    const float* q,
    const float* k,
    const float* v,
    float* qkv,
    std::int64_t rows,
    std::int64_t dim,
    cudaStream_t stream);
void launch_merge_heads_to_qkv_float32(
    const float* q_heads,
    const float* k_heads,
    const float* v_heads,
    float* qkv,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t heads,
    std::int64_t head_dim,
    cudaStream_t stream);
void launch_reshape_heads_float32(
    const float* x,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t heads,
    std::int64_t head_dim,
    cudaStream_t stream);
void launch_merge_heads_float32(
    const float* x,
    float* out,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    cudaStream_t stream);
void launch_adamw_step_float32(
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
    cudaStream_t stream);
void launch_adamw_step_with_device_scale_float32(
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
    cudaStream_t stream);
void launch_adamw_step_many_with_device_scale_float32(
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
    cudaStream_t stream);
void launch_adamw_step_many_with_device_scale_bf16_shadow_float32(
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
    cudaStream_t stream);
void launch_adamw_step_many_with_device_scale_bf16_param_float32(
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
    cudaStream_t stream);
void launch_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32(
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
    cudaStream_t stream);
void launch_linear_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    cudaStream_t stream);
void launch_linear_bf16_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    cudaStream_t stream);
void launch_linear_weight_bf16_float32(
    const float* x,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    cudaStream_t stream);
void launch_linear_bf16_output_float32(
    const float* x,
    const float* weight,
    const float* bias,
    std::uint16_t* out_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    cudaStream_t stream);
void launch_linear_weight_bf16_output_float32(
    const float* x,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    std::uint16_t* out_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    cudaStream_t stream);
void launch_linear_bf16_input_weight_bf16_output_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    std::uint16_t* out_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    cudaStream_t stream);
void launch_linear_bf16_input_float_weight_bf16_output_float32(
    const std::uint16_t* x_bf16_bits,
    const float* weight,
    const float* bias,
    std::uint16_t* out_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    cudaStream_t stream);
void launch_bf16_bits_add_bias_inplace_float32(
    std::uint16_t* values,
    const float* bias,
    std::int64_t rows,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_bf16_input_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    cudaStream_t stream);
void launch_linear_bf16_input_weight_bf16_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    cudaStream_t stream);
void launch_linear_bf16_gelu_bf16_float32(
    const float* x,
    const float* weight,
    const float* bias,
    std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* gelu_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_weight_bf16_gelu_bf16_float32(
    const float* x,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* gelu_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_bf16_input_weight_bf16_gelu_bf16_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* gelu_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_backward_input_float32(
    const float* grad_out,
    const float* weight,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_backward_input_bf16_float32(
    const float* grad_out,
    const float* weight,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_backward_input_weight_bf16_float32(
    const float* grad_out,
    const std::uint16_t* weight_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_backward_input_weight_bf16_to_bf16_bits_float32(
    const float* grad_out,
    const std::uint16_t* weight_bf16_bits,
    std::uint16_t* grad_x_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_backward_input_bf16_bits_weight_bf16_float32(
    const std::uint16_t* grad_out_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_backward_input_bf16_bits_float32(
    const std::uint16_t* grad_out_bf16_bits,
    const float* weight,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_backward_input_dgelu_bf16_bits_float32(
    const float* grad_out,
    const float* weight,
    const std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* grad_x_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_backward_input_dgelu_weight_bf16_bits_float32(
    const float* grad_out,
    const std::uint16_t* weight_bf16_bits,
    const std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* grad_x_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_backward_input_dgelu_weight_bf16_bits_only_float32(
    const float* grad_out,
    const std::uint16_t* weight_bf16_bits,
    const std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* grad_x_bf16_bits,
    float* grad_x_fallback,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_backward_weight_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_backward_weight_accumulate_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_backward_weight_accumulate_bf16_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_backward_input_dgelu_bf16_bits_weight_bf16_bits_only_float32(
    const std::uint16_t* grad_out_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    const std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* grad_x_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_backward_weight_accumulate_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const float* grad_out,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_backward_weight_bias_accumulate_bf16_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_backward_weight_bias_accumulate_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const float* grad_out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_backward_weight_bias_accumulate_bf16_bits_float32_beta(
    const std::uint16_t* x_bf16_bits,
    const float* grad_out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    cudaStream_t stream);
void launch_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32_beta(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    cudaStream_t stream);
void launch_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_to_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    std::uint16_t* grad_weight_bf16_bits,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    cudaStream_t stream);
void launch_linear_backward_weight_accumulate_float32_bf16_bits(
    const float* x,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_backward_weight_bias_accumulate_float32_bf16_bits(
    const float* x,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_backward_weight_bias_accumulate_float32_bf16_bits_beta(
    const float* x,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    cudaStream_t stream);
void launch_linear_backward_bias_float32(
    const float* grad_out,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_backward_bias_accumulate_float32(
    const float* grad_out,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_gelu_float32(
    const float* x,
    float* out,
    std::int64_t n,
    cudaStream_t stream);
void launch_gelu_add_bias_float32(
    const float* x,
    const float* bias,
    float* biased_out,
    float* gelu_out,
    std::int64_t rows,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_gelu_add_bias_bf16_act_float32(
    const float* x,
    const float* bias,
    float* biased_out,
    float* gelu_out,
    std::uint16_t* gelu_bf16_bits,
    std::int64_t rows,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_bias_residual_add_float32(
    const float* residual,
    const float* linear_out,
    const float* bias,
    const float* residual_scale,
    float* out,
    std::int64_t rows,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_bias_residual_add_bf16_linear_float32(
    const float* residual,
    const std::uint16_t* linear_out_bf16_bits,
    const float* bias,
    const float* residual_scale,
    float* out,
    std::int64_t rows,
    std::int64_t output_dim,
    cudaStream_t stream);
void launch_linear_bias_residual_layer_norm_float32(
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
    cudaStream_t stream);
void launch_linear_bias_residual_layer_norm_with_stats_float32(
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
    cudaStream_t stream);
void launch_linear_bias_residual_layer_norm_with_stats_bf16_linear_float32(
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
    cudaStream_t stream);
void launch_linear_bias_residual_layer_norm_with_stats_bf16_residual_float32(
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
    cudaStream_t stream);
void launch_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_float32(
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
    cudaStream_t stream);
void launch_linear_bias_residual_layer_norm_with_stats_bf16_residual_bf16_norm_float32(
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
    cudaStream_t stream);
void launch_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_bf16_norm_float32(
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
    cudaStream_t stream);
void launch_gelu_backward_float32(
    const float* x,
    const float* grad_out,
    float* grad_x,
    std::int64_t n,
    cudaStream_t stream);
void launch_gelu_backward_inplace_float32(
    const float* x,
    float* grad,
    std::int64_t n,
    cudaStream_t stream);
void launch_gelu_backward_inplace_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    float* grad,
    std::int64_t n,
    cudaStream_t stream);
void launch_dropout_forward_float32(
    const float* x,
    float* out,
    std::int64_t n,
    float dropout_p,
    std::int64_t seed,
    cudaStream_t stream);
void launch_dropout_backward_float32(
    const float* grad_out,
    float* grad_x,
    std::int64_t n,
    float dropout_p,
    std::int64_t seed,
    cudaStream_t stream);
void launch_absolute_position_embedding_float32(
    const float* weight,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    cudaStream_t stream);
void launch_absolute_position_embedding_backward_float32(
    const float* grad_out,
    float* grad_weight,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    cudaStream_t stream);
void launch_absolute_position_embedding_backward_accumulate_float32(
    const float* grad_out,
    float* grad_weight,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    cudaStream_t stream);
void launch_token_embedding_float32(
    const float* weight,
    const std::int64_t* token_ids,
    float* out,
    std::int64_t tokens,
    std::int64_t model_dim,
    cudaStream_t stream);
void launch_token_embedding_u16_float32(
    const float* weight,
    const std::uint16_t* token_ids,
    float* out,
    std::int64_t tokens,
    std::int64_t model_dim,
    cudaStream_t stream);
void launch_token_position_embedding_residual_float32(
    const float* token_weight,
    const std::int64_t* token_ids,
    const float* position_weight,
    const float* scale,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    cudaStream_t stream);
void launch_token_position_embedding_residual_u16_float32(
    const float* token_weight,
    const std::uint16_t* token_ids,
    const float* position_weight,
    const float* scale,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    cudaStream_t stream);
void launch_token_embedding_backward_weight_float32(
    const std::int64_t* token_ids,
    const float* grad_out,
    float* grad_weight,
    std::int64_t tokens,
    std::int64_t model_dim,
    cudaStream_t stream);
void launch_token_embedding_backward_weight_u16_float32(
    const std::uint16_t* token_ids,
    const float* grad_out,
    float* grad_weight,
    std::int64_t tokens,
    std::int64_t model_dim,
    cudaStream_t stream);
void launch_rms_norm_float32(
    const float* x,
    float* out,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    cudaStream_t stream);
void launch_rms_norm_backward_input_float32(
    const float* x,
    const float* grad_out,
    float* grad_x,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    cudaStream_t stream);
void launch_layer_norm_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    cudaStream_t stream);
void launch_layer_norm_with_stats_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    float* mean,
    float* rstd,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    cudaStream_t stream);
void launch_layer_norm_with_stats_bf16_out_float32(
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
    cudaStream_t stream);
void launch_layer_norm_apply_stats_bf16_out_float32(
    const float* x,
    const float* weight,
    const float* bias,
    const float* mean,
    const float* rstd,
    std::uint16_t* out_bf16_bits,
    std::int64_t rows,
    std::int64_t dim,
    cudaStream_t stream);
void launch_layer_norm_backward_input_float32(
    const float* x,
    const float* grad_out,
    const float* weight,
    float* grad_x,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    cudaStream_t stream);
void launch_layer_norm_backward_input_with_stats_float32(
    const float* x,
    const float* grad_out,
    const float* weight,
    const float* mean,
    const float* rstd,
    float* grad_x,
    std::int64_t rows,
    std::int64_t dim,
    cudaStream_t stream);
void launch_layer_norm_backward_input_residual_add_with_stats_float32(
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
    cudaStream_t stream);
void launch_layer_norm_backward_input_residual_add_with_stats_bf16_bits_float32(
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
    cudaStream_t stream);
bool launch_layer_norm_backward_affine_residual_add_accumulate_with_stats_float32(
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
    cudaStream_t stream);
bool launch_layer_norm_backward_affine_residual_add_accumulate_with_stats_bf16_bits_float32(
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
    cudaStream_t stream);
void launch_layer_norm_backward_affine_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    cudaStream_t stream);
void launch_layer_norm_backward_affine_accumulate_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    cudaStream_t stream);
void launch_layer_norm_backward_affine_accumulate_with_stats_float32(
    const float* x,
    const float* grad_out,
    const float* mean,
    const float* rstd,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    cudaStream_t stream);
void launch_layer_norm_backward_affine_accumulate_with_stats_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const float* grad_out,
    const float* mean,
    const float* rstd,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    cudaStream_t stream);
void launch_softmax_lastdim_float32(
    const float* x,
    float* out,
    std::int64_t rows,
    std::int64_t dim,
    cudaStream_t stream);
void launch_token_cross_entropy_partials_float32(
    const float* logits,
    const std::int64_t* targets,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    cudaStream_t stream);
void launch_token_cross_entropy_partials_bf16_bits(
    const std::uint16_t* logits_bf16_bits,
    const std::int64_t* targets,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    cudaStream_t stream);
void launch_token_cross_entropy_partials_strided_float32(
    const float* logits,
    const std::int64_t* targets,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    cudaStream_t stream);
void launch_token_cross_entropy_partials_strided_bf16_bits(
    const std::uint16_t* logits_bf16_bits,
    const std::int64_t* targets,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    cudaStream_t stream);
void launch_token_cross_entropy_partials_strided_bf16_bits_u16_targets(
    const std::uint16_t* logits_bf16_bits,
    const std::uint16_t* targets,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    cudaStream_t stream);
void launch_masked_token_cross_entropy_partials_float32(
    const float* logits,
    const std::int64_t* targets,
    const float* loss_mask,
    float* loss_partials,
    float* mask_partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t ignore_index,
    cudaStream_t stream);
void launch_token_cross_entropy_backward_float32(
    const float* logits,
    const std::int64_t* targets,
    float* grad_logits,
    std::int64_t rows,
    std::int64_t vocab,
    float loss_scale,
    cudaStream_t stream);
void launch_token_cross_entropy_backward_with_workspace_float32(
    const float* logits,
    const std::int64_t* targets,
    float* row_max,
    float* row_denom,
    float* grad_logits,
    std::int64_t rows,
    std::int64_t vocab,
    float loss_scale,
    cudaStream_t stream);
void launch_token_cross_entropy_backward_inplace_with_workspace_float32(
    float* logits,
    const std::int64_t* targets,
    float* row_max,
    float* row_denom,
    std::int64_t rows,
    std::int64_t vocab,
    float loss_scale,
    cudaStream_t stream);
void launch_token_cross_entropy_backward_inplace_bf16_bits_with_workspace(
    std::uint16_t* logits,
    const std::int64_t* targets,
    float* row_max,
    float* row_denom,
    std::int64_t rows,
    std::int64_t vocab,
    float loss_scale,
    cudaStream_t stream);
void launch_token_cross_entropy_backward_inplace_strided_with_workspace_float32(
    float* logits,
    const std::int64_t* targets,
    float* row_max,
    float* row_denom,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    cudaStream_t stream);
void launch_token_cross_entropy_backward_inplace_strided_bf16_bits_with_workspace(
    std::uint16_t* logits,
    const std::int64_t* targets,
    float* row_max,
    float* row_denom,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    cudaStream_t stream);
void launch_token_cross_entropy_backward_inplace_strided_bf16_bits_u16_targets_with_workspace(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* row_max,
    float* row_denom,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    cudaStream_t stream);
void launch_token_cross_entropy_backward_loss_inplace_strided_bf16_bits_u16_targets(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* loss_total,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    cudaStream_t stream);
void launch_masked_token_cross_entropy_backward_float32(
    const float* logits,
    const std::int64_t* targets,
    const float* loss_mask,
    float* grad_logits,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t ignore_index,
    float loss_scale,
    cudaStream_t stream);
void launch_masked_token_cross_entropy_backward_with_workspace_float32(
    const float* logits,
    const std::int64_t* targets,
    const float* loss_mask,
    float* row_max,
    float* row_denom,
    float* grad_logits,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t ignore_index,
    float loss_scale,
    cudaStream_t stream);
void launch_scaled_dot_product_attention_float32(
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
    cudaStream_t stream);
void launch_scaled_dot_product_attention_backward_float32(
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
    cudaStream_t stream);
void launch_scaled_dot_product_attention_backward_from_merged_grad_float32(
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
    cudaStream_t stream);
void launch_scaled_dot_product_attention_backward_to_qkv_from_merged_grad_float32(
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
    cudaStream_t stream);
void launch_scaled_dot_product_attention_backward_to_qkv_reuse_forward_from_merged_grad_float32(
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
    cudaStream_t stream);
int launch_scaled_dot_product_attention_packed_qkv_bf16_float32(
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
    cudaStream_t stream);
int launch_scaled_dot_product_attention_packed_qkv_store_lse_bf16_float32(
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
    cudaStream_t stream);
int launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_merged_grad_float32(
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
    cudaStream_t stream);
int launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_saved_lse_bf16_from_merged_grad_float32(
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
    cudaStream_t stream);
int launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_merged_grad_float32(
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
    cudaStream_t stream);
int launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_merged_grad_float32(
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
    cudaStream_t stream);
int launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_bf16_merged_grad_float32(
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
    cudaStream_t stream);
int launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_bf16_merged_grad_float32(
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
    cudaStream_t stream);
int launch_scaled_dot_product_attention_store_tk_bf16_float32(
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
    cudaStream_t stream);
int launch_attention_tk_store_forward_workspace_bf16(
    std::uint16_t* saved_q_bf16_bits,
    std::uint16_t* saved_k_bf16_bits,
    std::uint16_t* saved_v_bf16_bits,
    std::uint16_t* saved_o_bf16_bits,
    float* saved_lse,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    cudaStream_t stream);
int launch_scaled_dot_product_attention_backward_to_qkv_from_saved_tk_bf16_from_merged_grad_float32(
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
    cudaStream_t stream);

}  // namespace neuralfn::tile_cuda

namespace {

cudaStream_t as_stream(void* cuda_stream) {
    return reinterpret_cast<cudaStream_t>(cuda_stream);
}

int launch_status() {
    return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

extern "C" {

int nfn_native_tile_ops_abi_version() {
    return 1;
}

const char* nfn_native_tile_ops_error_string(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}

void nfn_native_tile_attention_forward_stats_reset() {
    neuralfn::tile_cuda::reset_attention_forward_launch_stats();
}

std::int64_t nfn_native_tile_attention_forward_row_launch_count() {
    return neuralfn::tile_cuda::attention_forward_row_launch_count();
}

std::int64_t nfn_native_tile_attention_forward_tk_launch_count() {
    return neuralfn::tile_cuda::attention_forward_tk_launch_count();
}

std::int64_t nfn_native_tile_attention_backward_tk_launch_count() {
    return neuralfn::tile_cuda::attention_backward_tk_launch_count();
}

std::int64_t nfn_native_tile_attention_backward_dprep_timing_us() {
    return neuralfn::tile_cuda::attention_backward_dprep_timing_us();
}

std::int64_t nfn_native_tile_attention_backward_dprep_timing_count() {
    return neuralfn::tile_cuda::attention_backward_dprep_timing_count();
}

std::int64_t nfn_native_tile_attention_backward_tk_timing_us() {
    return neuralfn::tile_cuda::attention_backward_tk_timing_us();
}

std::int64_t nfn_native_tile_attention_backward_tk_timing_count() {
    return neuralfn::tile_cuda::attention_backward_tk_timing_count();
}

std::int64_t nfn_native_tile_attention_tk_workspace_allocation_count() {
    return neuralfn::tile_cuda::attention_tk_workspace_allocation_count();
}

std::int64_t nfn_native_tile_attention_tk_workspace_element_capacity() {
    return neuralfn::tile_cuda::attention_tk_workspace_element_capacity();
}

std::int64_t nfn_native_tile_attention_tk_workspace_row_capacity() {
    return neuralfn::tile_cuda::attention_tk_workspace_row_capacity();
}

std::int64_t nfn_native_tile_token_cross_entropy_workspace_allocation_count() {
    return neuralfn::tile_cuda::token_cross_entropy_workspace_allocation_count();
}

std::int64_t nfn_native_tile_token_cross_entropy_workspace_row_capacity() {
    return neuralfn::tile_cuda::token_cross_entropy_workspace_row_capacity();
}

std::int64_t nfn_native_tile_attention_forward_row_fallback_count() {
    return neuralfn::tile_cuda::attention_forward_row_fallback_count();
}

std::int64_t nfn_native_tile_attention_forward_scalar_launch_count() {
    return neuralfn::tile_cuda::attention_forward_scalar_launch_count();
}

int nfn_native_tile_attention_forward_row_last_error() {
    return neuralfn::tile_cuda::attention_forward_row_last_error();
}

int nfn_native_tile_attention_forward_row_prelaunch_clear_error() {
    return neuralfn::tile_cuda::attention_forward_row_prelaunch_clear_error();
}

int nfn_native_tile_attention_forward_row_prelaunch_peek_error() {
    return neuralfn::tile_cuda::attention_forward_row_prelaunch_peek_error();
}

std::int64_t nfn_native_tile_attention_forward_row_grid_x() {
    return neuralfn::tile_cuda::attention_forward_row_grid_x();
}

std::int64_t nfn_native_tile_attention_forward_row_grid_y() {
    return neuralfn::tile_cuda::attention_forward_row_grid_y();
}

std::int64_t nfn_native_tile_attention_forward_row_grid_z() {
    return neuralfn::tile_cuda::attention_forward_row_grid_z();
}

std::int64_t nfn_native_tile_attention_forward_row_block_x() {
    return neuralfn::tile_cuda::attention_forward_row_block_x();
}

int nfn_native_tile_attention_forward_row_attr_status() {
    return neuralfn::tile_cuda::attention_forward_row_attr_status();
}

int nfn_native_tile_attention_forward_row_attr_max_threads_per_block() {
    return neuralfn::tile_cuda::attention_forward_row_attr_max_threads_per_block();
}

int nfn_native_tile_attention_forward_row_attr_num_regs() {
    return neuralfn::tile_cuda::attention_forward_row_attr_num_regs();
}

std::int64_t nfn_native_tile_attention_forward_row_attr_shared_size_bytes() {
    return neuralfn::tile_cuda::attention_forward_row_attr_shared_size_bytes();
}

std::int64_t nfn_native_tile_attention_forward_row_attr_const_size_bytes() {
    return neuralfn::tile_cuda::attention_forward_row_attr_const_size_bytes();
}

std::int64_t nfn_native_tile_attention_forward_row_attr_local_size_bytes() {
    return neuralfn::tile_cuda::attention_forward_row_attr_local_size_bytes();
}

void nfn_native_tile_trainer_linear_stats_reset() {
    neuralfn::tile_cuda::reset_trainer_linear_launch_stats();
}

void nfn_native_tile_trainer_linear_bf16_cache_reset() {
    neuralfn::tile_cuda::reset_trainer_linear_bf16_cache();
}

std::int64_t nfn_native_tile_trainer_linear_bf16_gemm_count() {
    return neuralfn::tile_cuda::trainer_linear_bf16_gemm_count();
}

std::int64_t nfn_native_tile_trainer_linear_tk_gemm_count() {
    return neuralfn::tile_cuda::trainer_linear_tk_gemm_count();
}

std::int64_t nfn_native_tile_trainer_linear_tk_float_out_gemm_count() {
    return neuralfn::tile_cuda::trainer_linear_tk_float_out_gemm_count();
}

std::int64_t nfn_native_tile_trainer_linear_cublaslt_gemm_count() {
    return neuralfn::tile_cuda::trainer_linear_cublaslt_gemm_count();
}

std::int64_t nfn_native_tile_trainer_linear_sgemm_count() {
    return neuralfn::tile_cuda::trainer_linear_sgemm_count();
}

std::int64_t nfn_native_tile_trainer_linear_bf16_a_pack_count() {
    return neuralfn::tile_cuda::trainer_linear_bf16_a_pack_count();
}

std::int64_t nfn_native_tile_trainer_linear_bf16_a_cache_hit_count() {
    return neuralfn::tile_cuda::trainer_linear_bf16_a_cache_hit_count();
}

std::int64_t nfn_native_tile_trainer_linear_bf16_cache_reset_count() {
    return neuralfn::tile_cuda::trainer_linear_bf16_cache_reset_count();
}

std::int64_t nfn_native_tile_trainer_linear_bf16_workspace_allocation_count() {
    return neuralfn::tile_cuda::trainer_linear_bf16_workspace_allocation_count();
}

std::int64_t nfn_native_tile_trainer_linear_bf16_workspace_a_capacity() {
    return neuralfn::tile_cuda::trainer_linear_bf16_workspace_a_capacity();
}

std::int64_t nfn_native_tile_trainer_linear_bf16_workspace_b_capacity() {
    return neuralfn::tile_cuda::trainer_linear_bf16_workspace_b_capacity();
}

std::int64_t nfn_native_tile_trainer_linear_bf16_cached_a_capacity() {
    return neuralfn::tile_cuda::trainer_linear_bf16_cached_a_capacity();
}

std::int64_t nfn_native_tile_trainer_linear_bf16_cache_entry_count() {
    return neuralfn::tile_cuda::trainer_linear_bf16_cache_entry_count();
}

std::int64_t nfn_native_tile_trainer_linear_shape_stats_count() {
    return neuralfn::tile_cuda::trainer_linear_shape_stats_count();
}

bool nfn_native_tile_trainer_linear_shape_stats_entry(
    std::int64_t index,
    int* path,
    int* m,
    int* n,
    int* k,
    int* op_a,
    int* op_b,
    std::int64_t* calls,
    std::int64_t* total_us) {
    return neuralfn::tile_cuda::trainer_linear_shape_stats_entry(
        index, path, m, n, k, op_a, op_b, calls, total_us);
}

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
    std::int64_t* cublaslt_workspace_bytes) {
    return neuralfn::tile_cuda::trainer_linear_shape_stats_entry_v2(
        index,
        path,
        m,
        n,
        k,
        op_a,
        op_b,
        calls,
        total_us,
        cublaslt_selected_heuristic,
        cublaslt_returned_heuristics,
        cublaslt_workspace_bytes);
}

int nfn_native_tile_gradient_accumulate_float32(
    float* buffer,
    const float* grad,
    std::int64_t n,
    float scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_gradient_accumulate_float32(buffer, grad, n, scale, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_fill_float32(
    float* values,
    std::int64_t n,
    float value,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_fill_float32(values, n, value, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_fill_many_float32(
    float* const* buffers,
    const std::int64_t* elements,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    float value,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_fill_many_float32(
        buffers,
        elements,
        buffer_count,
        max_elements,
        value,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_fill_many_values_float32(
    float* const* buffers,
    const std::int64_t* elements,
    const float* values,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_fill_many_values_float32(
        buffers,
        elements,
        values,
        buffer_count,
        max_elements,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_fill_many_values_bf16_bits_float32(
    std::uint16_t* const* buffers,
    const std::int64_t* elements,
    const float* values,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_fill_many_values_bf16_bits_float32(
        buffers,
        elements,
        values,
        buffer_count,
        max_elements,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_evo_mutate_candidates_float32(
    const float* base,
    float* candidates,
    std::int64_t elements,
    std::int64_t candidate_count,
    float mutation_scale,
    std::int64_t seed,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_evo_mutate_candidates_float32(
        base,
        candidates,
        elements,
        candidate_count,
        mutation_scale,
        seed,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_evo_select_best_loss_float32(
    const float* losses,
    std::int64_t candidate_count,
    std::int64_t* best_index,
    float* best_loss,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_evo_select_best_loss_float32(
        losses,
        candidate_count,
        best_index,
        best_loss,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_evo_adopt_candidate_float32(
    const float* candidates,
    const std::int64_t* best_index,
    float* target,
    std::int64_t elements,
    std::int64_t candidate_count,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_evo_adopt_candidate_float32(
        candidates,
        best_index,
        target,
        elements,
        candidate_count,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_init_gpt2_token_weight_float32(
    float* values,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_init_gpt2_token_weight_float32(values, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_init_gpt2_token_weight_fast_float32(
    float* values,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_init_gpt2_token_weight_fast_float32(values, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_init_gpt2_token_weight_with_bf16_shadow_float32(
    float* values,
    std::uint16_t* shadow_bf16_bits,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_init_gpt2_token_weight_with_bf16_shadow_float32(
        values, shadow_bf16_bits, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_init_gpt2_token_weight_fast_with_bf16_shadow_float32(
    float* values,
    std::uint16_t* shadow_bf16_bits,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_init_gpt2_token_weight_fast_with_bf16_shadow_float32(
        values, shadow_bf16_bits, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_copy_float32(
    const float* source,
    float* dest,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_copy_float32(source, dest, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_uint16_to_int64(
    const std::uint16_t* source,
    std::int64_t* dest,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_uint16_to_int64(source, dest, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_float32_to_bf16_bits(
    const float* source,
    std::uint16_t* dest,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_float32_to_bf16_bits(source, dest, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_bf16_bits_to_float32(
    const std::uint16_t* source,
    float* dest,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_bf16_bits_to_float32(source, dest, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_store_mlp_activations_bf16_float32(
    const float* ln2_out,
    const float* fc_out,
    const float* act,
    std::uint16_t* dest,
    std::int64_t activation_elements,
    std::int64_t hidden_elements,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_store_mlp_activations_bf16_float32(
        ln2_out, fc_out, act, dest, activation_elements, hidden_elements, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_restore_mlp_activations_bf16_float32(
    const std::uint16_t* source,
    float* ln2_out,
    float* fc_out,
    float* act,
    std::int64_t activation_elements,
    std::int64_t hidden_elements,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_restore_mlp_activations_bf16_float32(
        source, ln2_out, fc_out, act, activation_elements, hidden_elements, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_float32_to_bf16_bits_many(
    const float* const* sources,
    const std::int64_t* elements,
    const std::int64_t* offsets,
    std::uint16_t* dest,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_float32_to_bf16_bits_many(
        sources, elements, offsets, dest, buffer_count, max_elements, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_sumsq_partials_float32(
    const float* values,
    float* partials,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_sumsq_partials_float32(values, partials, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_sumsq_partials_many_float32(
    const float* const* buffers,
    const std::int64_t* elements,
    const std::int64_t* partial_offsets,
    float* partials,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_sumsq_partials_many_float32(
        buffers,
        elements,
        partial_offsets,
        partials,
        buffer_count,
        max_elements,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_sumsq_partials_many_bf16_bits_float32(
    const std::uint16_t* const* buffers,
    const std::int64_t* elements,
    const std::int64_t* partial_offsets,
    float* partials,
    std::int64_t buffer_count,
    std::int64_t max_elements,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_sumsq_partials_many_bf16_bits_float32(
        buffers,
        elements,
        partial_offsets,
        partials,
        buffer_count,
        max_elements,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_sum_partials_float32(
    const float* values,
    float* partials,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_sum_partials_float32(values, partials, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_scale_inplace_float32(
    float* values,
    std::int64_t n,
    float scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_scale_inplace_float32(values, n, scale, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_global_norm_clip_scale_float32(
    const float* sumsq_partials,
    float* clip_scale,
    std::int64_t partial_count,
    float max_norm,
    float eps,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_global_norm_clip_scale_float32(
        sumsq_partials, clip_scale, partial_count, max_norm, eps, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_scale_inplace_by_device_float32(
    float* values,
    const float* scale,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_scale_inplace_by_device_float32(values, scale, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_scaled_residual_add_float32(
    const float* lhs,
    const float* rhs,
    const float* scale,
    float* out,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_scaled_residual_add_float32(lhs, rhs, scale, out, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_split_qkv_float32(
    const float* qkv,
    float* q,
    float* k,
    float* v,
    std::int64_t rows,
    std::int64_t dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_split_qkv_float32(qkv, q, k, v, rows, dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_split_qkv_to_heads_float32(
    const float* qkv,
    float* q_heads,
    float* k_heads,
    float* v_heads,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t heads,
    std::int64_t head_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_split_qkv_to_heads_float32(
        qkv,
        q_heads,
        k_heads,
        v_heads,
        batch,
        seq_len,
        heads,
        head_dim,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_split_qkv_to_heads_add_bias_float32(
        qkv,
        bias,
        q_heads,
        k_heads,
        v_heads,
        batch,
        seq_len,
        heads,
        head_dim,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_merge_qkv_float32(
    const float* q,
    const float* k,
    const float* v,
    float* qkv,
    std::int64_t rows,
    std::int64_t dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_merge_qkv_float32(q, k, v, qkv, rows, dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_merge_heads_to_qkv_float32(
    const float* q_heads,
    const float* k_heads,
    const float* v_heads,
    float* qkv,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t heads,
    std::int64_t head_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_merge_heads_to_qkv_float32(
        q_heads,
        k_heads,
        v_heads,
        qkv,
        batch,
        seq_len,
        heads,
        head_dim,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_reshape_heads_float32(
    const float* x,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t heads,
    std::int64_t head_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_reshape_heads_float32(
        x, out, batch, seq_len, heads, head_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_merge_heads_float32(
    const float* x,
    float* out,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_merge_heads_float32(
        x, out, batch, heads, seq_len, head_dim, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_adamw_step_float32(
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        n,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bias_correction1,
        sqrt_bias_correction2,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_adamw_step_with_device_scale_float32(
        param,
        grad,
        grad_scale,
        exp_avg,
        exp_avg_sq,
        n,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bias_correction1,
        sqrt_bias_correction2,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_adamw_step_many_with_device_scale_float32(
        params,
        grads,
        grad_scale,
        exp_avgs,
        exp_avg_sqs,
        elements,
        weight_decays,
        buffer_count,
        max_elements,
        lr,
        beta1,
        beta2,
        eps,
        bias_correction1,
        sqrt_bias_correction2,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_adamw_step_many_with_device_scale_bf16_shadow_float32(
        params,
        grads,
        grad_scale,
        exp_avgs,
        exp_avg_sqs,
        elements,
        weight_decays,
        bf16_shadow_offsets,
        bf16_shadow_bits,
        buffer_count,
        max_elements,
        lr,
        beta1,
        beta2,
        eps,
        bias_correction1,
        sqrt_bias_correction2,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_adamw_step_many_with_device_scale_bf16_param_float32(
        params_bf16_bits,
        grads,
        grad_scale,
        exp_avgs,
        exp_avg_sqs,
        elements,
        weight_decays,
        buffer_count,
        max_elements,
        lr,
        beta1,
        beta2,
        eps,
        bias_correction1,
        sqrt_bias_correction2,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_adamw_step_many_with_device_scale_bf16_param_bf16_grad_float32(
        params_bf16_bits,
        grads_bf16_bits,
        grad_scale,
        exp_avgs,
        exp_avg_sqs,
        elements,
        weight_decays,
        buffer_count,
        max_elements,
        lr,
        beta1,
        beta2,
        eps,
        bias_correction1,
        sqrt_bias_correction2,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_float32(
        x, weight, bias, out, rows, input_dim, output_dim, has_bias, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_bf16_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_bf16_float32(
        x, weight, bias, out, rows, input_dim, output_dim, has_bias, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_weight_bf16_float32(
    const float* x,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_weight_bf16_float32(
        x, weight_bf16_bits, bias, out, rows, input_dim, output_dim, has_bias, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_bf16_output_float32(
    const float* x,
    const float* weight,
    const float* bias,
    std::uint16_t* out_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_bf16_output_float32(
        x, weight, bias, out_bf16_bits, rows, input_dim, output_dim, has_bias, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_weight_bf16_output_float32(
    const float* x,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    std::uint16_t* out_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_weight_bf16_output_float32(
        x, weight_bf16_bits, bias, out_bf16_bits, rows, input_dim, output_dim, has_bias, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_bf16_bits_add_bias_inplace_float32(
    std::uint16_t* values,
    const float* bias,
    std::int64_t rows,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_bf16_bits_add_bias_inplace_float32(
        values, bias, rows, output_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_bf16_input_weight_bf16_output_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    std::uint16_t* out_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_bf16_input_weight_bf16_output_float32(
        x_bf16_bits,
        weight_bf16_bits,
        bias,
        out_bf16_bits,
        rows,
        input_dim,
        output_dim,
        has_bias,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_bf16_input_float_weight_bf16_output_float32(
    const std::uint16_t* x_bf16_bits,
    const float* weight,
    const float* bias,
    std::uint16_t* out_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_bf16_input_float_weight_bf16_output_float32(
        x_bf16_bits,
        weight,
        bias,
        out_bf16_bits,
        rows,
        input_dim,
        output_dim,
        has_bias,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_bf16_input_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_bf16_input_bits_float32(
        x_bf16_bits, weight, bias, out, rows, input_dim, output_dim, has_bias, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_bf16_input_weight_bf16_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_bf16_input_weight_bf16_float32(
        x_bf16_bits, weight_bf16_bits, bias, out, rows, input_dim, output_dim, has_bias, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_input_float32(
    const float* grad_out,
    const float* weight,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_input_float32(
        grad_out, weight, grad_x, rows, input_dim, output_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_input_bf16_float32(
    const float* grad_out,
    const float* weight,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_input_bf16_float32(
        grad_out, weight, grad_x, rows, input_dim, output_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_input_weight_bf16_float32(
    const float* grad_out,
    const std::uint16_t* weight_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_input_weight_bf16_float32(
        grad_out, weight_bf16_bits, grad_x, rows, input_dim, output_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_input_weight_bf16_to_bf16_bits_float32(
    const float* grad_out,
    const std::uint16_t* weight_bf16_bits,
    std::uint16_t* grad_x_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_input_weight_bf16_to_bf16_bits_float32(
        grad_out, weight_bf16_bits, grad_x_bf16_bits, rows, input_dim, output_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_float32(
    const std::uint16_t* grad_out_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_input_bf16_bits_weight_bf16_float32(
        grad_out_bf16_bits, weight_bf16_bits, grad_x, rows, input_dim, output_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_input_bf16_bits_float32(
    const std::uint16_t* grad_out_bf16_bits,
    const float* weight,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_input_bf16_bits_float32(
        grad_out_bf16_bits, weight, grad_x, rows, input_dim, output_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_input_dgelu_bf16_bits_float32(
    const float* grad_out,
    const float* weight,
    const std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* grad_x_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_input_dgelu_bf16_bits_float32(
        grad_out,
        weight,
        pre_gelu_bf16_bits,
        grad_x_bf16_bits,
        grad_x,
        rows,
        input_dim,
        output_dim,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_input_dgelu_weight_bf16_bits_float32(
    const float* grad_out,
    const std::uint16_t* weight_bf16_bits,
    const std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* grad_x_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_input_dgelu_weight_bf16_bits_float32(
        grad_out,
        weight_bf16_bits,
        pre_gelu_bf16_bits,
        grad_x_bf16_bits,
        grad_x,
        rows,
        input_dim,
        output_dim,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_input_dgelu_weight_bf16_bits_only_float32(
    const float* grad_out,
    const std::uint16_t* weight_bf16_bits,
    const std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* grad_x_bf16_bits,
    float* grad_x_fallback,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_input_dgelu_weight_bf16_bits_only_float32(
        grad_out,
        weight_bf16_bits,
        pre_gelu_bf16_bits,
        grad_x_bf16_bits,
        grad_x_fallback,
        rows,
        input_dim,
        output_dim,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_input_dgelu_bf16_bits_weight_bf16_bits_only_float32(
    const std::uint16_t* grad_out_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    const std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* grad_x_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_input_dgelu_bf16_bits_weight_bf16_bits_only_float32(
        grad_out_bf16_bits,
        weight_bf16_bits,
        pre_gelu_bf16_bits,
        grad_x_bf16_bits,
        rows,
        input_dim,
        output_dim,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_weight_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_weight_float32(
        x, grad_out, grad_weight, rows, input_dim, output_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_weight_accumulate_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_weight_accumulate_float32(
        x, grad_out, grad_weight, rows, input_dim, output_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_weight_accumulate_bf16_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_weight_accumulate_bf16_float32(
        x, grad_out, grad_weight, rows, input_dim, output_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const float* grad_out,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_weight_accumulate_bf16_bits_float32(
        x_bf16_bits, grad_out, grad_weight, rows, input_dim, output_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_weight_bias_accumulate_bf16_float32(
        x, grad_out, grad_weight, grad_bias, rows, input_dim, output_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const float* grad_out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_weight_bias_accumulate_bf16_bits_float32(
        x_bf16_bits, grad_out, grad_weight, grad_bias, rows, input_dim, output_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_float32_beta(
    const std::uint16_t* x_bf16_bits,
    const float* grad_out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_weight_bias_accumulate_bf16_bits_float32_beta(
        x_bf16_bits, grad_out, grad_weight, grad_bias, rows, input_dim, output_dim, beta, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32(
        x_bf16_bits,
        grad_out_bf16_bits,
        grad_weight,
        grad_bias,
        rows,
        input_dim,
        output_dim,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32_beta(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_float32_beta(
        x_bf16_bits,
        grad_out_bf16_bits,
        grad_weight,
        grad_bias,
        rows,
        input_dim,
        output_dim,
        beta,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_to_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    std::uint16_t* grad_weight_bf16_bits,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_weight_bias_accumulate_bf16_bits_bf16_bits_to_bf16_bits_float32(
        x_bf16_bits,
        grad_out_bf16_bits,
        grad_weight_bf16_bits,
        grad_bias,
        rows,
        input_dim,
        output_dim,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32(
        x_bf16_bits,
        grad_out_bf16_bits,
        grad_weight,
        rows,
        input_dim,
        output_dim,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta(
        x_bf16_bits,
        grad_out_bf16_bits,
        grad_weight,
        rows,
        input_dim,
        output_dim,
        beta,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_weight_bias_accumulate_float32_bf16_bits(
    const float* x,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_weight_bias_accumulate_float32_bf16_bits(
        x,
        grad_out_bf16_bits,
        grad_weight,
        grad_bias,
        rows,
        input_dim,
        output_dim,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_weight_bias_accumulate_float32_bf16_bits_beta(
    const float* x,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_weight_bias_accumulate_float32_bf16_bits_beta(
        x,
        grad_out_bf16_bits,
        grad_weight,
        grad_bias,
        rows,
        input_dim,
        output_dim,
        beta,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_weight_accumulate_float32_bf16_bits(
    const float* x,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_weight_accumulate_float32_bf16_bits(
        x, grad_out_bf16_bits, grad_weight, rows, input_dim, output_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_bias_float32(
    const float* grad_out,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_bias_float32(
        grad_out, grad_bias, rows, output_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_bias_accumulate_float32(
    const float* grad_out,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_bias_accumulate_float32(
        grad_out, grad_bias, rows, output_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_gelu_float32(
    const float* x,
    float* out,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_gelu_float32(x, out, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_gelu_add_bias_float32(
    const float* x,
    const float* bias,
    float* biased_out,
    float* gelu_out,
    std::int64_t rows,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_gelu_add_bias_float32(
        x, bias, biased_out, gelu_out, rows, output_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_gelu_add_bias_bf16_act_float32(
    const float* x,
    const float* bias,
    float* biased_out,
    float* gelu_out,
    std::uint16_t* gelu_bf16_bits,
    std::int64_t rows,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_gelu_add_bias_bf16_act_float32(
        x, bias, biased_out, gelu_out, gelu_bf16_bits, rows, output_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_bf16_gelu_bf16_float32(
    const float* x,
    const float* weight,
    const float* bias,
    std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* gelu_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_bf16_gelu_bf16_float32(
        x,
        weight,
        bias,
        pre_gelu_bf16_bits,
        gelu_bf16_bits,
        rows,
        input_dim,
        output_dim,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_weight_bf16_gelu_bf16_float32(
    const float* x,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* gelu_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_weight_bf16_gelu_bf16_float32(
        x,
        weight_bf16_bits,
        bias,
        pre_gelu_bf16_bits,
        gelu_bf16_bits,
        rows,
        input_dim,
        output_dim,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_bf16_input_weight_bf16_gelu_bf16_float32(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    const float* bias,
    std::uint16_t* pre_gelu_bf16_bits,
    std::uint16_t* gelu_bf16_bits,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_bf16_input_weight_bf16_gelu_bf16_float32(
        x_bf16_bits,
        weight_bf16_bits,
        bias,
        pre_gelu_bf16_bits,
        gelu_bf16_bits,
        rows,
        input_dim,
        output_dim,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_bias_residual_add_float32(
    const float* residual,
    const float* linear_out,
    const float* bias,
    const float* residual_scale,
    float* out,
    std::int64_t rows,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_bias_residual_add_float32(
        residual, linear_out, bias, residual_scale, out, rows, output_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_bias_residual_add_bf16_linear_float32(
    const float* residual,
    const std::uint16_t* linear_out_bf16_bits,
    const float* bias,
    const float* residual_scale,
    float* out,
    std::int64_t rows,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_bias_residual_add_bf16_linear_float32(
        residual, linear_out_bf16_bits, bias, residual_scale, out, rows, output_dim, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_bias_residual_layer_norm_float32(
        residual,
        linear_out,
        linear_bias,
        residual_scale,
        norm_weight,
        norm_bias,
        residual_out,
        norm_out,
        rows,
        output_dim,
        eps,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_bias_residual_layer_norm_with_stats_float32(
        residual,
        linear_out,
        linear_bias,
        residual_scale,
        norm_weight,
        norm_bias,
        residual_out,
        norm_out,
        mean_out,
        rstd_out,
        rows,
        output_dim,
        eps,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_bias_residual_layer_norm_with_stats_bf16_linear_float32(
        residual,
        linear_out_bf16_bits,
        linear_bias,
        residual_scale,
        norm_weight,
        norm_bias,
        residual_out,
        norm_out,
        mean_out,
        rstd_out,
        rows,
        output_dim,
        eps,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_bias_residual_layer_norm_with_stats_bf16_residual_float32(
        residual,
        linear_out,
        linear_bias,
        residual_scale,
        norm_weight,
        norm_bias,
        residual_out,
        norm_out,
        mean_out,
        rstd_out,
        residual_bf16_out,
        rows,
        output_dim,
        eps,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_float32(
        residual,
        linear_out_bf16_bits,
        linear_bias,
        residual_scale,
        norm_weight,
        norm_bias,
        residual_out,
        norm_out,
        mean_out,
        rstd_out,
        residual_bf16_out,
        rows,
        output_dim,
        eps,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_bias_residual_layer_norm_with_stats_bf16_residual_bf16_norm_float32(
        residual,
        linear_out,
        linear_bias,
        residual_scale,
        norm_weight,
        norm_bias,
        residual_out,
        norm_out,
        mean_out,
        rstd_out,
        residual_bf16_out,
        norm_bf16_out,
        rows,
        output_dim,
        eps,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_bias_residual_layer_norm_with_stats_bf16_linear_bf16_residual_bf16_norm_float32(
        residual,
        linear_out_bf16_bits,
        linear_bias,
        residual_scale,
        norm_weight,
        norm_bias,
        residual_out,
        norm_out,
        mean_out,
        rstd_out,
        residual_bf16_out,
        norm_bf16_out,
        rows,
        output_dim,
        eps,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_gelu_backward_float32(
    const float* x,
    const float* grad_out,
    float* grad_x,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_gelu_backward_float32(x, grad_out, grad_x, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_gelu_backward_inplace_float32(
    const float* x,
    float* grad,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_gelu_backward_inplace_float32(x, grad, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_gelu_backward_inplace_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    float* grad,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_gelu_backward_inplace_bf16_bits_float32(
        x_bf16_bits, grad, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_dropout_forward_float32(
    const float* x,
    float* out,
    std::int64_t n,
    float dropout_p,
    std::int64_t seed,
    void* cuda_stream) {
    if (dropout_p < 0.0f || dropout_p >= 1.0f) {
        return 1;
    }
    neuralfn::tile_cuda::launch_dropout_forward_float32(
        x, out, n, dropout_p, seed, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_dropout_backward_float32(
    const float* grad_out,
    float* grad_x,
    std::int64_t n,
    float dropout_p,
    std::int64_t seed,
    void* cuda_stream) {
    if (dropout_p < 0.0f || dropout_p >= 1.0f) {
        return 1;
    }
    neuralfn::tile_cuda::launch_dropout_backward_float32(
        grad_out, grad_x, n, dropout_p, seed, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_absolute_position_embedding_float32(
    const float* weight,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_absolute_position_embedding_float32(
        weight, out, batch, seq_len, model_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_absolute_position_embedding_backward_float32(
    const float* grad_out,
    float* grad_weight,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_absolute_position_embedding_backward_float32(
        grad_out, grad_weight, batch, seq_len, model_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_absolute_position_embedding_backward_accumulate_float32(
    const float* grad_out,
    float* grad_weight,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_absolute_position_embedding_backward_accumulate_float32(
        grad_out, grad_weight, batch, seq_len, model_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_token_embedding_float32(
    const float* weight,
    const std::int64_t* token_ids,
    float* out,
    std::int64_t tokens,
    std::int64_t model_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_embedding_float32(weight, token_ids, out, tokens, model_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_token_embedding_u16_float32(
    const float* weight,
    const std::uint16_t* token_ids,
    float* out,
    std::int64_t tokens,
    std::int64_t model_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_embedding_u16_float32(
        weight, token_ids, out, tokens, model_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_token_position_embedding_residual_float32(
    const float* token_weight,
    const std::int64_t* token_ids,
    const float* position_weight,
    const float* scale,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_position_embedding_residual_float32(
        token_weight,
        token_ids,
        position_weight,
        scale,
        out,
        batch,
        seq_len,
        model_dim,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_token_position_embedding_residual_u16_float32(
    const float* token_weight,
    const std::uint16_t* token_ids,
    const float* position_weight,
    const float* scale,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_position_embedding_residual_u16_float32(
        token_weight,
        token_ids,
        position_weight,
        scale,
        out,
        batch,
        seq_len,
        model_dim,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_token_embedding_backward_weight_float32(
    const std::int64_t* token_ids,
    const float* grad_out,
    float* grad_weight,
    std::int64_t tokens,
    std::int64_t model_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_embedding_backward_weight_float32(
        token_ids, grad_out, grad_weight, tokens, model_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_token_embedding_backward_weight_u16_float32(
    const std::uint16_t* token_ids,
    const float* grad_out,
    float* grad_weight,
    std::int64_t tokens,
    std::int64_t model_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_embedding_backward_weight_u16_float32(
        token_ids, grad_out, grad_weight, tokens, model_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_rms_norm_float32(
    const float* x,
    float* out,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_rms_norm_float32(x, out, rows, dim, eps, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_rms_norm_backward_input_float32(
    const float* x,
    const float* grad_out,
    float* grad_x,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_rms_norm_backward_input_float32(
        x, grad_out, grad_x, rows, dim, eps, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_layer_norm_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_layer_norm_float32(x, weight, bias, out, rows, dim, eps, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_layer_norm_with_stats_float32(
        x, weight, bias, out, mean, rstd, rows, dim, eps, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_layer_norm_with_stats_bf16_out_float32(
        x, weight, bias, out, mean, rstd, out_bf16_bits, rows, dim, eps, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_layer_norm_apply_stats_bf16_out_float32(
    const float* x,
    const float* weight,
    const float* bias,
    const float* mean,
    const float* rstd,
    std::uint16_t* out_bf16_bits,
    std::int64_t rows,
    std::int64_t dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_layer_norm_apply_stats_bf16_out_float32(
        x, weight, bias, mean, rstd, out_bf16_bits, rows, dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_layer_norm_backward_affine_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_layer_norm_backward_affine_float32(
        x, grad_out, grad_weight, grad_bias, rows, dim, eps, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_layer_norm_backward_affine_accumulate_float32(
    const float* x,
    const float* grad_out,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_layer_norm_backward_affine_accumulate_float32(
        x, grad_out, grad_weight, grad_bias, rows, dim, eps, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_float32(
    const float* x,
    const float* grad_out,
    const float* mean,
    const float* rstd,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_layer_norm_backward_affine_accumulate_with_stats_float32(
        x, grad_out, mean, rstd, grad_weight, grad_bias, rows, dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_layer_norm_backward_affine_accumulate_with_stats_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    const float* grad_out,
    const float* mean,
    const float* rstd,
    float* grad_weight,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_layer_norm_backward_affine_accumulate_with_stats_bf16_bits_float32(
        x_bf16_bits, grad_out, mean, rstd, grad_weight, grad_bias, rows, dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_layer_norm_backward_input_float32(
    const float* x,
    const float* grad_out,
    const float* weight,
    float* grad_x,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_layer_norm_backward_input_float32(
        x, grad_out, weight, grad_x, rows, dim, eps, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_layer_norm_backward_input_with_stats_float32(
    const float* x,
    const float* grad_out,
    const float* weight,
    const float* mean,
    const float* rstd,
    float* grad_x,
    std::int64_t rows,
    std::int64_t dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_layer_norm_backward_input_with_stats_float32(
        x, grad_out, weight, mean, rstd, grad_x, rows, dim, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_layer_norm_backward_input_residual_add_with_stats_float32(
        x, grad_out, weight, mean, rstd, residual_grad, residual_scale, out, rows, dim, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_layer_norm_backward_input_residual_add_with_stats_bf16_bits_float32(
        x_bf16_bits, grad_out, weight, mean, rstd, residual_grad, residual_scale, out, rows, dim, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (!neuralfn::tile_cuda::launch_layer_norm_backward_affine_residual_add_accumulate_with_stats_float32(
            x, grad_out, weight, mean, rstd, residual_grad, residual_scale, out, grad_weight, grad_bias, rows, dim, as_stream(cuda_stream))) {
        return 2;
    }
    return launch_status();
}

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
    void* cuda_stream) {
    if (!neuralfn::tile_cuda::launch_layer_norm_backward_affine_residual_add_accumulate_with_stats_bf16_bits_float32(
            x_bf16_bits, grad_out, weight, mean, rstd, residual_grad, residual_scale, out, grad_weight, grad_bias, rows, dim, as_stream(cuda_stream))) {
        return 2;
    }
    return launch_status();
}

int nfn_native_tile_softmax_lastdim_float32(
    const float* x,
    float* out,
    std::int64_t rows,
    std::int64_t dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_softmax_lastdim_float32(x, out, rows, dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_token_cross_entropy_partials_float32(
    const float* logits,
    const std::int64_t* targets,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_cross_entropy_partials_float32(
        logits, targets, partials, rows, vocab, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_token_cross_entropy_partials_bf16_bits(
    const std::uint16_t* logits_bf16_bits,
    const std::int64_t* targets,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_cross_entropy_partials_bf16_bits(
        logits_bf16_bits, targets, partials, rows, vocab, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_token_cross_entropy_partials_strided_float32(
    const float* logits,
    const std::int64_t* targets,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_cross_entropy_partials_strided_float32(
        logits, targets, partials, rows, vocab, row_stride, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_token_cross_entropy_partials_strided_bf16_bits(
    const std::uint16_t* logits_bf16_bits,
    const std::int64_t* targets,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_cross_entropy_partials_strided_bf16_bits(
        logits_bf16_bits, targets, partials, rows, vocab, row_stride, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_token_cross_entropy_partials_strided_bf16_bits_u16_targets(
    const std::uint16_t* logits_bf16_bits,
    const std::uint16_t* targets,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_cross_entropy_partials_strided_bf16_bits_u16_targets(
        logits_bf16_bits, targets, partials, rows, vocab, row_stride, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_masked_token_cross_entropy_partials_float32(
    const float* logits,
    const std::int64_t* targets,
    const float* loss_mask,
    float* loss_partials,
    float* mask_partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t ignore_index,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_masked_token_cross_entropy_partials_float32(
        logits,
        targets,
        loss_mask,
        loss_partials,
        mask_partials,
        rows,
        vocab,
        ignore_index,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_token_cross_entropy_backward_float32(
    const float* logits,
    const std::int64_t* targets,
    float* grad_logits,
    std::int64_t rows,
    std::int64_t vocab,
    float loss_scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_cross_entropy_backward_float32(
        logits, targets, grad_logits, rows, vocab, loss_scale, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_token_cross_entropy_backward_with_workspace_float32(
    const float* logits,
    const std::int64_t* targets,
    float* row_max_workspace,
    float* row_denom_workspace,
    float* grad_logits,
    std::int64_t rows,
    std::int64_t vocab,
    float loss_scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_cross_entropy_backward_with_workspace_float32(
        logits,
        targets,
        row_max_workspace,
        row_denom_workspace,
        grad_logits,
        rows,
        vocab,
        loss_scale,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_token_cross_entropy_backward_inplace_with_workspace_float32(
    float* logits,
    const std::int64_t* targets,
    float* row_max_workspace,
    float* row_denom_workspace,
    std::int64_t rows,
    std::int64_t vocab,
    float loss_scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_cross_entropy_backward_inplace_with_workspace_float32(
        logits,
        targets,
        row_max_workspace,
        row_denom_workspace,
        rows,
        vocab,
        loss_scale,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_token_cross_entropy_backward_inplace_bf16_bits_with_workspace(
    std::uint16_t* logits,
    const std::int64_t* targets,
    float* row_max_workspace,
    float* row_denom_workspace,
    std::int64_t rows,
    std::int64_t vocab,
    float loss_scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_cross_entropy_backward_inplace_bf16_bits_with_workspace(
        logits,
        targets,
        row_max_workspace,
        row_denom_workspace,
        rows,
        vocab,
        loss_scale,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_token_cross_entropy_backward_inplace_strided_with_workspace_float32(
    float* logits,
    const std::int64_t* targets,
    float* row_max_workspace,
    float* row_denom_workspace,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_cross_entropy_backward_inplace_strided_with_workspace_float32(
        logits,
        targets,
        row_max_workspace,
        row_denom_workspace,
        rows,
        vocab,
        row_stride,
        loss_scale,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_token_cross_entropy_backward_inplace_strided_bf16_bits_with_workspace(
    std::uint16_t* logits,
    const std::int64_t* targets,
    float* row_max_workspace,
    float* row_denom_workspace,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_cross_entropy_backward_inplace_strided_bf16_bits_with_workspace(
        logits,
        targets,
        row_max_workspace,
        row_denom_workspace,
        rows,
        vocab,
        row_stride,
        loss_scale,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_token_cross_entropy_backward_inplace_strided_bf16_bits_u16_targets_with_workspace(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* row_max_workspace,
    float* row_denom_workspace,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_cross_entropy_backward_inplace_strided_bf16_bits_u16_targets_with_workspace(
        logits,
        targets,
        row_max_workspace,
        row_denom_workspace,
        rows,
        vocab,
        row_stride,
        loss_scale,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_token_cross_entropy_backward_loss_inplace_strided_bf16_bits_u16_targets(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* loss_total,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_cross_entropy_backward_loss_inplace_strided_bf16_bits_u16_targets(
        logits,
        targets,
        loss_total,
        rows,
        vocab,
        row_stride,
        loss_scale,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_masked_token_cross_entropy_backward_float32(
    const float* logits,
    const std::int64_t* targets,
    const float* loss_mask,
    float* grad_logits,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t ignore_index,
    float loss_scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_masked_token_cross_entropy_backward_float32(
        logits,
        targets,
        loss_mask,
        grad_logits,
        rows,
        vocab,
        ignore_index,
        loss_scale,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_masked_token_cross_entropy_backward_with_workspace_float32(
        logits,
        targets,
        loss_mask,
        row_max_workspace,
        row_denom_workspace,
        grad_logits,
        rows,
        vocab,
        ignore_index,
        loss_scale,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_scaled_dot_product_attention_float32(
        q,
        k,
        v,
        out,
        n,
        query_heads,
        key_heads,
        seq_q,
        seq_k,
        qk_dim,
        value_dim,
        scale,
        is_causal,
        right_align_causal,
        use_sparse_rules,
        window,
        num_sinks,
        block_size,
        compress_stride,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_scaled_dot_product_attention_backward_float32(
        q,
        k,
        v,
        grad_out,
        grad_q,
        grad_k,
        grad_v,
        batch,
        query_heads,
        key_heads,
        seq_q,
        seq_k,
        qk_dim,
        value_dim,
        scale,
        is_causal,
        right_align_causal,
        use_sparse_rules,
        window,
        num_sinks,
        block_size,
        compress_stride,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_scaled_dot_product_attention_backward_from_merged_grad_float32(
        q,
        k,
        v,
        grad_out,
        grad_q,
        grad_k,
        grad_v,
        batch,
        query_heads,
        key_heads,
        seq_q,
        seq_k,
        qk_dim,
        value_dim,
        scale,
        is_causal,
        right_align_causal,
        use_sparse_rules,
        window,
        num_sinks,
        block_size,
        compress_stride,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_scaled_dot_product_attention_backward_to_qkv_from_merged_grad_float32(
        q,
        k,
        v,
        grad_out,
        grad_qkv,
        batch,
        query_heads,
        key_heads,
        seq_q,
        seq_k,
        qk_dim,
        value_dim,
        scale,
        is_causal,
        right_align_causal,
        use_sparse_rules,
        window,
        num_sinks,
        block_size,
        compress_stride,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_scaled_dot_product_attention_backward_to_qkv_reuse_forward_from_merged_grad_float32(
        grad_out,
        grad_qkv,
        batch,
        query_heads,
        key_heads,
        seq_q,
        seq_k,
        qk_dim,
        value_dim,
        scale,
        is_causal,
        right_align_causal,
        use_sparse_rules,
        window,
        num_sinks,
        block_size,
        compress_stride,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    const int status = neuralfn::tile_cuda::launch_scaled_dot_product_attention_packed_qkv_bf16_float32(
        qkv_bf16_bits,
        out_bf16_bits,
        batch,
        query_heads,
        key_heads,
        seq_q,
        seq_k,
        qk_dim,
        value_dim,
        scale,
        is_causal,
        right_align_causal,
        use_sparse_rules,
        window,
        num_sinks,
        block_size,
        compress_stride,
        as_stream(cuda_stream));
    return status != 0 ? status : launch_status();
}

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
    void* cuda_stream) {
    const int status = neuralfn::tile_cuda::launch_scaled_dot_product_attention_packed_qkv_store_lse_bf16_float32(
        qkv_bf16_bits,
        out_bf16_bits,
        saved_lse,
        batch,
        query_heads,
        key_heads,
        seq_q,
        seq_k,
        qk_dim,
        value_dim,
        scale,
        is_causal,
        right_align_causal,
        use_sparse_rules,
        window,
        num_sinks,
        block_size,
        compress_stride,
        as_stream(cuda_stream));
    return status != 0 ? status : launch_status();
}

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
    void* cuda_stream) {
    const int status =
        neuralfn::tile_cuda::launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_merged_grad_float32(
            qkv_bf16_bits,
            out_bf16_bits,
            grad_out,
            grad_qkv,
            batch,
            query_heads,
            key_heads,
            seq_q,
            seq_k,
            qk_dim,
            value_dim,
            scale,
            is_causal,
            right_align_causal,
            use_sparse_rules,
            window,
            num_sinks,
            block_size,
            compress_stride,
            as_stream(cuda_stream));
    return status != 0 ? status : launch_status();
}

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
    void* cuda_stream) {
    const int status =
        neuralfn::tile_cuda::
            launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_from_saved_lse_bf16_from_merged_grad_float32(
                qkv_bf16_bits,
                out_bf16_bits,
                saved_lse,
                grad_out,
                grad_qkv,
                batch,
                query_heads,
                key_heads,
                seq_q,
                seq_k,
                qk_dim,
                value_dim,
                scale,
                is_causal,
                right_align_causal,
                use_sparse_rules,
                window,
                num_sinks,
                block_size,
                compress_stride,
                as_stream(cuda_stream));
    return status != 0 ? status : launch_status();
}

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
    void* cuda_stream) {
    const int status =
        neuralfn::tile_cuda::launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_merged_grad_float32(
            qkv_bf16_bits,
            out_bf16_bits,
            grad_out,
            grad_qkv_bf16_bits,
            batch,
            query_heads,
            key_heads,
            seq_q,
            seq_k,
            qk_dim,
            value_dim,
            scale,
            is_causal,
            right_align_causal,
            use_sparse_rules,
            window,
            num_sinks,
            block_size,
            compress_stride,
            as_stream(cuda_stream));
    return status != 0 ? status : launch_status();
}

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
    void* cuda_stream) {
    const int status =
        neuralfn::tile_cuda::
            launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_merged_grad_float32(
                qkv_bf16_bits,
                out_bf16_bits,
                saved_lse,
                grad_out,
                grad_qkv_bf16_bits,
                batch,
                query_heads,
                key_heads,
                seq_q,
                seq_k,
                qk_dim,
                value_dim,
                scale,
                is_causal,
                right_align_causal,
                use_sparse_rules,
                window,
                num_sinks,
                block_size,
                compress_stride,
                as_stream(cuda_stream));
    return status != 0 ? status : launch_status();
}

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
    void* cuda_stream) {
    const int status =
        neuralfn::tile_cuda::
            launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_bf16_merged_grad_float32(
                qkv_bf16_bits,
                out_bf16_bits,
                grad_out_bf16_bits,
                grad_qkv_bf16_bits,
                batch,
                query_heads,
                key_heads,
                seq_q,
                seq_k,
                qk_dim,
                value_dim,
                scale,
                is_causal,
                right_align_causal,
                use_sparse_rules,
                window,
                num_sinks,
                block_size,
                compress_stride,
                as_stream(cuda_stream));
    return status != 0 ? status : launch_status();
}

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
    void* cuda_stream) {
    const int status =
        neuralfn::tile_cuda::
            launch_scaled_dot_product_attention_packed_qkv_backward_to_qkv_bf16_bits_from_saved_lse_bf16_from_bf16_merged_grad_float32(
                qkv_bf16_bits,
                out_bf16_bits,
                saved_lse,
                grad_out_bf16_bits,
                grad_qkv_bf16_bits,
                batch,
                query_heads,
                key_heads,
                seq_q,
                seq_k,
                qk_dim,
                value_dim,
                scale,
                is_causal,
                right_align_causal,
                use_sparse_rules,
                window,
                num_sinks,
                block_size,
                compress_stride,
                as_stream(cuda_stream));
    return status != 0 ? status : launch_status();
}

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
    void* cuda_stream) {
    const int status = neuralfn::tile_cuda::launch_scaled_dot_product_attention_store_tk_bf16_float32(
        q,
        k,
        v,
        out,
        saved_q_bf16_bits,
        saved_k_bf16_bits,
        saved_v_bf16_bits,
        saved_o_bf16_bits,
        saved_lse,
        batch,
        query_heads,
        key_heads,
        seq_q,
        seq_k,
        qk_dim,
        value_dim,
        scale,
        is_causal,
        right_align_causal,
        use_sparse_rules,
        window,
        num_sinks,
        block_size,
        compress_stride,
        as_stream(cuda_stream));
    return status == 0 ? launch_status() : status;
}

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
    void* cuda_stream) {
    const int status = neuralfn::tile_cuda::launch_attention_tk_store_forward_workspace_bf16(
        saved_q_bf16_bits,
        saved_k_bf16_bits,
        saved_v_bf16_bits,
        saved_o_bf16_bits,
        saved_lse,
        batch,
        heads,
        seq_len,
        head_dim,
        as_stream(cuda_stream));
    return status == 0 ? launch_status() : status;
}

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
    void* cuda_stream) {
    const int status =
        neuralfn::tile_cuda::
            launch_scaled_dot_product_attention_backward_to_qkv_from_saved_tk_bf16_from_merged_grad_float32(
                saved_q_bf16_bits,
                saved_k_bf16_bits,
                saved_v_bf16_bits,
                saved_o_bf16_bits,
                saved_lse,
                grad_out,
                grad_qkv,
                batch,
                query_heads,
                key_heads,
                seq_q,
                seq_k,
                qk_dim,
                value_dim,
                scale,
                is_causal,
                right_align_causal,
                use_sparse_rules,
                window,
                num_sinks,
                block_size,
                compress_stride,
                as_stream(cuda_stream));
    return status == 0 ? launch_status() : status;
}

}
