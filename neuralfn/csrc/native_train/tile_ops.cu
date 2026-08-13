#include "tile_ops.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string_view>
#include <vector>

#include <cuda_runtime_api.h>

namespace neuralfn::tile_cuda {

void reset_attention_forward_launch_stats();
std::int64_t attention_forward_row_launch_count();
std::int64_t attention_forward_tk_launch_count();
std::int64_t attention_backward_tk_launch_count();
std::int64_t attention_backward_tk_batch_cap();
std::int64_t attention_backward_tk_chunk_batch_total();
std::int64_t attention_backward_tk_chunk_batch_max();
std::int64_t attention_backward_tk_chunk_batch_min();
std::int64_t attention_backward_tk_chunk_batch_last();
int attention_backward_tk_block_size();
std::int64_t tk_packed_attention_dprep_default_warps_per_block();
std::int64_t tk_sm120_memory_block_size();
std::int64_t tk_sm120_layernorm_bwd_blocks_per_sm();
std::int64_t attention_backward_float_hd64_dprep_launch_count();
std::int64_t attention_backward_dprep_timing_us();
std::int64_t attention_backward_dprep_timing_count();
std::int64_t attention_backward_tk_timing_us();
std::int64_t attention_backward_tk_timing_count();
std::int64_t attention_tk_workspace_allocation_count();
std::int64_t attention_tk_workspace_element_capacity();
std::int64_t attention_tk_workspace_row_capacity();
std::int64_t token_cross_entropy_workspace_allocation_count();
std::int64_t token_cross_entropy_workspace_row_capacity();
std::int64_t token_cross_entropy_bf16_threads_per_row();
std::int64_t lm_head_true_fused_mat_tile();
std::int64_t lm_head_true_fused_required_threads();
std::int64_t lm_head_prob_only_target_correction_threads();
void reset_lm_head_classifier_chunk_stats();
std::int64_t lm_head_classifier_chunk_launch_count();
std::int64_t lm_head_classifier_last_rows();
std::int64_t lm_head_classifier_last_vocab();
std::int64_t lm_head_classifier_last_row_stride();
std::int64_t lm_head_classifier_loss_bin_launch_count();
std::int64_t lm_head_classifier_true_fused_launch_count();
std::int64_t lm_head_true_fused_ce_cycles();
std::int64_t lm_head_true_fused_dhidden_cycles();
std::int64_t lm_head_true_fused_dweight_cycles();
std::int64_t lm_head_true_fused_ce_blocks();
std::int64_t lm_head_true_fused_dhidden_blocks();
std::int64_t lm_head_true_fused_dweight_blocks();
void launch_lm_head_classifier_backward_prob_only_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
    std::uint16_t*, const std::uint16_t*, std::int64_t, std::int64_t, std::int64_t, float, cudaStream_t);
void launch_lm_head_classifier_backward_prob_only_ce_target_correction_bf16_bits(
    std::uint16_t*,
    const std::uint16_t*,
    const std::uint16_t*,
    const std::uint16_t*,
    float*,
    float*,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    float,
    cudaStream_t);
cudaError_t launch_lm_head_classifier_backward_true_fused_cooperative_bf16_bits_u16(
    std::uint16_t*,
    const std::uint16_t*,
    float*,
    const std::uint16_t*,
    const std::uint16_t*,
    float*,
    float*,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    float,
    float,
    bool,
    cudaStream_t);
void launch_lm_head_prob_only_dhidden_target_correction_bf16_bits(
    const std::uint16_t*, const std::uint16_t*, float*, std::int64_t, std::int64_t, std::int64_t, float, cudaStream_t);
void launch_lm_head_prob_only_dweight_target_correction_bf16_bits(
    const std::uint16_t*, const std::uint16_t*, float*, std::int64_t, std::int64_t, std::int64_t, float, cudaStream_t);
void launch_lm_head_prob_only_combined_target_correction_bf16_bits(
    const std::uint16_t*,
    const std::uint16_t*,
    const std::uint16_t*,
    float*,
    float*,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    float,
    cudaStream_t);
void launch_unary_float32(const float*, float*, std::int64_t, int, cudaStream_t);
void launch_binary_float32(const float*, const float*, float*, std::int64_t, int, cudaStream_t);
void launch_vector_binary_float32(
    const float*, const float*, const float*, const float*, float*, std::int64_t, std::int64_t, int, cudaStream_t);
void launch_mhc_beta_gradient_float32(
    const float*, const float*, const float*, const float*, const float*, const float*, const float*,
    float*, std::int64_t, std::int64_t, float, cudaStream_t);
void launch_tanh_backward_float32(const float*, const float*, float*, std::int64_t, cudaStream_t);
void launch_random_timesteps_float32(float*, std::int64_t, std::int64_t, cudaStream_t);
void launch_mask_scheduler_int64(
    const std::int64_t*, const float*, std::int64_t*, std::int64_t, std::int64_t, std::int64_t, std::int64_t, cudaStream_t);
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
std::int64_t trainer_linear_bf16_gemm_fast16bf_request_count();
std::int64_t trainer_linear_tk_gemm_count();
std::int64_t trainer_linear_tk_float_out_gemm_count();
std::int64_t trainer_linear_tk_dweight_gemm_count();
std::int64_t trainer_linear_tk_dgelu_dinput_gemm_count();
int trainer_linear_tk_sm120_k_tile();
int trainer_linear_tk_sm120_grad_k_tile();
int trainer_linear_tk_sm120_super_m();
int trainer_linear_tk_sm120_dinput_super_m();
int trainer_linear_tk_sm120_dweight_super_m();
int trainer_linear_tk_sm120_huge_n_k_tile();
int trainer_linear_tk_sm120_fast_dgelu_enabled();
int trainer_linear_tk_sm120_approx_dgelu_tanh_enabled();
std::int64_t trainer_linear_cublaslt_gemm_count();
std::int64_t trainer_linear_cublaslt_bgrad_gemm_count();
std::int64_t trainer_linear_cublaslt_bgrad_direct_write_count();
std::int64_t trainer_linear_cublaslt_bgrad_accumulate_count();
int linear_backward_bias_threads_per_block();
std::int64_t trainer_linear_sgemm_count();
std::int64_t trainer_bf16_to_f32_vec4_count();
std::int64_t trainer_linear_bf16_a_pack_count();
std::int64_t trainer_linear_bf16_cached_a_pack_count();
std::int64_t trainer_linear_bf16_cached_b_pack_count();
std::int64_t trainer_linear_bf16_transient_a_pack_count();
std::int64_t trainer_linear_bf16_transient_b_pack_count();
std::int64_t trainer_linear_bf16_a_cache_hit_count();
std::int64_t trainer_linear_bf16_cache_reset_count();
std::int64_t trainer_linear_bf16_workspace_allocation_count();
std::int64_t trainer_linear_bf16_workspace_a_capacity();
std::int64_t trainer_linear_bf16_workspace_b_capacity();
std::int64_t trainer_linear_bf16_cached_a_capacity();
std::int64_t trainer_linear_bf16_cache_entry_count();
int trainer_linear_cublaslt_grouped_layout_probe_status();
int trainer_linear_cublaslt_grouped_matmul_probe_status();
int trainer_linear_cublas_grouped_bf16_gemm_probe_status();
bool trainer_linear_cublas_prewarm(cudaStream_t stream);
bool trainer_linear_bf16_workspace_prewarm(
    std::int64_t a_elements,
    std::int64_t b_elements,
    std::int64_t c_elements);
bool trainer_linear_cublaslt_prewarm_bf16_plan(
    int m,
    int n,
    int k,
    int op_a,
    int op_b,
    int lda,
    int ldb,
    int ldc,
    bool bgrad_epilogue);
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
std::int64_t trainer_linear_cublaslt_plan_cache_count();
bool trainer_linear_cublaslt_plan_cache_entry(
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
void launch_uint32_to_int64(const std::uint32_t* source, std::int64_t* dest, std::int64_t n, cudaStream_t stream);
void launch_uint8_to_int64(const std::uint8_t* source, std::int64_t* dest, std::int64_t n, cudaStream_t stream);
void launch_diffusion_mask_u16_int64(
    const std::uint16_t* source_tokens,
    std::uint16_t* masked_tokens,
    std::int64_t* targets,
    std::int64_t rows,
    std::int64_t seq_len,
    std::int64_t vocab,
    cudaStream_t stream);
void launch_float32_to_bf16_bits(const float* source, std::uint16_t* dest, std::int64_t n, cudaStream_t stream);
void launch_bf16_bits_to_float32(const std::uint16_t* source, float* dest, std::int64_t n, cudaStream_t stream);
void launch_float32_to_nvfp4_packed(
    const float* source,
    std::uint8_t* packed,
    std::uint8_t* block_scales_e4m3,
    float tensor_scale,
    std::int64_t n,
    cudaStream_t stream);
void launch_nvfp4_packed_to_float32(
    const std::uint8_t* packed,
    const std::uint8_t* block_scales_e4m3,
    float tensor_scale,
    float* dest,
    std::int64_t n,
    cudaStream_t stream);
void launch_linear_nvfp4_input_weight_bf16_float32(
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
    cudaStream_t stream);
void launch_linear_nvfp4_input_weight_bf16_output_float32(
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
    cudaStream_t stream);
void launch_linear_backward_weight_accumulate_nvfp4_input_float32_beta(
    const std::uint8_t* x_nvfp4_packed,
    const std::uint8_t* x_block_scales_e4m3,
    float x_tensor_scale,
    const float* grad_out,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    cudaStream_t stream);
void launch_linear_backward_weight_accumulate_nvfp4_input_bf16_grad_float32_beta(
    const std::uint8_t* x_nvfp4_packed,
    const std::uint8_t* x_block_scales_e4m3,
    float x_tensor_scale,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    float beta,
    cudaStream_t stream);
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
void launch_fill_many_values_mixed_float32_bf16_bits(
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
    cudaStream_t stream);
void launch_init_gpt2_token_weight_float32(float* values, std::int64_t n, cudaStream_t stream);
void launch_seeded_normal_float32(
    float* values,
    std::uint16_t* shadow_bf16_bits,
    std::int64_t n,
    std::uint64_t seed,
    std::uint64_t offset,
    float stddev,
    cudaStream_t stream);
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
void launch_init_gpt2_token_weight_fast_with_bf16_shadow_padded_float32(
    float* values,
    std::uint16_t* shadow_bf16_bits,
    std::int64_t public_n,
    std::int64_t total_n,
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
void launch_sum_accumulate_float32(const float* values, float* total, std::int64_t n, cudaStream_t stream);
void launch_extract_diagonal_float32(
    const float* matrix, float* diagonal, std::int64_t dim, cudaStream_t stream);
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
void launch_repeat_kv_float32(
    const float* x,
    float* out,
    std::int64_t batch,
    std::int64_t kv_heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    std::int64_t repeats,
    cudaStream_t stream);
void launch_repeat_kv_backward_float32(
    const float* grad_out,
    float* grad_x,
    std::int64_t batch,
    std::int64_t kv_heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    std::int64_t repeats,
    cudaStream_t stream);
void launch_byte_patch_embed_float32(
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
    cudaStream_t stream);
void launch_byte_patch_embed_backward_float32(
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
    cudaStream_t stream);
void launch_byte_patch_merge_float32(
    const float* x,
    float* out,
    std::int64_t batch,
    std::int64_t source_len,
    std::int64_t target_len,
    std::int64_t dim,
    cudaStream_t stream);
void launch_byte_patch_merge_backward_float32(
    const float* grad_out,
    float* grad_x,
    std::int64_t batch,
    std::int64_t source_len,
    std::int64_t target_len,
    std::int64_t dim,
    cudaStream_t stream);
void launch_causal_chunk_state_float32(
    const float* hidden,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t dim,
    std::int64_t chunk_size,
    std::int64_t chunks,
    int mode,
    cudaStream_t stream);
void launch_causal_chunk_state_backward_float32(
    const float* grad_out,
    float* grad_hidden,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t dim,
    std::int64_t chunk_size,
    std::int64_t chunks,
    int mode,
    cudaStream_t stream);
void launch_broadcast_expert_routes_float32(
    const float* weights,
    const std::int64_t* indices,
    float* out_weights,
    std::int64_t* out_indices,
    std::int64_t batch,
    std::int64_t route_seq,
    std::int64_t seq_len,
    std::int64_t route_width,
    cudaStream_t stream);
void launch_broadcast_chunk_routes_float32(
    const float* weights,
    const std::int64_t* indices,
    float* out_weights,
    std::int64_t* out_indices,
    std::int64_t batch,
    std::int64_t chunks,
    std::int64_t seq_len,
    std::int64_t route_width,
    std::int64_t chunk_size,
    cudaStream_t stream);
void launch_compact_chunk_routes_float32_int64(
    const float* weights,
    const std::int64_t* indices,
    float* chunk_weights,
    std::int64_t* chunk_indices,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t chunks,
    std::int64_t route_width,
    std::int64_t chunk_size,
    cudaStream_t stream);
void launch_aggregate_chunk_route_gradients_float32(
    const float* grad_weights,
    float* aggregated_grad_weights,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t route_width,
    std::int64_t chunk_size,
    cudaStream_t stream);
void launch_semantic_route_distillation_backward_float32(
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
    cudaStream_t stream);
void launch_semantic_target_topic_distillation_backward_float32(
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
    cudaStream_t stream);
void launch_semantic_target_topic_packed_distillation_backward_float32(
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
    cudaStream_t stream);
void launch_semantic_hash_table_backward_float32(
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
    cudaStream_t stream);
void launch_semantic_route_policy_float32(
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
    cudaStream_t stream);
void launch_semantic_route_policy_packed_topic_float32(
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
    cudaStream_t stream);
void launch_semantic_route_policy_packed_topic_matrix_float32(
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
    cudaStream_t stream);
void launch_semantic_vec_from_packed_topic_float32(
    const float* topic_logits,
    const std::int64_t* term_counts,
    const std::int64_t* term_offsets,
    float* semantic_vec,
    std::int64_t rows,
    std::int64_t semantic_vocab_dims,
    std::int64_t total_terms,
    cudaStream_t stream);
void launch_semantic_packed_topic_to_padded_float32(
    const float* packed_logits,
    const std::int64_t* term_counts,
    const std::int64_t* term_offsets,
    float* padded_logits,
    std::int64_t rows,
    std::int64_t semantic_vocab_dims,
    std::int64_t total_terms,
    std::int64_t max_terms,
    cudaStream_t stream);
void launch_semantic_signature_scalar_float32(
    const float* sig_logits,
    float* signature_scalar,
    std::int64_t rows,
    std::int64_t buckets,
    cudaStream_t stream);
void launch_semantic_vec_append_signature_float32(
    const float* topic_vec,
    const float* signature_scalar,
    float* semantic_vec,
    std::int64_t rows,
    std::int64_t topic_dims,
    cudaStream_t stream);
void launch_semantic_vec_split_signature_grad_float32(
    const float* grad_semantic_vec,
    float* grad_topic_vec,
    float* grad_signature_scalar,
    std::int64_t rows,
    std::int64_t topic_dims,
    cudaStream_t stream);
void launch_semantic_signature_scalar_backward_float32(
    const float* sig_logits,
    const float* signature_scalar,
    const float* grad_signature_scalar,
    float* grad_sig_logits,
    std::int64_t rows,
    std::int64_t buckets,
    cudaStream_t stream);
void launch_semantic_free_expert_projection_float32(
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
    cudaStream_t stream);
void launch_semantic_shared_expert_projection_float32(
    const float* semantic_vec,
    const float* shared_weight,
    float* route_logits,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t semantic_vocab_dims,
    std::int64_t semantic_shared_experts,
    std::int64_t weight_stride,
    cudaStream_t stream);
void launch_semantic_free_expert_projection_backward_float32(
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
    cudaStream_t stream);
void launch_semantic_shared_expert_projection_backward_float32(
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
    cudaStream_t stream);
void launch_semantic_router_bias_add_float32(
    float* route_logits,
    const float* shared_logits,
    const float* free_bias,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t semantic_vocab_dims,
    std::int64_t semantic_shared_experts,
    std::int64_t semantic_free_experts,
    cudaStream_t stream);
void launch_semantic_router_bias_backward_float32(
    const float* grad_route_logits,
    float* grad_shared_logits,
    float* grad_free_bias,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t semantic_vocab_dims,
    std::int64_t semantic_shared_experts,
    std::int64_t semantic_free_experts,
    cudaStream_t stream);
void launch_semantic_targets_from_matrix_int64(
    const std::int64_t* semantic_matrix,
    const std::int64_t* lm_targets,
    std::int64_t* semantic_targets,
    std::uint8_t* semantic_target_valid,
    std::int64_t rows,
    std::int64_t semantic_dims,
    std::int64_t semantic_vocab_dims,
    cudaStream_t stream);
void launch_semantic_targets_from_tokens_u16_int64(
    const std::uint16_t* tokens,
    const std::int64_t* lm_targets,
    std::int64_t* semantic_targets,
    std::uint8_t* semantic_target_valid,
    std::int64_t rows,
    std::int64_t semantic_dims,
    std::int64_t semantic_terms,
    std::int64_t semantic_vocab_dims,
    cudaStream_t stream);
void launch_semantic_target_matrix_from_tokens_u16_int64(
    const std::uint16_t* tokens,
    std::int64_t* semantic_matrix,
    const std::int64_t* term_counts,
    std::int64_t rows,
    std::int64_t semantic_dims,
    std::int64_t ignore_index,
    cudaStream_t stream);
void launch_moe_swiglu_forward_float32(
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
    cudaStream_t stream);
void launch_moe_swiglu_backward_float32(
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
    cudaStream_t stream);
void launch_moe_swiglu_forward_quantized_float32(
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
    int kind,
    cudaStream_t stream);
void launch_moe_swiglu_backward_quantized_float32(
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
    int kind,
    cudaStream_t stream);
void launch_latent_mse_partials_float32(
    const float* pred,
    const float* target,
    float* partials,
    std::int64_t n,
    cudaStream_t stream);
void launch_act_weighted_sum_float32(
    const float* states,
    const float* weights,
    float* out,
    std::int64_t batch,
    std::int64_t steps,
    std::int64_t inner,
    cudaStream_t stream);
void launch_act_pack_step_float32(
    const float* state_step,
    const float* halt_logits_step,
    float* state_stack,
    float* halt_logits_stack,
    std::int64_t rows,
    std::int64_t steps,
    std::int64_t inner,
    std::int64_t step,
    cudaStream_t stream);
void launch_act_prepare_weights_float32(
    const float* halt_logits_stack,
    const std::int64_t* targets,
    float* halt_targets,
    float* halt_weights,
    std::int64_t rows,
    std::int64_t steps,
    float halt_epsilon,
    cudaStream_t stream);
void launch_act_unpack_step_grad_float32(
    const float* grad_act,
    const float* halt_weights,
    const float* grad_halt_stack,
    float* grad_state_step,
    float* grad_halt_step,
    std::int64_t rows,
    std::int64_t steps,
    std::int64_t inner,
    std::int64_t step,
    cudaStream_t stream);
void launch_act_halting_bce_grad_float32(
    const float* logits,
    const float* targets,
    float* partials,
    float* grad_logits,
    float* probs_out,
    std::int64_t n,
    cudaStream_t stream);
void launch_latent_pool_float32(
    const float* x,
    const float* mask_values,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t dim,
    cudaStream_t stream);
void launch_latent_pool_backward_float32(
    const float* grad_pooled,
    const float* mask_values,
    float* grad_x,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t dim,
    cudaStream_t stream);
void launch_native_family_jepa_mask_float32(
    float* mask_values,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t masked_span,
    float mask_ratio,
    int strategy,
    cudaStream_t stream);
void launch_native_family_jepa_mask_u16_float32(
    const std::uint16_t* tokens,
    std::uint16_t* masked_tokens,
    float* mask_values,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t masked_span,
    float mask_ratio,
    int strategy,
    cudaStream_t stream);
void launch_semantic_alignment_loss_items_float32(
    const float* logits,
    const std::int64_t* targets,
    const std::int64_t* term_counts,
    float* losses,
    float* counts,
    std::int64_t n,
    std::int64_t dims,
    std::int64_t terms,
    std::int64_t ignore_index,
    cudaStream_t stream);
void launch_semantic_alignment_packed_loss_backward_float32(
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
    cudaStream_t stream);
void launch_semantic_hash_int64(
    const float* sem_vec,
    const float* proj,
    std::int64_t* out,
    std::int64_t batch,
    std::int64_t dim,
    std::int64_t tables,
    std::int64_t planes,
    cudaStream_t stream);
void launch_topk_route_float32(
    const float* logits,
    float* weights,
    std::int64_t* indices,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t top_k,
    cudaStream_t stream);
void launch_topk_route_sqrt_softplus_float32(
    const float* logits,
    float* weights,
    std::int64_t* indices,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t top_k,
    cudaStream_t stream);
void launch_topk_route_backward_float32(
    const float* weights,
    const std::int64_t* indices,
    const float* grad_weights,
    float* grad_logits,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t top_k,
    float route_scale,
    cudaStream_t stream);
void launch_semantic_shared_topk_route_float32(
    const float* logits,
    float* weights,
    std::int64_t* indices,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t shared_experts,
    std::int64_t top_k,
    cudaStream_t stream);
void launch_semantic_shared_forced_topk_route_float32(
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
    cudaStream_t stream);
void launch_semantic_shared_topk_route_backward_float32(
    const float* weights,
    const std::int64_t* indices,
    const float* grad_weights,
    float* grad_logits,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t shared_experts,
    std::int64_t top_k,
    float route_scale,
    cudaStream_t stream);
void launch_topk_route_sqrt_softplus_backward_float32(
    const float* logits,
    const float* weights,
    const std::int64_t* indices,
    const float* grad_weights,
    float* grad_logits,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t top_k,
    float route_scale,
    cudaStream_t stream);
void launch_attentionless_decoder_float32(
    const std::int64_t* bucket_indices,
    const float* expert_output,
    const float* bucket_embed,
    const float* out_weight,
    float* out,
    std::int64_t batch,
    std::int64_t residual_dim,
    std::int64_t vocab_size,
    std::int64_t n_buckets,
    cudaStream_t stream);
void launch_expert_bias_add_float32(
    const float* logits,
    const float* bias,
    float* out,
    std::int64_t n,
    std::int64_t experts,
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
void launch_adamw_step_many_with_device_scale_hyper_float32(
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
void launch_adamw_step_many_with_device_scale_bf16_shadow_hyper_float32(
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
void launch_linear_quantized_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    int kind,
    cudaStream_t stream);
void launch_linear_backward_input_quantized_float32(
    const float* grad_out,
    const float* weight,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    int kind,
    cudaStream_t stream);
void launch_packed_weight_dequantize_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1& descriptor,
    float* output,
    cudaStream_t stream);
void launch_linear_packed_weight_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1& descriptor,
    const float* input,
    const float* bias,
    float* output,
    std::int64_t rows,
    bool has_bias,
    cudaStream_t stream);
void launch_linear_backward_input_packed_weight_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1& descriptor,
    const float* grad_output,
    float* grad_input,
    std::int64_t rows,
    cudaStream_t stream);
void launch_glimmer_embedding_gather_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1& descriptor,
    std::int64_t token_id,
    float* output,
    cudaStream_t stream);
void launch_glimmer_embedding_batch_i32_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1& descriptor,
    const std::int32_t* token_ids,
    float* output,
    std::int64_t rows,
    cudaStream_t stream);
void launch_glimmer_rms_norm_affine_float32_v1(
    const float* input,
    const NfnNativeTilePackedWeightDescriptorV1& weight,
    bool has_weight,
    float* output,
    std::int64_t rows,
    std::int64_t width,
    float eps,
    bool centered,
    cudaStream_t stream);
void launch_glimmer_positioned_rope_float32_v1(
    float* query,
    float* key,
    std::int64_t query_heads,
    std::int64_t kv_heads,
    std::int64_t head_dim,
    std::int64_t position,
    float theta,
    std::uint32_t layout,
    cudaStream_t stream);
void launch_glimmer_gqa_decode_float32_v1(
    const NfnNativeTileGlimmerGqaDecodeDescriptorV1& descriptor,
    cudaStream_t stream);
void launch_glimmer_cache_commit_bf16_v1(
    const NfnNativeTileGlimmerCacheCommitDescriptorV1& descriptor,
    cudaStream_t stream);
void launch_dflash_block_attention_float32_v1(
    const NfnNativeTileDFlashBlockAttentionDescriptorV1& descriptor,
    cudaStream_t stream);
void launch_glimmer_vision_prepare_float32_v1(
    const NfnNativeTileGlimmerVisionPrepareDescriptorV1& descriptor,
    cudaStream_t stream);
void launch_glimmer_vision_layer_norm_float32_v1(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::int64_t rows,
    std::int64_t width,
    float eps,
    cudaStream_t stream);
void launch_glimmer_vision_attention_float32_v1(
    const NfnNativeTileGlimmerVisionAttentionDescriptorV1& descriptor,
    cudaStream_t stream);
void launch_glimmer_vision_pixel_shuffle_float32_v1(
    const NfnNativeTileGlimmerVisionPixelShuffleDescriptorV1& descriptor,
    cudaStream_t stream);
void launch_glimmer_sigmoid_gate_float32_v1(
    const float* values,
    const float* gate,
    float* output,
    std::int64_t count,
    cudaStream_t stream);
void launch_glimmer_logit_transform_float32_v1(
    float* logits,
    std::int64_t count,
    float multiplier,
    float softcap,
    cudaStream_t stream);
void launch_glimmer_attention_forward_float32_v1(
    const NfnNativeTileGlimmerAttentionTrainingDescriptorV1& descriptor,
    cudaStream_t stream);
void launch_glimmer_attention_backward_float32_v1(
    const NfnNativeTileGlimmerAttentionTrainingDescriptorV1& descriptor,
    cudaStream_t stream);
void launch_glimmer_rms_norm_backward_float32_v1(
    const NfnNativeTileGlimmerRmsNormBackwardDescriptorV1& descriptor,
    const NfnNativeTilePackedWeightDescriptorV1& weight,
    bool has_weight,
    cudaStream_t stream);
void launch_glimmer_positioned_rope_batch_float32_v1(
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
    cudaStream_t stream);
void launch_glimmer_sigmoid_gate_backward_float32_v1(
    const float* values,
    const float* gate,
    const float* grad_output,
    float* grad_values,
    float* grad_gate,
    std::int64_t count,
    cudaStream_t stream);
void launch_glimmer_logit_transform_backward_float32_v1(
    const float* transformed,
    const float* grad_transformed,
    float* grad_raw,
    std::int64_t count,
    float multiplier,
    float softcap,
    cudaStream_t stream);
void launch_glimmer_masked_cross_entropy_i32_float32_v1(
    const NfnNativeTileGlimmerMaskedCeDescriptorV1& descriptor,
    cudaStream_t stream);
void launch_sequence_logp_i32_float32_forward_v1(
    const NfnNativeTileSequenceLogpDescriptorV1& descriptor,
    cudaStream_t stream);
void launch_sequence_logp_i32_float32_backward_v1(
    const NfnNativeTileSequenceLogpDescriptorV1& descriptor,
    cudaStream_t stream);
void launch_dpo_pairwise_loss_float32_forward_v1(
    const NfnNativeTileDpoPairwiseDescriptorV1& descriptor,
    cudaStream_t stream);
void launch_dpo_pairwise_loss_float32_backward_v1(
    const NfnNativeTileDpoPairwiseDescriptorV1& descriptor,
    cudaStream_t stream);
void launch_masked_reward_head_float32_forward_v1(
    const NfnNativeTileMaskedRewardHeadDescriptorV1& descriptor,
    cudaStream_t stream);
void launch_masked_reward_head_float32_backward_v1(
    const NfnNativeTileMaskedRewardHeadDescriptorV1& descriptor,
    cudaStream_t stream);
void launch_preference_bce_loss_float32_forward_v1(
    const NfnNativeTilePreferenceBceDescriptorV1& descriptor,
    cudaStream_t stream);
void launch_preference_bce_loss_float32_backward_v1(
    const NfnNativeTilePreferenceBceDescriptorV1& descriptor,
    cudaStream_t stream);
void launch_token_logp_entropy_i32_float32_forward_v1(
    const NfnNativeTileTokenLogpEntropyDescriptorV1& descriptor,
    cudaStream_t stream);
void launch_token_logp_entropy_i32_float32_backward_v1(
    const NfnNativeTileTokenLogpEntropyDescriptorV1& descriptor,
    cudaStream_t stream);
void launch_masked_ppo_loss_float32_forward_v1(
    const NfnNativeTileMaskedPpoLossDescriptorV1& descriptor,
    cudaStream_t stream);
void launch_masked_ppo_loss_float32_backward_v1(
    const NfnNativeTileMaskedPpoLossDescriptorV1& descriptor,
    cudaStream_t stream);
void launch_token_embedding_backward_weight_i32_float32(
    const std::int32_t* token_ids,
    const float* grad_output,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t vocab_size,
    std::int64_t embedding_dim,
    cudaStream_t stream);
void launch_glimmer_adamw_bf16_float32_v1(
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
    cudaStream_t stream);
void launch_split_last_dim_float32(
    const float* input,
    float* first,
    float* second,
    std::int64_t rows,
    std::int64_t dim,
    cudaStream_t stream);
void launch_merge_last_dim_float32(
    const float* first,
    const float* second,
    float* output,
    std::int64_t rows,
    std::int64_t half_dim,
    cudaStream_t stream);
void launch_split_at_last_dim_float32(
    const float* input,
    float* first,
    float* second,
    std::int64_t rows,
    std::int64_t first_dim,
    std::int64_t second_dim,
    cudaStream_t stream);
void launch_concat_last_dim_float32(
    const float* first,
    const float* second,
    float* output,
    std::int64_t rows,
    std::int64_t first_dim,
    std::int64_t second_dim,
    cudaStream_t stream);
void launch_differential_combine_float32(
    const float* first,
    const float* second,
    float* output,
    std::int64_t elements,
    float lambda,
    float output_scale,
    cudaStream_t stream);
void launch_differential_backward_float32(
    const float* grad_output,
    float* grad_first,
    float* grad_second,
    std::int64_t elements,
    float lambda,
    float output_scale,
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
void launch_linear_backward_input_bf16_bits_weight_bf16_strided_float32(
    const std::uint16_t* grad_out_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    std::int64_t grad_out_row_stride,
    cudaStream_t stream);
bool cublaslt_linear_backward_input_bf16_bits_weight_bf16_strided_float32(
    const std::uint16_t* grad_out_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    std::int64_t grad_out_row_stride,
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
void launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_float32_beta(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    std::int64_t grad_out_row_stride,
    float beta,
    cudaStream_t stream);
bool cublaslt_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_float32_beta(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    std::int64_t grad_out_row_stride,
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
void launch_linear_backward_bias_accumulate_bf16_bits_float32(
    const std::uint16_t* grad_out_bf16_bits,
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
void launch_moa_add_bias_float32(
    const float* x,
    const float* bias,
    float* biased_out,
    float* activation_out,
    std::int64_t rows,
    std::int64_t output_dim,
    int activation_kind,
    cudaStream_t stream);
void launch_moa_add_bias_bf16_act_float32(
    const float* x,
    const float* bias,
    float* biased_out,
    float* activation_out,
    std::uint16_t* activation_bf16_bits,
    std::int64_t rows,
    std::int64_t output_dim,
    int activation_kind,
    cudaStream_t stream);
void launch_swiglu_float32(
    const float* gate,
    const float* up,
    float* out,
    std::int64_t n,
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
void launch_linear_bias_residual_add_bf16_linear_bf16_residual_float32(
    const float* residual,
    const std::uint16_t* linear_out_bf16_bits,
    const float* bias,
    const float* residual_scale,
    float* out,
    std::uint16_t* residual_bf16_out,
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
void launch_swiglu_backward_float32(
    const float* gate,
    const float* up,
    const float* grad_out,
    float* grad_gate,
    float* grad_up,
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
void launch_moa_backward_inplace_float32(
    const float* x,
    float* grad,
    std::int64_t n,
    int activation_kind,
    cudaStream_t stream);
void launch_moa_backward_inplace_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    float* grad,
    std::int64_t n,
    int activation_kind,
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
void launch_token_position_embedding_residual_u16_bf16_weight_float32(
    const std::uint16_t* token_weight_bf16,
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
void launch_rotary_embedding_float32(
    const float* x,
    const float* inv_freq,
    float* out,
    std::int64_t n,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    cudaStream_t stream);
void launch_rotary_embedding_backward_float32(
    const float* grad_out,
    const float* inv_freq,
    float* grad_x,
    std::int64_t n,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
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
void launch_token_cross_entropy_z_partials_strided_bf16_bits_u16_targets(
    const std::uint16_t* logits_bf16_bits,
    const std::uint16_t* targets,
    float* partials,
    float* z_partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    cudaStream_t stream);
void launch_token_cross_entropy_variant_bf16_u16(
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
    cudaStream_t stream);
void launch_qk_rms_norm_packed_bf16_forward(
    std::uint16_t*, float*, std::int64_t, std::int64_t, std::int64_t, float, cudaStream_t);
void launch_qk_rms_norm_packed_bf16_backward(
    const std::uint16_t*, const float*, float*, std::uint16_t*,
    std::int64_t, std::int64_t, std::int64_t, cudaStream_t);
int launch_differential_packed_attention_forward_bf16(
    const std::uint16_t*, std::uint16_t*, std::int64_t, std::int64_t, std::int64_t,
    std::int64_t, float, float, float, cudaStream_t);
int launch_differential_packed_attention_backward_bf16(
    const std::uint16_t*, const float*, std::uint16_t*,
    std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    float, float, cudaStream_t);
int launch_differential_packed_attention_forward_learned_lambda_bf16(
    const std::uint16_t*, std::uint16_t*, std::int64_t, std::int64_t, std::int64_t,
    std::int64_t, const float*, float, float, cudaStream_t);
int launch_differential_packed_attention_backward_learned_lambda_bf16(
    const std::uint16_t*, const std::uint16_t*, const float*, std::uint16_t*,
    std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    const float*, float, float, float*, cudaStream_t);
int release_differential_packed_attention_workspaces();
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
void launch_route_selection_loss_partials_float32(
    const float* route_logits,
    const std::int64_t* sem_targets,
    float* loss_partials,
    float* count_partials,
    std::int64_t n,
    std::int64_t seq_len,
    std::int64_t experts,
    std::int64_t num_vocab_dims,
    std::int64_t shared_experts,
    std::int64_t ignore_index,
    cudaStream_t stream);
void launch_route_balance_density_float32(
    const float* route_logits,
    float* density,
    std::int64_t rows,
    std::int64_t experts,
    cudaStream_t stream);
void launch_route_balance_loss_float32(
    const float* density,
    float* out,
    std::int64_t experts,
    cudaStream_t stream);
void launch_moe_router_aux_loss_backward_float32(
    const float* router_logits,
    float* density,
    float* weighted_loss_accumulator,
    float* grad_router_logits,
    std::int64_t rows,
    std::int64_t experts,
    float coefficient,
    cudaStream_t stream);
void launch_softmax_distillation_partials_float32(
    const float* teacher_logits,
    const float* student_logits,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
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
void launch_token_cross_entropy_backward_inplace_strided_no_pad_zero_with_workspace_float32(
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
void launch_token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_with_workspace(
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
void launch_token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* row_max,
    float* row_denom,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    cudaStream_t stream);
void launch_lm_head_classifier_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace(
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
void launch_token_cross_entropy_backward_loss_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* loss_total,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    cudaStream_t stream);
void launch_lm_head_classifier_backward_loss_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* loss_total,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    cudaStream_t stream);
void launch_lm_head_classifier_backward_row_losses_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* row_losses,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    cudaStream_t stream);
void launch_lm_head_classifier_backward_loss_bins_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* loss_bins,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    std::int64_t loss_bin_count,
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
void reset_turboquant_attention_launch_stats();
std::int64_t turboquant_attention_launch_count();
int launch_turboquant_attention_forward_v1(
    const NfnNativeTileTurboQuantAttentionDescriptorV1& descriptor,
    cudaStream_t stream);

}  // namespace neuralfn::tile_cuda

namespace {

constexpr int kLmHeadCooperativeFlagLossBins = 1 << 0;
constexpr int kLmHeadCooperativeFlagNoLoss = 1 << 1;
constexpr int kLmHeadCooperativeLossBinCountShift = 8;
constexpr int kLmHeadGraphThreadCacheCapacity = 8;
constexpr std::int64_t kRawSparseAttentionMaxKeySequenceLength = 1024;
constexpr std::int64_t kTurboQuantAttentionMaxSequenceLength = 16384;
constexpr std::int64_t kTurboQuantAttentionMaxHeadDimension = 256;

int validate_raw_sparse_attention_key_sequence_length(
    bool /*use_sparse_rules*/,
    std::int64_t seq_k) {
    // The scalar raw kernel has a fixed 1,024-key tile even for dense calls.
    // Reject every larger call until it is replaced; otherwise dense callers
    // can receive plausible output that silently omitted later keys.
    return seq_k > kRawSparseAttentionMaxKeySequenceLength
        ? static_cast<int>(cudaErrorInvalidValue)
        : static_cast<int>(cudaSuccess);
}

bool checked_positive_product(
    std::int64_t left,
    std::int64_t right,
    std::int64_t* output) {
    if (output == nullptr || left < 0 || right < 0 ||
        (left != 0 && right > std::numeric_limits<std::int64_t>::max() / left)) {
        return false;
    }
    *output = left * right;
    return true;
}

bool normalize_packed_weight_descriptor(
    const NfnNativeTilePackedWeightDescriptorV1* source,
    NfnNativeTilePackedWeightDescriptorV1* output) {
    if (source == nullptr || output == nullptr ||
        source->struct_size < sizeof(NfnNativeTilePackedWeightDescriptorV1) ||
        source->version != NFN_NATIVE_TILE_PACKED_WEIGHT_V1 ||
        source->flags != 0 || source->reserved0 != 0 || source->reserved1 != 0 ||
        source->data == nullptr || source->data_nbytes <= 0 ||
        source->output_dim <= 0 || source->input_dim <= 0 ||
        source->row_stride_bytes <= 0) {
        return false;
    }

    std::int64_t block_elements = 0;
    std::int64_t block_bytes = 0;
    switch (source->encoding) {
        case NFN_NATIVE_TILE_PACKED_WEIGHT_F32:
            block_elements = 1;
            block_bytes = 4;
            break;
        case NFN_NATIVE_TILE_PACKED_WEIGHT_Q4_K:
            block_elements = 256;
            block_bytes = 144;
            break;
        case NFN_NATIVE_TILE_PACKED_WEIGHT_Q5_K:
            block_elements = 256;
            block_bytes = 176;
            break;
        case NFN_NATIVE_TILE_PACKED_WEIGHT_Q6_K:
            block_elements = 256;
            block_bytes = 210;
            break;
        case NFN_NATIVE_TILE_PACKED_WEIGHT_BF16:
            block_elements = 1;
            block_bytes = 2;
            break;
        case NFN_NATIVE_TILE_PACKED_WEIGHT_NF4_GROUP64:
            block_elements = 64;
            block_bytes = 36;
            break;
        default:
            return false;
    }
    if (source->encoding != NFN_NATIVE_TILE_PACKED_WEIGHT_NF4_GROUP64 &&
        source->input_dim % block_elements != 0) {
        return false;
    }
    std::int64_t blocks_per_row =
        source->encoding == NFN_NATIVE_TILE_PACKED_WEIGHT_NF4_GROUP64
        ? (source->input_dim + block_elements - 1) / block_elements
        : source->input_dim / block_elements;
    std::int64_t expected_row_stride = 0;
    std::int64_t expected_nbytes = 0;
    if (!checked_positive_product(blocks_per_row, block_bytes, &expected_row_stride) ||
        !checked_positive_product(
            source->output_dim, expected_row_stride, &expected_nbytes) ||
        source->row_stride_bytes != expected_row_stride ||
        source->data_nbytes != expected_nbytes) {
        return false;
    }
    *output = *source;
    output->struct_size = sizeof(NfnNativeTilePackedWeightDescriptorV1);
    return true;
}

bool normalize_glimmer_attention_training_descriptor(
    const NfnNativeTileGlimmerAttentionTrainingDescriptorV1* source,
    NfnNativeTileGlimmerAttentionTrainingDescriptorV1* output,
    bool backward) {
    if (source == nullptr || output == nullptr ||
        source->struct_size < sizeof(NfnNativeTileGlimmerAttentionTrainingDescriptorV1) ||
        source->version != NFN_NATIVE_TILE_GLIMMER_TRAINING_V1 ||
        (source->flags & ~NFN_NATIVE_TILE_GLIMMER_TRAIN_CAUSAL) != 0 ||
        source->reserved0 != 0 || source->reserved1 != 0 ||
        source->reserved2 != 0 || source->reserved3 != 0 ||
        source->query == nullptr || source->key == nullptr || source->value == nullptr ||
        source->output == nullptr || source->logsumexp == nullptr ||
        source->batch_size <= 0 || source->sequence_length <= 0 ||
        source->query_heads <= 0 || source->kv_heads <= 0 ||
        source->query_heads % source->kv_heads != 0 ||
        source->head_dim <= 0 || source->head_dim > 256 ||
        source->window < 0 || source->window > source->sequence_length ||
        !std::isfinite(source->scale) || !(source->scale > 0.0f) ||
        (backward && (source->grad_output == nullptr || source->grad_query == nullptr ||
                      source->grad_key == nullptr || source->grad_value == nullptr))) {
        return false;
    }
    std::int64_t rows = 0;
    std::int64_t q_rows = 0;
    std::int64_t kv_rows = 0;
    if (!checked_positive_product(source->batch_size, source->sequence_length, &rows) ||
        !checked_positive_product(rows, source->query_heads, &q_rows) ||
        !checked_positive_product(rows, source->kv_heads, &kv_rows) ||
        !checked_positive_product(q_rows, source->head_dim, &q_rows) ||
        !checked_positive_product(kv_rows, source->head_dim, &kv_rows)) {
        return false;
    }
    *output = *source;
    output->struct_size = sizeof(NfnNativeTileGlimmerAttentionTrainingDescriptorV1);
    return true;
}

bool normalize_glimmer_masked_ce_descriptor(
    const NfnNativeTileGlimmerMaskedCeDescriptorV1* source,
    NfnNativeTileGlimmerMaskedCeDescriptorV1* output) {
    if (source == nullptr || output == nullptr ||
        source->struct_size < sizeof(NfnNativeTileGlimmerMaskedCeDescriptorV1) ||
        source->version != NFN_NATIVE_TILE_GLIMMER_TRAINING_V1 ||
        source->flags != 0 || source->reserved0 != 0 || source->reserved1 != 0 ||
        source->reserved2 != 0 || source->transformed_logits == nullptr ||
        source->targets == nullptr || source->row_loss == nullptr ||
        source->rows <= 0 || source->vocab_size <= 0 ||
        source->vocab_size > std::numeric_limits<std::int32_t>::max() ||
        !std::isfinite(source->grad_scale) || source->grad_scale < 0.0f) {
        return false;
    }
    *output = *source;
    output->struct_size = sizeof(NfnNativeTileGlimmerMaskedCeDescriptorV1);
    return true;
}

bool normalize_sequence_logp_descriptor(
    const NfnNativeTileSequenceLogpDescriptorV1* source,
    NfnNativeTileSequenceLogpDescriptorV1* output,
    bool backward) {
    if (source == nullptr || output == nullptr ||
        source->struct_size < sizeof(NfnNativeTileSequenceLogpDescriptorV1) ||
        source->version != NFN_NATIVE_TILE_GLIMMER_TRAINING_V1 ||
        source->flags != 0 || source->reserved0 != 0 || source->reserved1 != 0 ||
        source->transformed_logits == nullptr || source->targets == nullptr ||
        source->loss_mask == nullptr || source->batch_size <= 0 ||
        source->sequence_length <= 0 || source->vocab_size <= 0 ||
        source->vocab_size > std::numeric_limits<std::int32_t>::max() ||
        (!backward && source->sequence_logp == nullptr) ||
        (backward &&
         (source->grad_sequence_logp == nullptr ||
          source->grad_transformed_logits == nullptr))) {
        return false;
    }
    std::int64_t rows = 0;
    std::int64_t values = 0;
    if (!checked_positive_product(
            source->batch_size, source->sequence_length, &rows) ||
        !checked_positive_product(rows, source->vocab_size, &values)) {
        return false;
    }
    *output = *source;
    output->struct_size = sizeof(NfnNativeTileSequenceLogpDescriptorV1);
    return true;
}

bool normalize_dpo_pairwise_descriptor(
    const NfnNativeTileDpoPairwiseDescriptorV1* source,
    NfnNativeTileDpoPairwiseDescriptorV1* output,
    bool backward) {
    if (source == nullptr || output == nullptr ||
        source->struct_size < sizeof(NfnNativeTileDpoPairwiseDescriptorV1) ||
        source->version != NFN_NATIVE_TILE_GLIMMER_TRAINING_V1 ||
        source->loss_type > NFN_NATIVE_TILE_DPO_LOSS_IPO || source->flags != 0 ||
        source->reserved0 != 0 || source->policy_logp_chosen == nullptr ||
        source->policy_logp_rejected == nullptr ||
        source->reference_logp_chosen == nullptr ||
        source->reference_logp_rejected == nullptr || source->examples <= 0 ||
        !std::isfinite(source->beta) || !(source->beta > 0.0f) ||
        !std::isfinite(source->label_smoothing) ||
        source->label_smoothing < 0.0f || source->label_smoothing > 1.0f ||
        !std::isfinite(source->grad_scale) || source->grad_scale < 0.0f ||
        (!backward &&
         (source->row_loss == nullptr || source->chosen_reward == nullptr ||
          source->rejected_reward == nullptr)) ||
        (backward &&
         (source->grad_policy_logp_chosen == nullptr ||
          source->grad_policy_logp_rejected == nullptr))) {
        return false;
    }
    *output = *source;
    output->struct_size = sizeof(NfnNativeTileDpoPairwiseDescriptorV1);
    return true;
}

bool normalize_masked_reward_head_descriptor(
    const NfnNativeTileMaskedRewardHeadDescriptorV1* source,
    NfnNativeTileMaskedRewardHeadDescriptorV1* output,
    bool backward) {
    NfnNativeTilePackedWeightDescriptorV1 weight{};
    if (source == nullptr || output == nullptr ||
        source->struct_size < sizeof(NfnNativeTileMaskedRewardHeadDescriptorV1) ||
        source->version != NFN_NATIVE_TILE_GLIMMER_TRAINING_V1 ||
        source->flags != 0 || source->reserved0 != 0 || source->hidden == nullptr ||
        source->sequence_mask == nullptr || source->weight == nullptr ||
        source->batch_size <= 0 || source->sequence_length <= 0 ||
        source->hidden_size <= 0 ||
        !normalize_packed_weight_descriptor(source->weight, &weight) ||
        weight.encoding != NFN_NATIVE_TILE_PACKED_WEIGHT_BF16 ||
        weight.output_dim != 1 || weight.input_dim != source->hidden_size ||
        (!backward &&
         (source->reward == nullptr || source->selected_positions == nullptr)) ||
        (backward &&
         (source->selected_positions == nullptr || source->grad_reward == nullptr ||
          source->grad_hidden == nullptr || source->grad_weight == nullptr))) {
        return false;
    }
    std::int64_t rows = 0;
    std::int64_t elements = 0;
    if (!checked_positive_product(
            source->batch_size, source->sequence_length, &rows) ||
        !checked_positive_product(rows, source->hidden_size, &elements)) {
        return false;
    }
    *output = *source;
    output->struct_size = sizeof(NfnNativeTileMaskedRewardHeadDescriptorV1);
    return true;
}

bool normalize_preference_bce_descriptor(
    const NfnNativeTilePreferenceBceDescriptorV1* source,
    NfnNativeTilePreferenceBceDescriptorV1* output,
    bool backward) {
    if (source == nullptr || output == nullptr ||
        source->struct_size < sizeof(NfnNativeTilePreferenceBceDescriptorV1) ||
        source->version != NFN_NATIVE_TILE_GLIMMER_TRAINING_V1 ||
        source->flags != 0 || source->reserved0 != 0 || source->reserved1 != 0 ||
        source->reward_chosen == nullptr || source->reward_rejected == nullptr ||
        source->examples <= 0 || !std::isfinite(source->grad_scale) ||
        source->grad_scale < 0.0f || (!backward && source->row_loss == nullptr) ||
        (backward &&
         (source->grad_reward_chosen == nullptr ||
          source->grad_reward_rejected == nullptr))) {
        return false;
    }
    *output = *source;
    output->struct_size = sizeof(NfnNativeTilePreferenceBceDescriptorV1);
    return true;
}

bool normalize_token_logp_entropy_descriptor(
    const NfnNativeTileTokenLogpEntropyDescriptorV1* source,
    NfnNativeTileTokenLogpEntropyDescriptorV1* output,
    bool backward) {
    if (source == nullptr || output == nullptr ||
        source->struct_size < sizeof(NfnNativeTileTokenLogpEntropyDescriptorV1) ||
        source->version != NFN_NATIVE_TILE_GLIMMER_TRAINING_V1 ||
        source->flags != 0 || source->reserved0 != 0 || source->reserved1 != 0 ||
        source->transformed_logits == nullptr || source->targets == nullptr ||
        source->loss_mask == nullptr || source->rows <= 0 ||
        source->vocab_size <= 0 ||
        source->vocab_size > std::numeric_limits<std::int32_t>::max() ||
        (!backward &&
         (source->token_logp == nullptr || source->token_entropy == nullptr)) ||
        (backward &&
         (source->grad_token_logp == nullptr ||
          source->grad_token_entropy == nullptr ||
          source->grad_transformed_logits == nullptr))) {
        return false;
    }
    std::int64_t values = 0;
    if (!checked_positive_product(source->rows, source->vocab_size, &values)) {
        return false;
    }
    *output = *source;
    output->struct_size = sizeof(NfnNativeTileTokenLogpEntropyDescriptorV1);
    return true;
}

bool normalize_masked_ppo_loss_descriptor(
    const NfnNativeTileMaskedPpoLossDescriptorV1* source,
    NfnNativeTileMaskedPpoLossDescriptorV1* output,
    bool backward) {
    if (source == nullptr || output == nullptr ||
        source->struct_size < sizeof(NfnNativeTileMaskedPpoLossDescriptorV1) ||
        source->version != NFN_NATIVE_TILE_GLIMMER_TRAINING_V1 ||
        (source->flags & ~NFN_NATIVE_TILE_PPO_NORMALIZE_ADVANTAGES) != 0 ||
        source->reserved0 != 0 || source->logp_new == nullptr ||
        source->logp_old == nullptr || source->advantages == nullptr ||
        source->value_new == nullptr || source->value_old == nullptr ||
        source->returns == nullptr || source->loss_mask == nullptr ||
        source->entropy == nullptr || source->rows <= 0 ||
        !std::isfinite(source->clip_range) || !(source->clip_range > 0.0f) ||
        source->clip_range >= 1.0f ||
        !std::isfinite(source->value_coefficient) ||
        source->value_coefficient < 0.0f ||
        !std::isfinite(source->entropy_coefficient) ||
        source->entropy_coefficient < 0.0f ||
        !std::isfinite(source->epsilon) || !(source->epsilon > 0.0f) ||
        (!backward &&
         (source->policy_loss == nullptr || source->value_loss == nullptr ||
          source->entropy_bonus == nullptr || source->total_loss == nullptr)) ||
        (backward &&
         (source->grad_logp_new == nullptr ||
          source->grad_value_new == nullptr ||
          source->grad_entropy == nullptr))) {
        return false;
    }
    *output = *source;
    output->struct_size = sizeof(NfnNativeTileMaskedPpoLossDescriptorV1);
    return true;
}

bool normalize_glimmer_gqa_decode_descriptor(
    const NfnNativeTileGlimmerGqaDecodeDescriptorV1* source,
    NfnNativeTileGlimmerGqaDecodeDescriptorV1* output) {
    if (source == nullptr || output == nullptr ||
        source->struct_size < sizeof(NfnNativeTileGlimmerGqaDecodeDescriptorV1) ||
        source->version != NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1 ||
        source->flags != 0 || source->reserved0 != 0 || source->reserved1 != 0 ||
        source->query == nullptr || source->current_key == nullptr ||
        source->current_value == nullptr || source->key_cache_bf16 == nullptr ||
        source->value_cache_bf16 == nullptr || source->output == nullptr ||
        source->query_heads <= 0 || source->kv_heads <= 0 ||
        source->query_heads % source->kv_heads != 0 ||
        source->head_dim <= 0 || source->head_dim > 256 ||
        source->position < 0 || source->first_key_position < 0 ||
        source->first_key_position > source->position ||
        source->cache_capacity <= 0 || source->cache_row_stride <= 0 ||
        !std::isfinite(source->scale) || !(source->scale > 0.0f)) {
        return false;
    }
    std::int64_t kv_width = 0;
    if (!checked_positive_product(source->kv_heads, source->head_dim, &kv_width) ||
        source->cache_row_stride < kv_width) {
        return false;
    }
    *output = *source;
    output->struct_size = sizeof(NfnNativeTileGlimmerGqaDecodeDescriptorV1);
    return true;
}

bool normalize_glimmer_cache_commit_descriptor(
    const NfnNativeTileGlimmerCacheCommitDescriptorV1* source,
    NfnNativeTileGlimmerCacheCommitDescriptorV1* output) {
    if (source == nullptr || output == nullptr ||
        source->struct_size < sizeof(NfnNativeTileGlimmerCacheCommitDescriptorV1) ||
        source->version != NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1 ||
        source->flags != 0 || source->reserved0 != 0 ||
        source->reserved1 != 0 || source->reserved2 != 0 ||
        source->current_key == nullptr || source->current_value == nullptr ||
        source->key_cache_bf16 == nullptr || source->value_cache_bf16 == nullptr ||
        source->kv_heads <= 0 || source->head_dim <= 0 || source->head_dim > 256 ||
        source->position < 0 || source->cache_capacity <= 0 ||
        source->cache_row_stride <= 0) {
        return false;
    }
    std::int64_t kv_width = 0;
    if (!checked_positive_product(source->kv_heads, source->head_dim, &kv_width) ||
        source->cache_row_stride < kv_width) {
        return false;
    }
    *output = *source;
    output->struct_size = sizeof(NfnNativeTileGlimmerCacheCommitDescriptorV1);
    return true;
}

bool normalize_dflash_block_attention_descriptor(
    const NfnNativeTileDFlashBlockAttentionDescriptorV1* source,
    NfnNativeTileDFlashBlockAttentionDescriptorV1* output) {
    if (source == nullptr || output == nullptr ||
        source->struct_size < sizeof(NfnNativeTileDFlashBlockAttentionDescriptorV1) ||
        source->version != NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1 ||
        (source->flags & ~NFN_NATIVE_TILE_BLOCK_ATTENTION_CAUSAL) != 0 ||
        source->reserved0 != 0 || source->reserved1 != 0 ||
        source->query == nullptr || source->block_key == nullptr ||
        source->block_value == nullptr || source->key_cache_bf16 == nullptr ||
        source->value_cache_bf16 == nullptr || source->output == nullptr ||
        source->query_rows <= 0 || source->block_rows <= 0 ||
        source->query_rows > source->block_rows || source->block_rows > 64 ||
        source->query_heads <= 0 || source->kv_heads <= 0 ||
        source->query_heads % source->kv_heads != 0 ||
        source->head_dim <= 0 || source->head_dim > 256 ||
        source->context_length < 0 || source->sliding_window <= 0 ||
        source->cache_capacity < source->sliding_window ||
        source->cache_row_stride <= 0 || !std::isfinite(source->scale) ||
        !(source->scale > 0.0f)) {
        return false;
    }
    std::int64_t kv_width = 0;
    if (!checked_positive_product(source->kv_heads, source->head_dim, &kv_width) ||
        source->cache_row_stride < kv_width) {
        return false;
    }
    *output = *source;
    output->struct_size = sizeof(NfnNativeTileDFlashBlockAttentionDescriptorV1);
    return true;
}

bool normalize_turboquant_attention_descriptor(
    const NfnNativeTileTurboQuantAttentionDescriptorV1* source,
    NfnNativeTileTurboQuantAttentionDescriptorV1* output) {
    if (source == nullptr || output == nullptr ||
        source->struct_size < sizeof(NfnNativeTileTurboQuantAttentionDescriptorV1) ||
        source->version != NFN_NATIVE_TILE_TURBOQUANT_ATTENTION_V1 ||
        source->flags != 0 || source->reserved0 != 0 ||
        (source->profile != NFN_NATIVE_TILE_TURBOQUANT_PROFILE_MSE_3_5 &&
         source->profile != NFN_NATIVE_TILE_TURBOQUANT_PROFILE_QJL_3_5) ||
        source->query == nullptr || source->key_records == nullptr ||
        source->value_records == nullptr || source->current_key == nullptr ||
        source->current_value == nullptr || source->output == nullptr ||
        source->rotation == nullptr || source->centroids_2bit == nullptr ||
        source->centroids_3bit == nullptr || source->centroids_4bit == nullptr ||
        (source->profile == NFN_NATIVE_TILE_TURBOQUANT_PROFILE_QJL_3_5 &&
         source->qjl_projection == nullptr) ||
        source->batch_size <= 0 || source->num_layers <= 0 ||
        source->layer_index < 0 || source->layer_index >= source->num_layers ||
        source->query_heads <= 0 || source->kv_heads <= 0 ||
        source->query_heads % source->kv_heads != 0 ||
        source->head_dim < 2 ||
        source->head_dim > kTurboQuantAttentionMaxHeadDimension ||
        source->head_dim % 2 != 0 || source->past_sequence_length < 0 ||
        source->past_sequence_length >= kTurboQuantAttentionMaxSequenceLength ||
        source->cache_capacity <= 0 ||
        source->past_sequence_length > source->cache_capacity ||
        !std::isfinite(source->scale) || !(source->scale > 0.0f)) {
        return false;
    }

    const std::int64_t pairs = source->head_dim / 2;
    const std::int64_t value_index_bytes = (pairs * 7 + 7) / 8;
    const std::int64_t key_index_bytes =
        source->profile == NFN_NATIVE_TILE_TURBOQUANT_PROFILE_QJL_3_5
        ? (pairs * 5 + 7) / 8
        : value_index_bytes;
    const std::int64_t sign_bytes = (source->head_dim + 7) / 8;
    const std::int64_t expected_key_record_bytes =
        4 + key_index_bytes +
        (source->profile == NFN_NATIVE_TILE_TURBOQUANT_PROFILE_QJL_3_5
             ? 4 + sign_bytes
             : 0);
    const std::int64_t expected_value_record_bytes = 4 + value_index_bytes;
    if (source->key_record_bytes != expected_key_record_bytes ||
        source->value_record_bytes != expected_value_record_bytes) {
        return false;
    }

    std::int64_t cache_records_per_batch = 0;
    std::int64_t cache_positions_per_batch = 0;
    std::int64_t minimum_key_batch_stride = 0;
    std::int64_t minimum_value_batch_stride = 0;
    std::int64_t query_elements_per_batch = 0;
    std::int64_t kv_elements_per_batch = 0;
    std::int64_t launch_blocks = 0;
    if (!checked_positive_product(
            source->num_layers, source->cache_capacity,
            &cache_positions_per_batch) ||
        !checked_positive_product(
            cache_positions_per_batch, source->kv_heads,
            &cache_records_per_batch) ||
        !checked_positive_product(
            cache_records_per_batch, expected_key_record_bytes,
            &minimum_key_batch_stride) ||
        !checked_positive_product(
            cache_records_per_batch, expected_value_record_bytes,
            &minimum_value_batch_stride) ||
        !checked_positive_product(
            source->query_heads, source->head_dim,
            &query_elements_per_batch) ||
        !checked_positive_product(
            source->kv_heads, source->head_dim,
            &kv_elements_per_batch) ||
        !checked_positive_product(
            source->batch_size, source->query_heads, &launch_blocks) ||
        launch_blocks > static_cast<std::int64_t>(
            std::numeric_limits<unsigned int>::max())) {
        return false;
    }

    *output = *source;
    auto normalize_stride = [](std::int64_t supplied, std::int64_t minimum,
                               std::int64_t* target) {
        if (target == nullptr || supplied < 0 || (supplied != 0 && supplied < minimum)) {
            return false;
        }
        *target = supplied == 0 ? minimum : supplied;
        return true;
    };
    const bool strides_valid = normalize_stride(
               source->key_cache_batch_stride_bytes,
               minimum_key_batch_stride,
               &output->key_cache_batch_stride_bytes) &&
        normalize_stride(
               source->value_cache_batch_stride_bytes,
               minimum_value_batch_stride,
               &output->value_cache_batch_stride_bytes) &&
        normalize_stride(
               source->query_batch_stride,
               query_elements_per_batch,
               &output->query_batch_stride) &&
        normalize_stride(
               source->current_key_batch_stride,
               kv_elements_per_batch,
               &output->current_key_batch_stride) &&
        normalize_stride(
               source->current_value_batch_stride,
               kv_elements_per_batch,
               &output->current_value_batch_stride) &&
        normalize_stride(
               source->output_batch_stride,
               query_elements_per_batch,
               &output->output_batch_stride);
    if (!strides_valid) {
        return false;
    }
    auto batch_span_fits = [source](std::int64_t stride, std::int64_t span) {
        const std::int64_t remaining_batches = source->batch_size - 1;
        return remaining_batches == 0 ||
            (stride <= (std::numeric_limits<std::int64_t>::max() - span) /
                 remaining_batches);
    };
    return batch_span_fits(
               output->key_cache_batch_stride_bytes,
               minimum_key_batch_stride) &&
        batch_span_fits(
               output->value_cache_batch_stride_bytes,
               minimum_value_batch_stride) &&
        batch_span_fits(output->query_batch_stride, query_elements_per_batch) &&
        batch_span_fits(
               output->current_key_batch_stride, kv_elements_per_batch) &&
        batch_span_fits(
               output->current_value_batch_stride, kv_elements_per_batch) &&
        batch_span_fits(output->output_batch_stride, query_elements_per_batch);
}

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    const std::string_view text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "yes" ||
           text == "YES" || text == "on" || text == "ON";
}

bool lm_head_graph_body_serial_enabled() {
    static const bool enabled =
        env_flag_enabled("NFN_TILE_CUDA_LM_HEAD_GRAPH_BODY_SERIAL") ||
        env_flag_enabled("NFN_NATIVE_GPT_LM_HEAD_GRAPH_BODY_SERIAL") ||
        env_flag_enabled("NFN_NATIVE_GPT2_LM_HEAD_GRAPH_BODY_SERIAL");
    return enabled;
}

bool lm_head_graph_upload_enabled() {
    static const bool enabled = []() {
        const char* tile_value = std::getenv("NFN_TILE_CUDA_LM_HEAD_GRAPH_UPLOAD");
        const char* gpt_value = std::getenv("NFN_NATIVE_GPT_LM_HEAD_GRAPH_UPLOAD");
        const char* gpt2_value = std::getenv("NFN_NATIVE_GPT2_LM_HEAD_GRAPH_UPLOAD");
        const char* value = tile_value != nullptr ? tile_value : (gpt_value != nullptr ? gpt_value : gpt2_value);
        if (value == nullptr) {
            return true;
        }
        std::string_view text(value);
        return !(text == "0" || text == "false" || text == "FALSE" ||
                 text == "no" || text == "NO" || text == "off" || text == "OFF");
    }();
    return enabled;
}

bool lm_head_graph_prewarm_thread_cache_enabled() {
    static const bool enabled = []() {
        const char* tile_value = std::getenv("NFN_TILE_CUDA_LM_HEAD_GRAPH_PREWARM_THREAD_CACHE");
        const char* gpt_value = std::getenv("NFN_NATIVE_GPT_LM_HEAD_GRAPH_PREWARM_THREAD_CACHE");
        const char* gpt2_value = std::getenv("NFN_NATIVE_GPT2_LM_HEAD_GRAPH_PREWARM_THREAD_CACHE");
        const char* value = tile_value != nullptr ? tile_value : (gpt_value != nullptr ? gpt_value : gpt2_value);
        if (value == nullptr) {
            return false;
        }
        std::string_view text(value);
        return !(text == "0" || text == "false" || text == "FALSE" ||
                 text == "no" || text == "NO" || text == "off" || text == "OFF");
    }();
    return enabled;
}

bool lm_head_graph_body_cublaslt_enabled() {
    static const bool enabled =
        env_flag_enabled("NFN_TILE_CUDA_LM_HEAD_GRAPH_BODY_CUBLASLT") ||
        env_flag_enabled("NFN_NATIVE_GPT_LM_HEAD_GRAPH_BODY_CUBLASLT") ||
        env_flag_enabled("NFN_NATIVE_GPT2_LM_HEAD_GRAPH_BODY_CUBLASLT");
    return enabled;
}

bool lm_head_graph_body_cublaslt_dhidden_enabled() {
    static const bool enabled =
        lm_head_graph_body_cublaslt_enabled() ||
        env_flag_enabled("NFN_TILE_CUDA_LM_HEAD_GRAPH_BODY_CUBLASLT_DHIDDEN") ||
        env_flag_enabled("NFN_NATIVE_GPT_LM_HEAD_GRAPH_BODY_CUBLASLT_DHIDDEN") ||
        env_flag_enabled("NFN_NATIVE_GPT2_LM_HEAD_GRAPH_BODY_CUBLASLT_DHIDDEN");
    return enabled;
}

bool lm_head_graph_body_cublaslt_dweight_enabled() {
    static const bool enabled =
        lm_head_graph_body_cublaslt_enabled() ||
        env_flag_enabled("NFN_TILE_CUDA_LM_HEAD_GRAPH_BODY_CUBLASLT_DWEIGHT") ||
        env_flag_enabled("NFN_NATIVE_GPT_LM_HEAD_GRAPH_BODY_CUBLASLT_DWEIGHT") ||
        env_flag_enabled("NFN_NATIVE_GPT2_LM_HEAD_GRAPH_BODY_CUBLASLT_DWEIGHT");
    return enabled;
}

bool lm_head_true_fused_cooperative_enabled() {
    static const bool enabled = []() {
        const bool requested =
            env_flag_enabled("NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_COOPERATIVE") ||
            env_flag_enabled("NFN_NATIVE_GPT_LM_HEAD_TRUE_FUSED_COOPERATIVE") ||
            env_flag_enabled("NFN_NATIVE_GPT2_LM_HEAD_TRUE_FUSED_COOPERATIVE");
        if (!requested) {
            return false;
        }
        return neuralfn::tile_cuda::token_cross_entropy_bf16_threads_per_row() ==
               neuralfn::tile_cuda::lm_head_true_fused_required_threads();
    }();
    return enabled;
}

std::atomic<std::int64_t> g_lm_head_cooperative_sequence_launch_count{0};
std::atomic<std::int64_t> g_lm_head_cooperative_sequence_ce_launch_count{0};
std::atomic<std::int64_t> g_lm_head_cooperative_sequence_dhidden_launch_count{0};
std::atomic<std::int64_t> g_lm_head_cooperative_sequence_dweight_launch_count{0};
std::atomic<std::int64_t> g_lm_head_cooperative_sequence_concurrent_count{0};
std::atomic<std::int64_t> g_lm_head_cooperative_sequence_legacy_count{0};
std::atomic<std::int64_t> g_lm_head_cooperative_sequence_loss_bin_count{0};
std::atomic<std::int64_t> g_lm_head_fused_graph_capture_attempt_count{0};
std::atomic<std::int64_t> g_lm_head_fused_graph_capture_success_count{0};
std::atomic<std::int64_t> g_lm_head_fused_graph_upload_success_count{0};
std::atomic<std::int64_t> g_lm_head_fused_graph_upload_failure_count{0};
std::atomic<std::int64_t> g_lm_head_fused_graph_cache_hit_count{0};
std::atomic<std::int64_t> g_lm_head_fused_graph_thread_cache_hit_count{0};
std::atomic<std::int64_t> g_lm_head_fused_graph_replay_count{0};
std::atomic<std::int64_t> g_lm_head_fused_graph_replay_success_count{0};
std::atomic<std::int64_t> g_lm_head_fused_graph_fallback_count{0};
std::atomic<std::int64_t> g_lm_head_graph_body_cublaslt_dhidden_launch_count{0};
std::atomic<std::int64_t> g_lm_head_graph_body_cublaslt_dweight_launch_count{0};
std::atomic<std::int64_t> g_lm_head_graph_body_tile_dhidden_fallback_count{0};
std::atomic<std::int64_t> g_lm_head_graph_body_tile_dweight_fallback_count{0};

struct LmHeadGraphLocalStats {
    std::int64_t cache_hit_count = 0;
    std::int64_t thread_cache_hit_count = 0;
    std::int64_t replay_count = 0;
    std::int64_t replay_success_count = 0;
    std::int64_t fallback_count = 0;
};

LmHeadGraphLocalStats& lm_head_graph_local_stats() {
    thread_local LmHeadGraphLocalStats stats;
    return stats;
}

void reset_lm_head_graph_local_stats() {
    lm_head_graph_local_stats() = LmHeadGraphLocalStats{};
}

struct LmHeadBackwardGraphKey {
    std::uint16_t* logits_bf16 = nullptr;
    const std::uint16_t* targets_u16 = nullptr;
    float* row_losses = nullptr;
    const std::uint16_t* hidden_bf16 = nullptr;
    const std::uint16_t* token_weight_bf16 = nullptr;
    float* grad_hidden = nullptr;
    float* grad_weight = nullptr;
    std::int64_t rows = 0;
    std::int64_t hidden_dim = 0;
    std::int64_t vocab = 0;
    std::int64_t row_stride = 0;
    std::int64_t loss_bin_count = 0;
    float loss_scale = 0.0f;
    float dweight_beta = 0.0f;
    int flags = 0;
};

bool operator==(const LmHeadBackwardGraphKey& lhs, const LmHeadBackwardGraphKey& rhs) {
    return lhs.logits_bf16 == rhs.logits_bf16 &&
        lhs.targets_u16 == rhs.targets_u16 &&
        lhs.row_losses == rhs.row_losses &&
        lhs.hidden_bf16 == rhs.hidden_bf16 &&
        lhs.token_weight_bf16 == rhs.token_weight_bf16 &&
        lhs.grad_hidden == rhs.grad_hidden &&
        lhs.grad_weight == rhs.grad_weight &&
        lhs.rows == rhs.rows &&
        lhs.hidden_dim == rhs.hidden_dim &&
        lhs.vocab == rhs.vocab &&
        lhs.row_stride == rhs.row_stride &&
        lhs.loss_bin_count == rhs.loss_bin_count &&
        lhs.loss_scale == rhs.loss_scale &&
        lhs.dweight_beta == rhs.dweight_beta &&
        lhs.flags == rhs.flags;
}

struct LmHeadBackwardGraphEntry {
    LmHeadBackwardGraphKey key;
    cudaGraphExec_t exec = nullptr;
};

struct LmHeadBackwardThreadGraphCache {
    LmHeadBackwardGraphEntry entries[kLmHeadGraphThreadCacheCapacity];
    int count = 0;
};

std::mutex g_lm_head_backward_graph_mutex;
std::vector<LmHeadBackwardGraphEntry> g_lm_head_backward_graph_cache;
cudaStream_t g_lm_head_backward_graph_capture_stream = nullptr;

LmHeadBackwardThreadGraphCache& lm_head_backward_thread_graph_cache() {
    thread_local LmHeadBackwardThreadGraphCache cache;
    return cache;
}

cudaGraphExec_t find_lm_head_backward_thread_graph(const LmHeadBackwardGraphKey& key) {
    LmHeadBackwardThreadGraphCache& cache = lm_head_backward_thread_graph_cache();
    for (int i = 0; i < cache.count; ++i) {
        if (cache.entries[i].key == key) {
            LmHeadGraphLocalStats& stats = lm_head_graph_local_stats();
            stats.cache_hit_count += 1;
            stats.thread_cache_hit_count += 1;
            return cache.entries[i].exec;
        }
    }
    return nullptr;
}

void store_lm_head_backward_thread_graph(const LmHeadBackwardGraphKey& key, cudaGraphExec_t exec) {
    LmHeadBackwardThreadGraphCache& cache = lm_head_backward_thread_graph_cache();
    if (cache.count < kLmHeadGraphThreadCacheCapacity) {
        cache.entries[cache.count++] = {key, exec};
    } else {
        cache.entries[kLmHeadGraphThreadCacheCapacity - 1] = {key, exec};
    }
}

struct LmHeadCooperativeStreams {
    cudaStream_t dhidden = nullptr;
    cudaStream_t dweight = nullptr;
    cudaEvent_t ce_done = nullptr;
    cudaEvent_t dhidden_done = nullptr;
    cudaEvent_t dweight_done = nullptr;
    int status = 0;
};

LmHeadCooperativeStreams& lm_head_cooperative_streams() {
    static LmHeadCooperativeStreams resources;
    static std::once_flag init_once;
    std::call_once(init_once, []() {
        int status = static_cast<int>(cudaStreamCreateWithFlags(
            &resources.dhidden,
            cudaStreamNonBlocking));
        if (status == 0) {
            status = static_cast<int>(cudaStreamCreateWithFlags(
                &resources.dweight,
                cudaStreamNonBlocking));
        }
        if (status == 0) {
            status = static_cast<int>(cudaEventCreateWithFlags(
                &resources.ce_done,
                cudaEventDisableTiming));
        }
        if (status == 0) {
            status = static_cast<int>(cudaEventCreateWithFlags(
                &resources.dhidden_done,
                cudaEventDisableTiming));
        }
        if (status == 0) {
            status = static_cast<int>(cudaEventCreateWithFlags(
                &resources.dweight_done,
                cudaEventDisableTiming));
        }
        resources.status = status;
    });
    return resources;
}

void reset_lm_head_cooperative_sequence_stats() {
    g_lm_head_cooperative_sequence_launch_count.store(0, std::memory_order_relaxed);
    g_lm_head_cooperative_sequence_ce_launch_count.store(0, std::memory_order_relaxed);
    g_lm_head_cooperative_sequence_dhidden_launch_count.store(0, std::memory_order_relaxed);
    g_lm_head_cooperative_sequence_dweight_launch_count.store(0, std::memory_order_relaxed);
    g_lm_head_cooperative_sequence_concurrent_count.store(0, std::memory_order_relaxed);
    g_lm_head_cooperative_sequence_legacy_count.store(0, std::memory_order_relaxed);
    g_lm_head_cooperative_sequence_loss_bin_count.store(0, std::memory_order_relaxed);
    g_lm_head_fused_graph_capture_attempt_count.store(0, std::memory_order_relaxed);
    g_lm_head_fused_graph_capture_success_count.store(0, std::memory_order_relaxed);
    g_lm_head_fused_graph_upload_success_count.store(0, std::memory_order_relaxed);
    g_lm_head_fused_graph_upload_failure_count.store(0, std::memory_order_relaxed);
    g_lm_head_fused_graph_cache_hit_count.store(0, std::memory_order_relaxed);
    g_lm_head_fused_graph_thread_cache_hit_count.store(0, std::memory_order_relaxed);
    g_lm_head_fused_graph_replay_count.store(0, std::memory_order_relaxed);
    g_lm_head_fused_graph_replay_success_count.store(0, std::memory_order_relaxed);
    g_lm_head_fused_graph_fallback_count.store(0, std::memory_order_relaxed);
    g_lm_head_graph_body_cublaslt_dhidden_launch_count.store(0, std::memory_order_relaxed);
    g_lm_head_graph_body_cublaslt_dweight_launch_count.store(0, std::memory_order_relaxed);
    g_lm_head_graph_body_tile_dhidden_fallback_count.store(0, std::memory_order_relaxed);
    g_lm_head_graph_body_tile_dweight_fallback_count.store(0, std::memory_order_relaxed);
    reset_lm_head_graph_local_stats();
}

std::int64_t lm_head_cooperative_loss_bin_count_from_flags(int flags, std::int64_t rows) {
    const std::int64_t encoded =
        static_cast<std::int64_t>(static_cast<unsigned int>(flags) >> kLmHeadCooperativeLossBinCountShift);
    const std::int64_t requested = encoded > 0 ? encoded : 1024;
    return std::max<std::int64_t>(1, std::min<std::int64_t>(rows, requested));
}

void launch_lm_head_classifier_backward_graph_body_bf16_u16(
    std::uint16_t* logits_bf16,
    const std::uint16_t* targets_u16,
    float* row_losses,
    const std::uint16_t* hidden_bf16,
    const std::uint16_t* token_weight_bf16,
    float* grad_hidden,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t hidden_dim,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    float dweight_beta,
    int flags,
    cudaStream_t stream) {
    const bool no_loss = (flags & kLmHeadCooperativeFlagNoLoss) != 0;
    if ((flags & kLmHeadCooperativeFlagLossBins) != 0) {
        neuralfn::tile_cuda::launch_lm_head_classifier_backward_loss_bins_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
            logits_bf16,
            targets_u16,
            row_losses,
            rows,
            vocab,
            row_stride,
            lm_head_cooperative_loss_bin_count_from_flags(flags, rows),
            loss_scale,
            stream);
    } else if (no_loss) {
        neuralfn::tile_cuda::launch_lm_head_classifier_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace(
            logits_bf16,
            targets_u16,
            nullptr,
            nullptr,
            rows,
            vocab,
            row_stride,
            loss_scale,
            stream);
    } else {
        neuralfn::tile_cuda::launch_lm_head_classifier_backward_row_losses_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
            logits_bf16,
            targets_u16,
            row_losses,
            rows,
            vocab,
            row_stride,
            loss_scale,
            stream);
    }
    const bool serial_graph_body = lm_head_graph_body_serial_enabled();
    const bool cublaslt_graph_body_dhidden = lm_head_graph_body_cublaslt_dhidden_enabled();
    const bool cublaslt_graph_body_dweight = lm_head_graph_body_cublaslt_dweight_enabled();
    LmHeadCooperativeStreams& cooperative_streams = lm_head_cooperative_streams();
    if (!serial_graph_body && cooperative_streams.status == 0) {
        cudaEventRecord(cooperative_streams.ce_done, stream);
        cudaStreamWaitEvent(cooperative_streams.dhidden, cooperative_streams.ce_done, 0);
        cudaStreamWaitEvent(cooperative_streams.dweight, cooperative_streams.ce_done, 0);
        const bool cublaslt_dhidden_launched =
            cublaslt_graph_body_dhidden &&
            neuralfn::tile_cuda::cublaslt_linear_backward_input_bf16_bits_weight_bf16_strided_float32(
                logits_bf16,
                token_weight_bf16,
                grad_hidden,
                rows,
                hidden_dim,
                vocab,
                row_stride,
                cooperative_streams.dhidden);
        if (!cublaslt_dhidden_launched) {
            g_lm_head_graph_body_tile_dhidden_fallback_count.fetch_add(1, std::memory_order_relaxed);
            neuralfn::tile_cuda::launch_linear_backward_input_bf16_bits_weight_bf16_float32(
                logits_bf16,
                token_weight_bf16,
                grad_hidden,
                rows,
                hidden_dim,
                row_stride,
                cooperative_streams.dhidden);
        } else {
            g_lm_head_graph_body_cublaslt_dhidden_launch_count.fetch_add(1, std::memory_order_relaxed);
        }
        const bool cublaslt_dweight_launched =
            cublaslt_graph_body_dweight &&
            neuralfn::tile_cuda::cublaslt_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_float32_beta(
                hidden_bf16,
                logits_bf16,
                grad_weight,
                rows,
                hidden_dim,
                vocab,
                row_stride,
                dweight_beta,
                cooperative_streams.dweight);
        if (!cublaslt_dweight_launched) {
            g_lm_head_graph_body_tile_dweight_fallback_count.fetch_add(1, std::memory_order_relaxed);
            neuralfn::tile_cuda::launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta(
                hidden_bf16,
                logits_bf16,
                grad_weight,
                rows,
                hidden_dim,
                row_stride,
                dweight_beta,
                cooperative_streams.dweight);
        } else {
            g_lm_head_graph_body_cublaslt_dweight_launch_count.fetch_add(1, std::memory_order_relaxed);
        }
        cudaEventRecord(cooperative_streams.dhidden_done, cooperative_streams.dhidden);
        cudaEventRecord(cooperative_streams.dweight_done, cooperative_streams.dweight);
        cudaStreamWaitEvent(stream, cooperative_streams.dhidden_done, 0);
        cudaStreamWaitEvent(stream, cooperative_streams.dweight_done, 0);
        return;
    }
    const bool cublaslt_dhidden_launched =
        cublaslt_graph_body_dhidden &&
        neuralfn::tile_cuda::cublaslt_linear_backward_input_bf16_bits_weight_bf16_strided_float32(
            logits_bf16,
            token_weight_bf16,
            grad_hidden,
            rows,
            hidden_dim,
            vocab,
            row_stride,
            stream);
    if (!cublaslt_dhidden_launched) {
        g_lm_head_graph_body_tile_dhidden_fallback_count.fetch_add(1, std::memory_order_relaxed);
        neuralfn::tile_cuda::launch_linear_backward_input_bf16_bits_weight_bf16_float32(
            logits_bf16,
            token_weight_bf16,
            grad_hidden,
            rows,
            hidden_dim,
            row_stride,
            stream);
    } else {
        g_lm_head_graph_body_cublaslt_dhidden_launch_count.fetch_add(1, std::memory_order_relaxed);
    }
    const bool cublaslt_dweight_launched =
        cublaslt_graph_body_dweight &&
        neuralfn::tile_cuda::cublaslt_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_float32_beta(
            hidden_bf16,
            logits_bf16,
            grad_weight,
            rows,
            hidden_dim,
            vocab,
            row_stride,
            dweight_beta,
            stream);
    if (!cublaslt_dweight_launched) {
        g_lm_head_graph_body_tile_dweight_fallback_count.fetch_add(1, std::memory_order_relaxed);
        neuralfn::tile_cuda::launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta(
            hidden_bf16,
            logits_bf16,
            grad_weight,
            rows,
            hidden_dim,
            row_stride,
            dweight_beta,
            stream);
    } else {
        g_lm_head_graph_body_cublaslt_dweight_launch_count.fetch_add(1, std::memory_order_relaxed);
    }
}

int capture_lm_head_classifier_backward_graph_bf16_u16(
    const LmHeadBackwardGraphKey& key,
    cudaGraphExec_t* exec_out) {
    g_lm_head_fused_graph_capture_attempt_count.fetch_add(1, std::memory_order_relaxed);
    if (g_lm_head_backward_graph_capture_stream == nullptr) {
        const cudaError_t create_status = cudaStreamCreateWithFlags(
            &g_lm_head_backward_graph_capture_stream,
            cudaStreamNonBlocking);
        if (create_status != cudaSuccess) {
            return static_cast<int>(create_status);
        }
    }
    cudaStream_t stream = g_lm_head_backward_graph_capture_stream;
    cudaGraph_t graph = nullptr;
    cudaError_t status = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (status != cudaSuccess) {
        cudaGetLastError();
        return static_cast<int>(status);
    }
    launch_lm_head_classifier_backward_graph_body_bf16_u16(
        key.logits_bf16,
        key.targets_u16,
        key.row_losses,
        key.hidden_bf16,
        key.token_weight_bf16,
        key.grad_hidden,
        key.grad_weight,
        key.rows,
        key.hidden_dim,
        key.vocab,
        key.row_stride,
        key.loss_scale,
        key.dweight_beta,
        key.flags,
        stream);
    status = cudaStreamEndCapture(stream, &graph);
    if (status != cudaSuccess) {
        cudaGetLastError();
        return static_cast<int>(status);
    }
    cudaGraphExec_t exec = nullptr;
    status = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    const cudaError_t destroy_status = cudaGraphDestroy(graph);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    if (destroy_status != cudaSuccess) {
        cudaGraphExecDestroy(exec);
        return static_cast<int>(destroy_status);
    }
    if (lm_head_graph_upload_enabled()) {
        const cudaError_t upload_status = cudaGraphUpload(exec, stream);
        if (upload_status == cudaSuccess) {
            g_lm_head_fused_graph_upload_success_count.fetch_add(1, std::memory_order_relaxed);
        } else {
            g_lm_head_fused_graph_upload_failure_count.fetch_add(1, std::memory_order_relaxed);
            cudaGraphExecDestroy(exec);
            return static_cast<int>(upload_status);
        }
    }
    *exec_out = exec;
    g_lm_head_fused_graph_capture_success_count.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

int run_lm_head_classifier_backward_graph_bf16_u16(
    const LmHeadBackwardGraphKey& key,
    cudaStream_t stream) {
    cudaGraphExec_t exec = find_lm_head_backward_thread_graph(key);
    if (exec == nullptr) {
        std::lock_guard<std::mutex> lock(g_lm_head_backward_graph_mutex);
        for (const auto& entry : g_lm_head_backward_graph_cache) {
            if (entry.key == key) {
                exec = entry.exec;
                LmHeadGraphLocalStats& stats = lm_head_graph_local_stats();
                stats.cache_hit_count += 1;
                break;
            }
        }
        if (exec == nullptr) {
            const int status = capture_lm_head_classifier_backward_graph_bf16_u16(key, &exec);
            if (status != 0) {
                return status;
            }
            g_lm_head_backward_graph_cache.push_back({key, exec});
        }
        store_lm_head_backward_thread_graph(key, exec);
    }
    LmHeadGraphLocalStats& stats = lm_head_graph_local_stats();
    stats.replay_count += 1;
    const int launch_status = static_cast<int>(cudaGraphLaunch(exec, stream));
    if (launch_status == 0) {
        stats.replay_success_count += 1;
    }
    return launch_status;
}

int prewarm_lm_head_classifier_backward_graph_bf16_u16(
    const LmHeadBackwardGraphKey& key) {
    const bool prime_thread_cache = lm_head_graph_prewarm_thread_cache_enabled();
    if (prime_thread_cache && find_lm_head_backward_thread_graph(key) != nullptr) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(g_lm_head_backward_graph_mutex);
    for (const auto& entry : g_lm_head_backward_graph_cache) {
        if (entry.key == key) {
            g_lm_head_fused_graph_cache_hit_count.fetch_add(1, std::memory_order_relaxed);
            if (prime_thread_cache) {
                store_lm_head_backward_thread_graph(key, entry.exec);
            }
            return 0;
        }
    }
    cudaGraphExec_t exec = nullptr;
    const int status = capture_lm_head_classifier_backward_graph_bf16_u16(key, &exec);
    if (status != 0) {
        return status;
    }
    g_lm_head_backward_graph_cache.push_back({key, exec});
    if (prime_thread_cache) {
        store_lm_head_backward_thread_graph(key, exec);
    }
    return 0;
}

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

int nfn_native_tile_strict_math_abi_version() {
#if defined(NFN_TILE_CUDA_STRICT_MATH_BUILD) && \
    !defined(NFN_TILE_CUDA_USE_CUBLAS_LINEAR) && \
    !defined(NFN_TILE_CUDA_USE_TK_ATTENTION) && \
    !defined(__CUDA_FAST_MATH__)
    return 1;
#else
    return 0;
#endif
}

int nfn_native_tile_turboquant_attention_abi_version() {
    return NFN_NATIVE_TILE_TURBOQUANT_ATTENTION_V1;
}

int nfn_native_tile_packed_weight_abi_version() {
    return NFN_NATIVE_TILE_PACKED_WEIGHT_V1;
}

int nfn_native_tile_glimmer_inference_abi_version() {
    return NFN_NATIVE_TILE_GLIMMER_INFERENCE_V1;
}

int nfn_native_tile_glimmer_vision_abi_version() {
    return NFN_NATIVE_TILE_GLIMMER_VISION_V1;
}

int nfn_native_tile_glimmer_training_abi_version() {
    return NFN_NATIVE_TILE_GLIMMER_TRAINING_V1;
}

int nfn_native_tile_packed_weight_validate_v1(
    const NfnNativeTilePackedWeightDescriptorV1* descriptor) {
    NfnNativeTilePackedWeightDescriptorV1 normalized{};
    return normalize_packed_weight_descriptor(descriptor, &normalized)
        ? static_cast<int>(cudaSuccess)
        : static_cast<int>(cudaErrorInvalidValue);
}

int nfn_native_tile_packed_weight_dequantize_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* descriptor,
    float* output) {
    NfnNativeTilePackedWeightDescriptorV1 normalized{};
    if (!normalize_packed_weight_descriptor(descriptor, &normalized) || output == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_packed_weight_dequantize_float32_v1(
        normalized, output, as_stream(normalized.cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_packed_weight_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* descriptor,
    const float* input,
    const float* bias,
    float* output,
    std::int64_t rows,
    bool has_bias) {
    NfnNativeTilePackedWeightDescriptorV1 normalized{};
    if (!normalize_packed_weight_descriptor(descriptor, &normalized) ||
        input == nullptr || output == nullptr || rows <= 0 ||
        (has_bias && bias == nullptr)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_linear_packed_weight_float32_v1(
        normalized,
        input,
        bias,
        output,
        rows,
        has_bias,
        as_stream(normalized.cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_input_packed_weight_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* descriptor,
    const float* grad_output,
    float* grad_input,
    std::int64_t rows) {
    NfnNativeTilePackedWeightDescriptorV1 normalized{};
    if (!normalize_packed_weight_descriptor(descriptor, &normalized) ||
        grad_output == nullptr || grad_input == nullptr || rows <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_linear_backward_input_packed_weight_float32_v1(
        normalized,
        grad_output,
        grad_input,
        rows,
        as_stream(normalized.cuda_stream));
    return launch_status();
}

int nfn_native_tile_glimmer_embedding_gather_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* descriptor,
    std::int64_t token_id,
    float* output) {
    NfnNativeTilePackedWeightDescriptorV1 normalized{};
    if (!normalize_packed_weight_descriptor(descriptor, &normalized) ||
        token_id < 0 || token_id >= normalized.output_dim || output == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_glimmer_embedding_gather_float32_v1(
        normalized, token_id, output, as_stream(normalized.cuda_stream));
    return launch_status();
}

int nfn_native_tile_glimmer_embedding_batch_i32_float32_v1(
    const NfnNativeTilePackedWeightDescriptorV1* descriptor,
    const std::int32_t* token_ids,
    float* output,
    std::int64_t rows) {
    NfnNativeTilePackedWeightDescriptorV1 normalized{};
    if (!normalize_packed_weight_descriptor(descriptor, &normalized) ||
        token_ids == nullptr || output == nullptr || rows <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_glimmer_embedding_batch_i32_float32_v1(
        normalized, token_ids, output, rows, as_stream(normalized.cuda_stream));
    return launch_status();
}

int nfn_native_tile_glimmer_rms_norm_affine_float32_v1(
    const float* input,
    const NfnNativeTilePackedWeightDescriptorV1* weight,
    float* output,
    std::int64_t rows,
    std::int64_t width,
    float eps,
    bool centered,
    void* cuda_stream) {
    NfnNativeTilePackedWeightDescriptorV1 normalized{};
    const bool has_weight = weight != nullptr;
    if (input == nullptr || output == nullptr || rows <= 0 || width <= 0 ||
        width > 65536 || !std::isfinite(eps) || !(eps > 0.0f) ||
        (has_weight &&
         (!normalize_packed_weight_descriptor(weight, &normalized) ||
          normalized.output_dim != 1 || normalized.input_dim != width))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (!has_weight) {
        normalized.struct_size = sizeof(normalized);
        normalized.version = NFN_NATIVE_TILE_PACKED_WEIGHT_V1;
    }
    neuralfn::tile_cuda::launch_glimmer_rms_norm_affine_float32_v1(
        input, normalized, has_weight, output, rows, width, eps, centered,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_glimmer_positioned_rope_float32_v1(
    float* query,
    float* key,
    std::int64_t query_heads,
    std::int64_t kv_heads,
    std::int64_t head_dim,
    std::int64_t position,
    float theta,
    std::uint32_t layout,
    void* cuda_stream) {
    if (query == nullptr || key == nullptr || query_heads <= 0 || kv_heads <= 0 ||
        query_heads % kv_heads != 0 || head_dim <= 0 || head_dim > 256 ||
        head_dim % 2 != 0 || position < 0 || !std::isfinite(theta) ||
        !(theta > 0.0f) ||
        (layout != NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT &&
         layout != NFN_NATIVE_TILE_GLIMMER_ROPE_INTERLEAVED)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_glimmer_positioned_rope_float32_v1(
        query, key, query_heads, kv_heads, head_dim, position, theta, layout,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_glimmer_gqa_decode_float32_v1(
    const NfnNativeTileGlimmerGqaDecodeDescriptorV1* descriptor) {
    NfnNativeTileGlimmerGqaDecodeDescriptorV1 normalized{};
    if (!normalize_glimmer_gqa_decode_descriptor(descriptor, &normalized)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_glimmer_gqa_decode_float32_v1(
        normalized, as_stream(normalized.cuda_stream));
    return launch_status();
}

int nfn_native_tile_glimmer_cache_commit_bf16_v1(
    const NfnNativeTileGlimmerCacheCommitDescriptorV1* descriptor) {
    NfnNativeTileGlimmerCacheCommitDescriptorV1 normalized{};
    if (!normalize_glimmer_cache_commit_descriptor(descriptor, &normalized)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_glimmer_cache_commit_bf16_v1(
        normalized, as_stream(normalized.cuda_stream));
    return launch_status();
}

int nfn_native_tile_dflash_block_attention_float32_v1(
    const NfnNativeTileDFlashBlockAttentionDescriptorV1* descriptor) {
    NfnNativeTileDFlashBlockAttentionDescriptorV1 normalized{};
    if (!normalize_dflash_block_attention_descriptor(descriptor, &normalized)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_dflash_block_attention_float32_v1(
        normalized, as_stream(normalized.cuda_stream));
    return launch_status();
}

int nfn_native_tile_glimmer_vision_prepare_float32_v1(
    const NfnNativeTileGlimmerVisionPrepareDescriptorV1* descriptor) {
    if (descriptor == nullptr ||
        descriptor->struct_size != sizeof(*descriptor) ||
        descriptor->version != NFN_NATIVE_TILE_GLIMMER_VISION_V1 ||
        descriptor->projected == nullptr || descriptor->position_table == nullptr ||
        descriptor->corner_indices == nullptr || descriptor->corner_weights == nullptr ||
        descriptor->permutation == nullptr || descriptor->output == nullptr ||
        descriptor->rows <= 0 || descriptor->width <= 0 ||
        descriptor->position_rows <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_glimmer_vision_prepare_float32_v1(
        *descriptor, as_stream(descriptor->cuda_stream));
    return launch_status();
}

int nfn_native_tile_glimmer_vision_layer_norm_float32_v1(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::int64_t rows,
    std::int64_t width,
    float eps,
    void* cuda_stream) {
    if (input == nullptr || weight == nullptr || bias == nullptr || output == nullptr ||
        rows <= 0 || width <= 0 || width > 8192 || !std::isfinite(eps) ||
        !(eps > 0.0f)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_glimmer_vision_layer_norm_float32_v1(
        input, weight, bias, output, rows, width, eps, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_glimmer_vision_attention_float32_v1(
    const NfnNativeTileGlimmerVisionAttentionDescriptorV1* descriptor) {
    if (descriptor == nullptr ||
        descriptor->struct_size != sizeof(*descriptor) ||
        descriptor->version != NFN_NATIVE_TILE_GLIMMER_VISION_V1 ||
        descriptor->interleaved_rope > 1 || descriptor->reserved0 != 0 ||
        descriptor->reserved1 != 0 || descriptor->query == nullptr ||
        descriptor->key == nullptr || descriptor->value == nullptr ||
        descriptor->position_width == nullptr ||
        descriptor->position_height == nullptr ||
        descriptor->row_begin == nullptr || descriptor->row_end == nullptr ||
        descriptor->output == nullptr || descriptor->rows <= 0 ||
        descriptor->heads <= 0 || descriptor->head_dim <= 0 ||
        descriptor->head_dim > 256 || descriptor->head_dim % 4 != 0 ||
        !std::isfinite(descriptor->rope_theta) || !(descriptor->rope_theta > 0.0f)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_glimmer_vision_attention_float32_v1(
        *descriptor, as_stream(descriptor->cuda_stream));
    return launch_status();
}

int nfn_native_tile_glimmer_vision_pixel_shuffle_float32_v1(
    const NfnNativeTileGlimmerVisionPixelShuffleDescriptorV1* descriptor) {
    if (descriptor == nullptr ||
        descriptor->struct_size != sizeof(*descriptor) ||
        descriptor->version != NFN_NATIVE_TILE_GLIMMER_VISION_V1 ||
        descriptor->reordered_hidden == nullptr ||
        descriptor->source_rows == nullptr || descriptor->output == nullptr ||
        descriptor->merged_rows <= 0 || descriptor->hidden_size <= 0 ||
        descriptor->merge_area <= 0 || descriptor->merge_area > 16) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_glimmer_vision_pixel_shuffle_float32_v1(
        *descriptor, as_stream(descriptor->cuda_stream));
    return launch_status();
}

int nfn_native_tile_glimmer_sigmoid_gate_float32_v1(
    const float* values,
    const float* gate,
    float* output,
    std::int64_t count,
    void* cuda_stream) {
    if (values == nullptr || gate == nullptr || output == nullptr || count <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_glimmer_sigmoid_gate_float32_v1(
        values, gate, output, count, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_glimmer_logit_transform_float32_v1(
    float* logits,
    std::int64_t count,
    float multiplier,
    float softcap,
    void* cuda_stream) {
    if (logits == nullptr || count <= 0 || !std::isfinite(multiplier) ||
        !(multiplier > 0.0f) || !std::isfinite(softcap) || !(softcap > 0.0f)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_glimmer_logit_transform_float32_v1(
        logits, count, multiplier, softcap, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_glimmer_attention_forward_float32_v1(
    const NfnNativeTileGlimmerAttentionTrainingDescriptorV1* descriptor) {
    NfnNativeTileGlimmerAttentionTrainingDescriptorV1 normalized{};
    if (!normalize_glimmer_attention_training_descriptor(
            descriptor, &normalized, false)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_glimmer_attention_forward_float32_v1(
        normalized, as_stream(normalized.cuda_stream));
    return launch_status();
}

int nfn_native_tile_glimmer_attention_backward_float32_v1(
    const NfnNativeTileGlimmerAttentionTrainingDescriptorV1* descriptor) {
    NfnNativeTileGlimmerAttentionTrainingDescriptorV1 normalized{};
    if (!normalize_glimmer_attention_training_descriptor(
            descriptor, &normalized, true)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_glimmer_attention_backward_float32_v1(
        normalized, as_stream(normalized.cuda_stream));
    return launch_status();
}

int nfn_native_tile_glimmer_rms_norm_backward_float32_v1(
    const NfnNativeTileGlimmerRmsNormBackwardDescriptorV1* descriptor) {
    if (descriptor == nullptr ||
        descriptor->struct_size < sizeof(NfnNativeTileGlimmerRmsNormBackwardDescriptorV1) ||
        descriptor->version != NFN_NATIVE_TILE_GLIMMER_TRAINING_V1 ||
        descriptor->flags != 0 || descriptor->reserved0 != 0 ||
        descriptor->input == nullptr || descriptor->grad_output == nullptr ||
        descriptor->grad_input == nullptr || descriptor->rows <= 0 ||
        descriptor->width <= 0 || descriptor->width > 65536 ||
        !std::isfinite(descriptor->eps) || !(descriptor->eps > 0.0f) ||
        descriptor->centered > 1 ||
        (descriptor->weight == nullptr && descriptor->grad_weight != nullptr)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    NfnNativeTilePackedWeightDescriptorV1 normalized_weight{};
    const bool has_weight = descriptor->weight != nullptr;
    if (has_weight &&
        (!normalize_packed_weight_descriptor(descriptor->weight, &normalized_weight) ||
         normalized_weight.output_dim != 1 ||
         normalized_weight.input_dim != descriptor->width)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (!has_weight) {
        normalized_weight.struct_size = sizeof(normalized_weight);
        normalized_weight.version = NFN_NATIVE_TILE_PACKED_WEIGHT_V1;
    }
    NfnNativeTileGlimmerRmsNormBackwardDescriptorV1 normalized = *descriptor;
    normalized.struct_size = sizeof(normalized);
    neuralfn::tile_cuda::launch_glimmer_rms_norm_backward_float32_v1(
        normalized, normalized_weight, has_weight,
        as_stream(normalized.cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (query == nullptr || key == nullptr || rows <= 0 || query_heads <= 0 ||
        kv_heads <= 0 || query_heads % kv_heads != 0 || head_dim <= 0 ||
        head_dim > 256 || head_dim % 2 != 0 || start_position < 0 ||
        !std::isfinite(theta) || !(theta > 0.0f) ||
        (layout != NFN_NATIVE_TILE_GLIMMER_ROPE_HALF_SPLIT &&
         layout != NFN_NATIVE_TILE_GLIMMER_ROPE_INTERLEAVED)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_glimmer_positioned_rope_batch_float32_v1(
        query, key, rows, query_heads, kv_heads, head_dim, start_position,
        theta, layout, inverse, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_glimmer_sigmoid_gate_backward_float32_v1(
    const float* values,
    const float* gate,
    const float* grad_output,
    float* grad_values,
    float* grad_gate,
    std::int64_t count,
    void* cuda_stream) {
    if (values == nullptr || gate == nullptr || grad_output == nullptr ||
        grad_values == nullptr || grad_gate == nullptr || count <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_glimmer_sigmoid_gate_backward_float32_v1(
        values, gate, grad_output, grad_values, grad_gate, count,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_glimmer_logit_transform_backward_float32_v1(
    const float* transformed_logits,
    const float* grad_transformed_logits,
    float* grad_raw_logits,
    std::int64_t count,
    float multiplier,
    float softcap,
    void* cuda_stream) {
    if (transformed_logits == nullptr || grad_transformed_logits == nullptr ||
        grad_raw_logits == nullptr || count <= 0 || !std::isfinite(multiplier) ||
        !(multiplier > 0.0f) || !std::isfinite(softcap) || !(softcap > 0.0f)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_glimmer_logit_transform_backward_float32_v1(
        transformed_logits, grad_transformed_logits, grad_raw_logits, count,
        multiplier, softcap, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_glimmer_masked_cross_entropy_i32_float32_v1(
    const NfnNativeTileGlimmerMaskedCeDescriptorV1* descriptor) {
    NfnNativeTileGlimmerMaskedCeDescriptorV1 normalized{};
    if (!normalize_glimmer_masked_ce_descriptor(descriptor, &normalized)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_glimmer_masked_cross_entropy_i32_float32_v1(
        normalized, as_stream(normalized.cuda_stream));
    return launch_status();
}

int nfn_native_tile_sequence_logp_i32_float32_forward_v1(
    const NfnNativeTileSequenceLogpDescriptorV1* descriptor) {
    NfnNativeTileSequenceLogpDescriptorV1 normalized{};
    if (!normalize_sequence_logp_descriptor(descriptor, &normalized, false)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_sequence_logp_i32_float32_forward_v1(
        normalized, as_stream(normalized.cuda_stream));
    return launch_status();
}

int nfn_native_tile_sequence_logp_i32_float32_backward_v1(
    const NfnNativeTileSequenceLogpDescriptorV1* descriptor) {
    NfnNativeTileSequenceLogpDescriptorV1 normalized{};
    if (!normalize_sequence_logp_descriptor(descriptor, &normalized, true)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_sequence_logp_i32_float32_backward_v1(
        normalized, as_stream(normalized.cuda_stream));
    return launch_status();
}

int nfn_native_tile_dpo_pairwise_loss_float32_forward_v1(
    const NfnNativeTileDpoPairwiseDescriptorV1* descriptor) {
    NfnNativeTileDpoPairwiseDescriptorV1 normalized{};
    if (!normalize_dpo_pairwise_descriptor(descriptor, &normalized, false)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_dpo_pairwise_loss_float32_forward_v1(
        normalized, as_stream(normalized.cuda_stream));
    return launch_status();
}

int nfn_native_tile_dpo_pairwise_loss_float32_backward_v1(
    const NfnNativeTileDpoPairwiseDescriptorV1* descriptor) {
    NfnNativeTileDpoPairwiseDescriptorV1 normalized{};
    if (!normalize_dpo_pairwise_descriptor(descriptor, &normalized, true)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_dpo_pairwise_loss_float32_backward_v1(
        normalized, as_stream(normalized.cuda_stream));
    return launch_status();
}

int nfn_native_tile_masked_reward_head_float32_forward_v1(
    const NfnNativeTileMaskedRewardHeadDescriptorV1* descriptor) {
    NfnNativeTileMaskedRewardHeadDescriptorV1 normalized{};
    if (!normalize_masked_reward_head_descriptor(descriptor, &normalized, false)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_masked_reward_head_float32_forward_v1(
        normalized, as_stream(normalized.cuda_stream));
    return launch_status();
}

int nfn_native_tile_masked_reward_head_float32_backward_v1(
    const NfnNativeTileMaskedRewardHeadDescriptorV1* descriptor) {
    NfnNativeTileMaskedRewardHeadDescriptorV1 normalized{};
    if (!normalize_masked_reward_head_descriptor(descriptor, &normalized, true)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_masked_reward_head_float32_backward_v1(
        normalized, as_stream(normalized.cuda_stream));
    return launch_status();
}

int nfn_native_tile_preference_bce_loss_float32_forward_v1(
    const NfnNativeTilePreferenceBceDescriptorV1* descriptor) {
    NfnNativeTilePreferenceBceDescriptorV1 normalized{};
    if (!normalize_preference_bce_descriptor(descriptor, &normalized, false)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_preference_bce_loss_float32_forward_v1(
        normalized, as_stream(normalized.cuda_stream));
    return launch_status();
}

int nfn_native_tile_preference_bce_loss_float32_backward_v1(
    const NfnNativeTilePreferenceBceDescriptorV1* descriptor) {
    NfnNativeTilePreferenceBceDescriptorV1 normalized{};
    if (!normalize_preference_bce_descriptor(descriptor, &normalized, true)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_preference_bce_loss_float32_backward_v1(
        normalized, as_stream(normalized.cuda_stream));
    return launch_status();
}

int nfn_native_tile_token_logp_entropy_i32_float32_forward_v1(
    const NfnNativeTileTokenLogpEntropyDescriptorV1* descriptor) {
    NfnNativeTileTokenLogpEntropyDescriptorV1 normalized{};
    if (!normalize_token_logp_entropy_descriptor(
            descriptor, &normalized, false)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_token_logp_entropy_i32_float32_forward_v1(
        normalized, as_stream(normalized.cuda_stream));
    return launch_status();
}

int nfn_native_tile_token_logp_entropy_i32_float32_backward_v1(
    const NfnNativeTileTokenLogpEntropyDescriptorV1* descriptor) {
    NfnNativeTileTokenLogpEntropyDescriptorV1 normalized{};
    if (!normalize_token_logp_entropy_descriptor(
            descriptor, &normalized, true)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_token_logp_entropy_i32_float32_backward_v1(
        normalized, as_stream(normalized.cuda_stream));
    return launch_status();
}

int nfn_native_tile_masked_ppo_loss_float32_forward_v1(
    const NfnNativeTileMaskedPpoLossDescriptorV1* descriptor) {
    NfnNativeTileMaskedPpoLossDescriptorV1 normalized{};
    if (!normalize_masked_ppo_loss_descriptor(descriptor, &normalized, false)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_masked_ppo_loss_float32_forward_v1(
        normalized, as_stream(normalized.cuda_stream));
    return launch_status();
}

int nfn_native_tile_masked_ppo_loss_float32_backward_v1(
    const NfnNativeTileMaskedPpoLossDescriptorV1* descriptor) {
    NfnNativeTileMaskedPpoLossDescriptorV1 normalized{};
    if (!normalize_masked_ppo_loss_descriptor(descriptor, &normalized, true)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_masked_ppo_loss_float32_backward_v1(
        normalized, as_stream(normalized.cuda_stream));
    return launch_status();
}

int nfn_native_tile_token_embedding_backward_weight_i32_float32(
    const std::int32_t* token_ids,
    const float* grad_output,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t vocab_size,
    std::int64_t embedding_dim,
    void* cuda_stream) {
    std::int64_t count = 0;
    if (token_ids == nullptr || grad_output == nullptr || grad_weight == nullptr ||
        rows <= 0 || vocab_size <= 0 || embedding_dim <= 0 ||
        !checked_positive_product(vocab_size, embedding_dim, &count)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_token_embedding_backward_weight_i32_float32(
        token_ids, grad_output, grad_weight, rows, vocab_size, embedding_dim,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (parameter_bf16 == nullptr || gradient == nullptr || exp_avg == nullptr ||
        exp_avg_sq == nullptr || count <= 0 || !std::isfinite(learning_rate) ||
        learning_rate < 0.0f || !std::isfinite(beta1) || beta1 < 0.0f ||
        beta1 >= 1.0f || !std::isfinite(beta2) || beta2 < 0.0f || beta2 >= 1.0f ||
        !std::isfinite(eps) || !(eps > 0.0f) || !std::isfinite(weight_decay) ||
        weight_decay < 0.0f || step <= 0 || !std::isfinite(gradient_scale) ||
        gradient_scale < 0.0f) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_glimmer_adamw_bf16_float32_v1(
        parameter_bf16, gradient, exp_avg, exp_avg_sq, count, learning_rate,
        beta1, beta2, eps, weight_decay, step, gradient_scale,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_turboquant_attention_forward_v1(
    const NfnNativeTileTurboQuantAttentionDescriptorV1* descriptor) {
    NfnNativeTileTurboQuantAttentionDescriptorV1 normalized{};
    if (!normalize_turboquant_attention_descriptor(descriptor, &normalized)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    return neuralfn::tile_cuda::launch_turboquant_attention_forward_v1(
        normalized, as_stream(normalized.cuda_stream));
}

void nfn_native_tile_turboquant_attention_stats_reset() {
    neuralfn::tile_cuda::reset_turboquant_attention_launch_stats();
}

std::int64_t nfn_native_tile_turboquant_attention_launch_count() {
    return neuralfn::tile_cuda::turboquant_attention_launch_count();
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

std::int64_t nfn_native_tile_attention_backward_tk_batch_cap() {
    return neuralfn::tile_cuda::attention_backward_tk_batch_cap();
}

std::int64_t nfn_native_tile_attention_backward_tk_chunk_batch_total() {
    return neuralfn::tile_cuda::attention_backward_tk_chunk_batch_total();
}

std::int64_t nfn_native_tile_attention_backward_tk_chunk_batch_max() {
    return neuralfn::tile_cuda::attention_backward_tk_chunk_batch_max();
}

std::int64_t nfn_native_tile_attention_backward_tk_chunk_batch_min() {
    return neuralfn::tile_cuda::attention_backward_tk_chunk_batch_min();
}

std::int64_t nfn_native_tile_attention_backward_tk_chunk_batch_last() {
    return neuralfn::tile_cuda::attention_backward_tk_chunk_batch_last();
}

int nfn_native_tile_attention_backward_tk_block_size() {
    return neuralfn::tile_cuda::attention_backward_tk_block_size();
}

std::int64_t nfn_native_tile_attention_backward_dprep_default_warps_per_block() {
    return neuralfn::tile_cuda::tk_packed_attention_dprep_default_warps_per_block();
}

std::int64_t nfn_native_tile_sm120_memory_block_size() {
    return neuralfn::tile_cuda::tk_sm120_memory_block_size();
}

std::int64_t nfn_native_tile_sm120_layernorm_bwd_blocks_per_sm() {
    return neuralfn::tile_cuda::tk_sm120_layernorm_bwd_blocks_per_sm();
}

std::int64_t nfn_native_tile_attention_backward_float_hd64_dprep_launch_count() {
    return neuralfn::tile_cuda::attention_backward_float_hd64_dprep_launch_count();
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

std::int64_t nfn_native_tile_token_cross_entropy_bf16_threads_per_row() {
    return neuralfn::tile_cuda::token_cross_entropy_bf16_threads_per_row();
}

std::int64_t nfn_native_tile_lm_head_true_fused_mat_tile() {
    return neuralfn::tile_cuda::lm_head_true_fused_mat_tile();
}

std::int64_t nfn_native_tile_lm_head_true_fused_required_threads() {
    return neuralfn::tile_cuda::lm_head_true_fused_required_threads();
}

std::int64_t nfn_native_tile_lm_head_prob_only_target_correction_threads() {
    return neuralfn::tile_cuda::lm_head_prob_only_target_correction_threads();
}

void nfn_native_tile_lm_head_classifier_stats_reset() {
    neuralfn::tile_cuda::reset_lm_head_classifier_chunk_stats();
    reset_lm_head_cooperative_sequence_stats();
}

std::int64_t nfn_native_tile_lm_head_classifier_chunk_launch_count() {
    return neuralfn::tile_cuda::lm_head_classifier_chunk_launch_count();
}

std::int64_t nfn_native_tile_lm_head_classifier_last_rows() {
    return neuralfn::tile_cuda::lm_head_classifier_last_rows();
}

std::int64_t nfn_native_tile_lm_head_classifier_last_vocab() {
    return neuralfn::tile_cuda::lm_head_classifier_last_vocab();
}

std::int64_t nfn_native_tile_lm_head_classifier_last_row_stride() {
    return neuralfn::tile_cuda::lm_head_classifier_last_row_stride();
}

std::int64_t nfn_native_tile_lm_head_classifier_loss_bin_launch_count() {
    return neuralfn::tile_cuda::lm_head_classifier_loss_bin_launch_count();
}

std::int64_t nfn_native_tile_lm_head_classifier_true_fused_launch_count() {
    return neuralfn::tile_cuda::lm_head_classifier_true_fused_launch_count();
}

std::int64_t nfn_native_tile_lm_head_true_fused_ce_cycles() {
    return neuralfn::tile_cuda::lm_head_true_fused_ce_cycles();
}

std::int64_t nfn_native_tile_lm_head_true_fused_dhidden_cycles() {
    return neuralfn::tile_cuda::lm_head_true_fused_dhidden_cycles();
}

std::int64_t nfn_native_tile_lm_head_true_fused_dweight_cycles() {
    return neuralfn::tile_cuda::lm_head_true_fused_dweight_cycles();
}

std::int64_t nfn_native_tile_lm_head_true_fused_ce_blocks() {
    return neuralfn::tile_cuda::lm_head_true_fused_ce_blocks();
}

std::int64_t nfn_native_tile_lm_head_true_fused_dhidden_blocks() {
    return neuralfn::tile_cuda::lm_head_true_fused_dhidden_blocks();
}

std::int64_t nfn_native_tile_lm_head_true_fused_dweight_blocks() {
    return neuralfn::tile_cuda::lm_head_true_fused_dweight_blocks();
}

std::int64_t nfn_native_tile_lm_head_cooperative_sequence_launch_count() {
    return g_lm_head_cooperative_sequence_launch_count.load(std::memory_order_relaxed);
}

std::int64_t nfn_native_tile_lm_head_cooperative_sequence_ce_launch_count() {
    return g_lm_head_cooperative_sequence_ce_launch_count.load(std::memory_order_relaxed);
}

std::int64_t nfn_native_tile_lm_head_cooperative_sequence_dhidden_launch_count() {
    return g_lm_head_cooperative_sequence_dhidden_launch_count.load(std::memory_order_relaxed);
}

std::int64_t nfn_native_tile_lm_head_cooperative_sequence_dweight_launch_count() {
    return g_lm_head_cooperative_sequence_dweight_launch_count.load(std::memory_order_relaxed);
}

std::int64_t nfn_native_tile_lm_head_cooperative_sequence_concurrent_count() {
    return g_lm_head_cooperative_sequence_concurrent_count.load(std::memory_order_relaxed);
}

std::int64_t nfn_native_tile_lm_head_cooperative_sequence_legacy_count() {
    return g_lm_head_cooperative_sequence_legacy_count.load(std::memory_order_relaxed);
}

std::int64_t nfn_native_tile_lm_head_cooperative_sequence_loss_bin_count() {
    return g_lm_head_cooperative_sequence_loss_bin_count.load(std::memory_order_relaxed);
}

std::int64_t nfn_native_tile_lm_head_fused_graph_capture_attempt_count() {
    return g_lm_head_fused_graph_capture_attempt_count.load(std::memory_order_relaxed);
}

std::int64_t nfn_native_tile_lm_head_fused_graph_capture_success_count() {
    return g_lm_head_fused_graph_capture_success_count.load(std::memory_order_relaxed);
}

std::int64_t nfn_native_tile_lm_head_fused_graph_upload_success_count() {
    return g_lm_head_fused_graph_upload_success_count.load(std::memory_order_relaxed);
}

std::int64_t nfn_native_tile_lm_head_fused_graph_upload_failure_count() {
    return g_lm_head_fused_graph_upload_failure_count.load(std::memory_order_relaxed);
}

std::int64_t nfn_native_tile_lm_head_fused_graph_cache_hit_count() {
    return g_lm_head_fused_graph_cache_hit_count.load(std::memory_order_relaxed) +
        lm_head_graph_local_stats().cache_hit_count;
}

std::int64_t nfn_native_tile_lm_head_fused_graph_thread_cache_hit_count() {
    return g_lm_head_fused_graph_thread_cache_hit_count.load(std::memory_order_relaxed) +
        lm_head_graph_local_stats().thread_cache_hit_count;
}

std::int64_t nfn_native_tile_lm_head_fused_graph_cache_entry_count() {
    std::lock_guard<std::mutex> lock(g_lm_head_backward_graph_mutex);
    return static_cast<std::int64_t>(g_lm_head_backward_graph_cache.size());
}

std::int64_t nfn_native_tile_lm_head_fused_graph_replay_count() {
    return g_lm_head_fused_graph_replay_count.load(std::memory_order_relaxed) +
        lm_head_graph_local_stats().replay_count;
}

std::int64_t nfn_native_tile_lm_head_fused_graph_replay_success_count() {
    return g_lm_head_fused_graph_replay_success_count.load(std::memory_order_relaxed) +
        lm_head_graph_local_stats().replay_success_count;
}

std::int64_t nfn_native_tile_lm_head_fused_graph_fallback_count() {
    return g_lm_head_fused_graph_fallback_count.load(std::memory_order_relaxed) +
        lm_head_graph_local_stats().fallback_count;
}

std::int64_t nfn_native_tile_lm_head_graph_body_cublaslt_dhidden_launch_count() {
    return g_lm_head_graph_body_cublaslt_dhidden_launch_count.load(std::memory_order_relaxed);
}

std::int64_t nfn_native_tile_lm_head_graph_body_cublaslt_dweight_launch_count() {
    return g_lm_head_graph_body_cublaslt_dweight_launch_count.load(std::memory_order_relaxed);
}

std::int64_t nfn_native_tile_lm_head_graph_body_tile_dhidden_fallback_count() {
    return g_lm_head_graph_body_tile_dhidden_fallback_count.load(std::memory_order_relaxed);
}

std::int64_t nfn_native_tile_lm_head_graph_body_tile_dweight_fallback_count() {
    return g_lm_head_graph_body_tile_dweight_fallback_count.load(std::memory_order_relaxed);
}

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
    void* cuda_stream) {
    (void)hidden_float;
    (void)token_weight_float;
    (void)cuda_stream;
    const bool no_loss = (flags & kLmHeadCooperativeFlagNoLoss) != 0;
    if (logits_bf16 == nullptr ||
        targets_u16 == nullptr ||
        (!no_loss && row_losses == nullptr) ||
        hidden_bf16 == nullptr ||
        token_weight_bf16 == nullptr ||
        grad_hidden == nullptr ||
        grad_weight == nullptr ||
        rows <= 0 ||
        hidden_dim <= 0 ||
        vocab <= 0 ||
        row_stride < vocab) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const LmHeadBackwardGraphKey key{
        logits_bf16,
        targets_u16,
        row_losses,
        hidden_bf16,
        token_weight_bf16,
        grad_hidden,
        grad_weight,
        rows,
        hidden_dim,
        vocab,
        row_stride,
        lm_head_cooperative_loss_bin_count_from_flags(flags, rows),
        loss_scale,
        dweight_beta,
        flags,
    };
    return prewarm_lm_head_classifier_backward_graph_bf16_u16(key);
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

std::int64_t nfn_native_tile_trainer_linear_bf16_gemm_fast16bf_request_count() {
    return neuralfn::tile_cuda::trainer_linear_bf16_gemm_fast16bf_request_count();
}

std::int64_t nfn_native_tile_trainer_linear_tk_gemm_count() {
    return neuralfn::tile_cuda::trainer_linear_tk_gemm_count();
}

std::int64_t nfn_native_tile_trainer_linear_tk_float_out_gemm_count() {
    return neuralfn::tile_cuda::trainer_linear_tk_float_out_gemm_count();
}

std::int64_t nfn_native_tile_trainer_linear_tk_dweight_gemm_count() {
    return neuralfn::tile_cuda::trainer_linear_tk_dweight_gemm_count();
}

std::int64_t nfn_native_tile_trainer_linear_tk_dgelu_dinput_gemm_count() {
    return neuralfn::tile_cuda::trainer_linear_tk_dgelu_dinput_gemm_count();
}

int nfn_native_tile_trainer_linear_tk_sm120_k_tile() {
    return neuralfn::tile_cuda::trainer_linear_tk_sm120_k_tile();
}

int nfn_native_tile_trainer_linear_tk_sm120_grad_k_tile() {
    return neuralfn::tile_cuda::trainer_linear_tk_sm120_grad_k_tile();
}

int nfn_native_tile_trainer_linear_tk_sm120_super_m() {
    return neuralfn::tile_cuda::trainer_linear_tk_sm120_super_m();
}

int nfn_native_tile_trainer_linear_tk_sm120_dinput_super_m() {
    return neuralfn::tile_cuda::trainer_linear_tk_sm120_dinput_super_m();
}

int nfn_native_tile_trainer_linear_tk_sm120_dweight_super_m() {
    return neuralfn::tile_cuda::trainer_linear_tk_sm120_dweight_super_m();
}

int nfn_native_tile_trainer_linear_tk_sm120_huge_n_k_tile() {
    return neuralfn::tile_cuda::trainer_linear_tk_sm120_huge_n_k_tile();
}

int nfn_native_tile_trainer_linear_tk_sm120_fast_dgelu_enabled() {
    return neuralfn::tile_cuda::trainer_linear_tk_sm120_fast_dgelu_enabled();
}

int nfn_native_tile_trainer_linear_tk_sm120_approx_dgelu_tanh_enabled() {
    return neuralfn::tile_cuda::trainer_linear_tk_sm120_approx_dgelu_tanh_enabled();
}

std::int64_t nfn_native_tile_trainer_linear_cublaslt_gemm_count() {
    return neuralfn::tile_cuda::trainer_linear_cublaslt_gemm_count();
}

std::int64_t nfn_native_tile_trainer_linear_cublaslt_bgrad_gemm_count() {
    return neuralfn::tile_cuda::trainer_linear_cublaslt_bgrad_gemm_count();
}

std::int64_t nfn_native_tile_trainer_linear_cublaslt_bgrad_direct_write_count() {
    return neuralfn::tile_cuda::trainer_linear_cublaslt_bgrad_direct_write_count();
}

std::int64_t nfn_native_tile_trainer_linear_cublaslt_bgrad_accumulate_count() {
    return neuralfn::tile_cuda::trainer_linear_cublaslt_bgrad_accumulate_count();
}

int nfn_native_tile_linear_backward_bias_threads_per_block() {
    return neuralfn::tile_cuda::linear_backward_bias_threads_per_block();
}

std::int64_t nfn_native_tile_trainer_linear_sgemm_count() {
    return neuralfn::tile_cuda::trainer_linear_sgemm_count();
}

std::int64_t nfn_native_tile_trainer_bf16_to_f32_vec4_count() {
    return neuralfn::tile_cuda::trainer_bf16_to_f32_vec4_count();
}

std::int64_t nfn_native_tile_trainer_linear_bf16_a_pack_count() {
    return neuralfn::tile_cuda::trainer_linear_bf16_a_pack_count();
}

std::int64_t nfn_native_tile_trainer_linear_bf16_cached_a_pack_count() {
    return neuralfn::tile_cuda::trainer_linear_bf16_cached_a_pack_count();
}

std::int64_t nfn_native_tile_trainer_linear_bf16_cached_b_pack_count() {
    return neuralfn::tile_cuda::trainer_linear_bf16_cached_b_pack_count();
}

std::int64_t nfn_native_tile_trainer_linear_bf16_transient_a_pack_count() {
    return neuralfn::tile_cuda::trainer_linear_bf16_transient_a_pack_count();
}

std::int64_t nfn_native_tile_trainer_linear_bf16_transient_b_pack_count() {
    return neuralfn::tile_cuda::trainer_linear_bf16_transient_b_pack_count();
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

int nfn_native_tile_trainer_linear_cublaslt_grouped_layout_probe_status() {
    return neuralfn::tile_cuda::trainer_linear_cublaslt_grouped_layout_probe_status();
}

int nfn_native_tile_trainer_linear_cublaslt_grouped_matmul_probe_status() {
    return neuralfn::tile_cuda::trainer_linear_cublaslt_grouped_matmul_probe_status();
}

int nfn_native_tile_trainer_linear_cublas_grouped_bf16_gemm_probe_status() {
    return neuralfn::tile_cuda::trainer_linear_cublas_grouped_bf16_gemm_probe_status();
}

int nfn_native_tile_trainer_linear_cublas_prewarm(void* stream) {
    return neuralfn::tile_cuda::trainer_linear_cublas_prewarm(static_cast<cudaStream_t>(stream)) ? 1 : 0;
}

int nfn_native_tile_trainer_linear_bf16_workspace_prewarm(
    std::int64_t a_elements,
    std::int64_t b_elements,
    std::int64_t c_elements) {
    return neuralfn::tile_cuda::trainer_linear_bf16_workspace_prewarm(
               a_elements,
               b_elements,
               c_elements)
        ? 1
        : 0;
}

int nfn_native_tile_trainer_linear_cublaslt_prewarm_bf16_plan(
    int m,
    int n,
    int k,
    int op_a,
    int op_b,
    int lda,
    int ldb,
    int ldc,
    int bgrad_epilogue) {
    return neuralfn::tile_cuda::trainer_linear_cublaslt_prewarm_bf16_plan(
               m,
               n,
               k,
               op_a,
               op_b,
               lda,
               ldb,
               ldc,
               bgrad_epilogue != 0)
        ? 1
        : 0;
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

std::int64_t nfn_native_tile_trainer_linear_cublaslt_plan_cache_count() {
    return neuralfn::tile_cuda::trainer_linear_cublaslt_plan_cache_count();
}

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
    int* epilogue) {
    return neuralfn::tile_cuda::trainer_linear_cublaslt_plan_cache_entry(
        index,
        m,
        n,
        k,
        op_a,
        op_b,
        selected_heuristic,
        returned_heuristics,
        workspace_bytes,
        epilogue);
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

int nfn_native_tile_tanh_float32(
    const float* x,
    float* out,
    std::int64_t n,
    void* cuda_stream) {
    if (n <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_unary_float32(x, out, n, 4, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_tanh_backward_float32(
    const float* grad_out,
    const float* tanh_out,
    float* grad_x,
    std::int64_t n,
    void* cuda_stream) {
    if (n <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_tanh_backward_float32(grad_out, tanh_out, grad_x, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_add_float32(
    const float* lhs,
    const float* rhs,
    float* out,
    std::int64_t n,
    void* cuda_stream) {
    if (n <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_binary_float32(lhs, rhs, out, n, 0, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_vector_binary_float32(
    const float* lhs,
    const float* rhs,
    const float* scale0,
    const float* scale1,
    float* out,
    std::int64_t n,
    std::int64_t dim,
    std::int64_t op,
    void* cuda_stream) {
    if (lhs == nullptr || rhs == nullptr || scale0 == nullptr || out == nullptr || n <= 0 || dim <= 0 ||
        (op != 0 && op != 1 && op != 2) || (op == 1 && scale1 == nullptr)) {
        return 1;
    }
    neuralfn::tile_cuda::launch_vector_binary_float32(
        lhs, rhs, scale0, scale1, out, n, dim, static_cast<int>(op), as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (beta_logit == nullptr || input == nullptr || attention_proj == nullptr ||
        residual1 == nullptr || ffn_out == nullptr || grad_second == nullptr ||
        grad_first == nullptr || grad_beta_logit == nullptr || rows <= 0 ||
        model_dim <= 0 || !std::isfinite(scale)) {
        return 1;
    }
    neuralfn::tile_cuda::launch_mhc_beta_gradient_float32(
        beta_logit, input, attention_proj, residual1, ffn_out, grad_second, grad_first,
        grad_beta_logit, rows, model_dim, scale, as_stream(cuda_stream));
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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_fill_many_values_mixed_float32_bf16_bits(
        float_buffers,
        float_elements,
        float_values,
        float_buffer_count,
        float_max_elements,
        bf16_buffers,
        bf16_elements,
        bf16_values,
        bf16_buffer_count,
        bf16_max_elements,
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

int nfn_native_tile_seeded_normal_float32(
    float* values,
    std::uint16_t* shadow_bf16_bits,
    std::int64_t n,
    std::uint64_t seed,
    std::uint64_t offset,
    float stddev,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_seeded_normal_float32(
        values, shadow_bf16_bits, n, seed, offset, stddev, as_stream(cuda_stream));
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

int nfn_native_tile_uint32_to_int64(
    const std::uint32_t* source,
    std::int64_t* dest,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_uint32_to_int64(source, dest, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_uint8_to_int64(
    const std::uint8_t* source,
    std::int64_t* dest,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_uint8_to_int64(source, dest, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_diffusion_mask_u16_int64(
    const std::uint16_t* source_tokens,
    std::uint16_t* masked_tokens,
    std::int64_t* targets,
    std::int64_t rows,
    std::int64_t seq_len,
    std::int64_t vocab,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_diffusion_mask_u16_int64(
        source_tokens, masked_tokens, targets, rows, seq_len, vocab, as_stream(cuda_stream));
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

int nfn_native_tile_float32_to_nvfp4_packed(
    const float* source,
    std::uint8_t* packed,
    std::uint8_t* block_scales_e4m3,
    float tensor_scale,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_float32_to_nvfp4_packed(
        source, packed, block_scales_e4m3, tensor_scale, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_nvfp4_packed_to_float32(
    const std::uint8_t* packed,
    const std::uint8_t* block_scales_e4m3,
    float tensor_scale,
    float* dest,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_nvfp4_packed_to_float32(
        packed, block_scales_e4m3, tensor_scale, dest, n, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_nvfp4_input_weight_bf16_float32(
        x_nvfp4_packed,
        x_block_scales_e4m3,
        x_tensor_scale,
        weight_bf16_bits,
        bias,
        out,
        rows,
        input_dim,
        output_dim,
        has_bias,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_nvfp4_input_weight_bf16_output_float32(
        x_nvfp4_packed,
        x_block_scales_e4m3,
        x_tensor_scale,
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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_weight_accumulate_nvfp4_input_float32_beta(
        x_nvfp4_packed,
        x_block_scales_e4m3,
        x_tensor_scale,
        grad_out,
        grad_weight,
        rows,
        input_dim,
        output_dim,
        beta,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_weight_accumulate_nvfp4_input_bf16_grad_float32_beta(
        x_nvfp4_packed,
        x_block_scales_e4m3,
        x_tensor_scale,
        grad_out_bf16_bits,
        grad_weight,
        rows,
        input_dim,
        output_dim,
        beta,
        as_stream(cuda_stream));
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

int nfn_native_tile_init_gpt2_token_weight_fast_with_bf16_shadow_padded_float32(
    float* values,
    std::uint16_t* shadow_bf16_bits,
    std::int64_t public_n,
    std::int64_t total_n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_init_gpt2_token_weight_fast_with_bf16_shadow_padded_float32(
        values,
        shadow_bf16_bits,
        public_n,
        total_n,
        as_stream(cuda_stream));
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

int nfn_native_tile_optimizer_tile_size() {
#ifndef NFN_TILE_CUDA_OPTIMIZER_TILE_SIZE
    return 1024;
#else
    return NFN_TILE_CUDA_OPTIMIZER_TILE_SIZE;
#endif
}

int nfn_native_tile_sum_partials_float32(
    const float* values,
    float* partials,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_sum_partials_float32(values, partials, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_sum_accumulate_float32(
    const float* values,
    float* total,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_sum_accumulate_float32(values, total, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_extract_diagonal_float32(
    const float* matrix,
    float* diagonal,
    std::int64_t dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_extract_diagonal_float32(
        matrix, diagonal, dim, as_stream(cuda_stream));
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

int nfn_native_tile_repeat_kv_float32(
    const float* input,
    float* output,
    std::int64_t batch,
    std::int64_t kv_heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    std::int64_t repeats,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_repeat_kv_float32(
        input, output, batch, kv_heads, seq_len, head_dim, repeats, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_repeat_kv_backward_float32(
    const float* grad_output,
    float* grad_input,
    std::int64_t batch,
    std::int64_t kv_heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    std::int64_t repeats,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_repeat_kv_backward_float32(
        grad_output, grad_input, batch, kv_heads, seq_len, head_dim, repeats, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (batch <= 0 || seq_len <= 0 || model_dim <= 0 || patch_size <= 0 ||
        stride <= 0 || out_len <= 0 || vocab_size <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_byte_patch_embed_float32(
        tokens,
        embedding,
        proj,
        out,
        batch,
        seq_len,
        model_dim,
        patch_size,
        stride,
        out_len,
        vocab_size,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_byte_patch_merge_float32(
    const float* x,
    float* out,
    std::int64_t batch,
    std::int64_t source_len,
    std::int64_t target_len,
    std::int64_t dim,
    void* cuda_stream) {
    if (batch <= 0 || source_len <= 0 || target_len < 0 || dim <= 0) {
        return 1;
    }
    if (target_len == 0) {
        return 0;
    }
    neuralfn::tile_cuda::launch_byte_patch_merge_float32(
        x, out, batch, source_len, target_len, dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_byte_patch_merge_backward_float32(
    const float* grad_out,
    float* grad_x,
    std::int64_t batch,
    std::int64_t source_len,
    std::int64_t target_len,
    std::int64_t dim,
    void* cuda_stream) {
    if (batch <= 0 || source_len <= 0 || target_len < 0 || dim <= 0) {
        return 1;
    }
    if (target_len == 0) {
        return 0;
    }
    neuralfn::tile_cuda::launch_byte_patch_merge_backward_float32(
        grad_out, grad_x, batch, source_len, target_len, dim, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (batch <= 0 || seq_len <= 0 || model_dim <= 0 || patch_size <= 0 ||
        stride <= 0 || out_len <= 0 || vocab_size <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_byte_patch_embed_backward_float32(
        tokens,
        embedding,
        proj,
        grad_out,
        grad_embedding,
        grad_proj,
        batch,
        seq_len,
        model_dim,
        patch_size,
        stride,
        out_len,
        vocab_size,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_causal_chunk_state_float32(
    const float* hidden,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t dim,
    std::int64_t chunk_size,
    std::int64_t chunks,
    std::int64_t mode,
    void* cuda_stream) {
    if (batch <= 0 || seq_len <= 0 || dim <= 0 || chunk_size <= 0 || chunks <= 0 ||
        (mode != 0 && mode != 1)) {
        return 1;
    }
    neuralfn::tile_cuda::launch_causal_chunk_state_float32(
        hidden, out, batch, seq_len, dim, chunk_size, chunks, static_cast<int>(mode), as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_causal_chunk_state_backward_float32(
    const float* grad_out,
    float* grad_hidden,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t dim,
    std::int64_t chunk_size,
    std::int64_t chunks,
    std::int64_t mode,
    void* cuda_stream) {
    if (batch <= 0 || seq_len <= 0 || dim <= 0 || chunk_size <= 0 || chunks <= 0 ||
        (mode != 0 && mode != 1)) {
        return 1;
    }
    neuralfn::tile_cuda::launch_causal_chunk_state_backward_float32(
        grad_out, grad_hidden, batch, seq_len, dim, chunk_size, chunks, static_cast<int>(mode), as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_topk_route_float32(
    const float* logits,
    float* weights,
    std::int64_t* indices,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t top_k,
    void* cuda_stream) {
    if (rows <= 0 || experts <= 0 || top_k <= 0 || top_k > experts || top_k > 64) {
        return 1;
    }
    neuralfn::tile_cuda::launch_topk_route_float32(
        logits, weights, indices, rows, experts, top_k, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_topk_route_sqrt_softplus_float32(
    const float* logits,
    float* weights,
    std::int64_t* indices,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t top_k,
    void* cuda_stream) {
    if (rows <= 0 || experts <= 0 || top_k <= 0 || top_k > experts || top_k > 64) {
        return 1;
    }
    neuralfn::tile_cuda::launch_topk_route_sqrt_softplus_float32(
        logits, weights, indices, rows, experts, top_k, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_topk_route_backward_float32(
    const float* weights,
    const std::int64_t* indices,
    const float* grad_weights,
    float* grad_logits,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t top_k,
    float route_scale,
    void* cuda_stream) {
    if (rows <= 0 || experts <= 0 || top_k <= 0 || top_k > experts || top_k > 64) {
        return 1;
    }
    neuralfn::tile_cuda::launch_topk_route_backward_float32(
        weights, indices, grad_weights, grad_logits, rows, experts, top_k, route_scale, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_semantic_shared_topk_route_float32(
    const float* logits,
    float* weights,
    std::int64_t* indices,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t shared_experts,
    std::int64_t top_k,
    void* cuda_stream) {
    const std::int64_t route_width = shared_experts + top_k;
    if (rows <= 0 || experts <= 0 || shared_experts <= 0 || top_k <= 0 ||
        shared_experts >= experts || top_k > experts - shared_experts || route_width > 64) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_shared_topk_route_float32(
        logits, weights, indices, rows, experts, shared_experts, top_k, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    const std::int64_t route_width = shared_experts + top_k;
    if (logits == nullptr || semantic_target_matrix == nullptr || weights == nullptr ||
        indices == nullptr || rows <= 0 || experts <= 0 || semantic_vocab_dims <= 0 ||
        shared_experts <= 0 || shared_experts + semantic_vocab_dims > experts ||
        top_k <= 0 || top_k > experts - shared_experts || route_width > 64) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_shared_forced_topk_route_float32(
        logits, semantic_target_matrix, weights, indices, rows, experts, semantic_vocab_dims,
        shared_experts, top_k, ignore_index, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    const std::int64_t route_width = shared_experts + top_k;
    if (rows <= 0 || experts <= 0 || shared_experts <= 0 || top_k <= 0 ||
        shared_experts >= experts || top_k > experts - shared_experts || route_width > 64) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_shared_topk_route_backward_float32(
        weights, indices, grad_weights, grad_logits, rows, experts, shared_experts, top_k, route_scale,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (rows <= 0 || experts <= 0 || top_k <= 0 || top_k > experts || top_k > 64) {
        return 1;
    }
    neuralfn::tile_cuda::launch_topk_route_sqrt_softplus_backward_float32(
        logits, weights, indices, grad_weights, grad_logits, rows, experts,
        top_k, route_scale, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_broadcast_expert_routes_float32(
    const float* weights,
    const std::int64_t* indices,
    float* out_weights,
    std::int64_t* out_indices,
    std::int64_t batch,
    std::int64_t route_seq,
    std::int64_t seq_len,
    std::int64_t route_width,
    void* cuda_stream) {
    if (batch <= 0 || route_seq <= 0 || seq_len < 0 || route_width <= 0 ||
        (route_seq != 1 && route_seq != seq_len)) {
        return 1;
    }
    neuralfn::tile_cuda::launch_broadcast_expert_routes_float32(
        weights, indices, out_weights, out_indices, batch, route_seq, seq_len, route_width, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (batch <= 0 || chunks <= 0 || seq_len < 0 || route_width <= 0 || chunk_size <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_broadcast_chunk_routes_float32(
        weights, indices, out_weights, out_indices, batch, chunks, seq_len, route_width, chunk_size, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (batch <= 0 || seq_len <= 0 || chunks <= 0 || route_width <= 0 || chunk_size <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_compact_chunk_routes_float32_int64(
        weights, indices, chunk_weights, chunk_indices, batch, seq_len, chunks, route_width, chunk_size,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_aggregate_chunk_route_gradients_float32(
    const float* grad_weights,
    float* aggregated_grad_weights,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t route_width,
    std::int64_t chunk_size,
    void* cuda_stream) {
    if (batch <= 0 || seq_len <= 0 || route_width <= 0 || chunk_size <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_aggregate_chunk_route_gradients_float32(
        grad_weights, aggregated_grad_weights, batch, seq_len, route_width, chunk_size, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (rows <= 0 || seq_len <= 0 || experts <= 1 || semantic_vocab_dims <= 0 || route_chunk_size <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_route_distillation_backward_float32(
        route_logits, semantic_targets, semantic_target_valid, grad_route_logits, loss_items,
        rows, seq_len, experts, semantic_vocab_dims, shared_experts, route_chunk_size,
        distill_weight, teacher_target, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (rows <= 0 || seq_len <= 0 || experts <= 1 || semantic_vocab_dims <= 0 || route_chunk_size <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_target_topic_distillation_backward_float32(
        route_logits, target_topic_logits, grad_route_logits, loss_items,
        rows, seq_len, experts, semantic_vocab_dims, shared_experts, route_chunk_size,
        distill_weight, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (route_logits == nullptr || target_topic_logits == nullptr ||
        term_counts == nullptr || term_offsets == nullptr ||
        grad_route_logits == nullptr || loss_items == nullptr ||
        rows <= 0 || seq_len <= 0 || experts <= 1 || semantic_vocab_dims <= 0 ||
        total_terms <= 0 || route_chunk_size <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_target_topic_packed_distillation_backward_float32(
        route_logits, target_topic_logits, term_counts, term_offsets, grad_route_logits,
        loss_items, rows, seq_len, experts, semantic_vocab_dims, total_terms,
        shared_experts, route_chunk_size, distill_weight, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (hash_indices == nullptr || hash_embedding == nullptr || table_gate_logits == nullptr ||
        grad_route_logits == nullptr || grad_hash_embedding == nullptr ||
        grad_table_gate == nullptr || grad_dimension_bias == nullptr ||
        rows <= 0 || experts <= 0 || semantic_vocab_dims <= 0 || tables <= 0 ||
        tables > 32 || buckets <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_hash_table_backward_float32(
        hash_indices, hash_embedding, table_gate_logits, grad_route_logits,
        grad_hash_embedding, grad_table_gate, grad_dimension_bias, rows, experts,
        semantic_vocab_dims, shared_experts, tables, buckets, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (route_logits == nullptr || hash_indices == nullptr || hash_embedding == nullptr ||
        table_gate_logits == nullptr || dimension_bias == nullptr || semantic_targets == nullptr ||
        semantic_target_valid == nullptr || rows <= 0 || experts <= 0 ||
        semantic_vocab_dims <= 0 || tables <= 0 || tables > 32 || buckets <= 0 ||
        top_k <= 0 || top_k > experts || !std::isfinite(target_boost)) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_route_policy_float32(
        route_logits, hash_indices, hash_embedding, table_gate_logits, dimension_bias,
        semantic_targets, semantic_target_valid, rows, experts, semantic_vocab_dims,
        shared_experts, tables, buckets, top_k, target_boost, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (route_logits == nullptr || hash_indices == nullptr || hash_embedding == nullptr ||
        table_gate_logits == nullptr || dimension_bias == nullptr || topic_logits == nullptr ||
        term_counts == nullptr || term_offsets == nullptr || semantic_targets == nullptr ||
        semantic_target_valid == nullptr || rows <= 0 || experts <= 0 ||
        semantic_vocab_dims <= 0 || total_terms <= 0 || tables <= 0 || tables > 32 ||
        buckets <= 0 || top_k <= 0 || top_k > experts || !std::isfinite(target_boost)) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_route_policy_packed_topic_float32(
        route_logits, hash_indices, hash_embedding, table_gate_logits, dimension_bias,
        topic_logits, term_counts, term_offsets, semantic_targets, semantic_target_valid,
        rows, experts, semantic_vocab_dims, total_terms, shared_experts, tables, buckets,
        top_k, target_boost, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (route_logits == nullptr || hash_indices == nullptr || hash_embedding == nullptr ||
        table_gate_logits == nullptr || dimension_bias == nullptr || topic_logits == nullptr ||
        term_counts == nullptr || term_offsets == nullptr || semantic_target_matrix == nullptr ||
        rows <= 0 || experts <= 0 || semantic_vocab_dims <= 0 || total_terms <= 0 ||
        tables <= 0 || tables > 32 || buckets <= 0 || top_k <= 0 || top_k > experts ||
        !std::isfinite(target_boost)) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_route_policy_packed_topic_matrix_float32(
        route_logits, hash_indices, hash_embedding, table_gate_logits, dimension_bias,
        topic_logits, term_counts, term_offsets, semantic_target_matrix, rows, experts,
        semantic_vocab_dims, total_terms, shared_experts, tables, buckets, top_k,
        target_boost, ignore_index, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_semantic_vec_from_packed_topic_float32(
    const float* topic_logits,
    const std::int64_t* term_counts,
    const std::int64_t* term_offsets,
    float* semantic_vec,
    std::int64_t rows,
    std::int64_t semantic_vocab_dims,
    std::int64_t total_terms,
    void* cuda_stream) {
    if (topic_logits == nullptr || term_counts == nullptr || term_offsets == nullptr ||
        semantic_vec == nullptr || rows <= 0 || semantic_vocab_dims <= 0 ||
        total_terms <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_vec_from_packed_topic_float32(
        topic_logits, term_counts, term_offsets, semantic_vec, rows,
        semantic_vocab_dims, total_terms, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_semantic_packed_topic_to_padded_float32(
    const float* packed_logits,
    const std::int64_t* term_counts,
    const std::int64_t* term_offsets,
    float* padded_logits,
    std::int64_t rows,
    std::int64_t semantic_vocab_dims,
    std::int64_t total_terms,
    std::int64_t max_terms,
    void* cuda_stream) {
    if (packed_logits == nullptr || term_counts == nullptr || term_offsets == nullptr ||
        padded_logits == nullptr || rows <= 0 || semantic_vocab_dims <= 0 ||
        total_terms <= 0 || max_terms <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_packed_topic_to_padded_float32(
        packed_logits, term_counts, term_offsets, padded_logits, rows,
        semantic_vocab_dims, total_terms, max_terms, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_semantic_signature_scalar_float32(
    const float* sig_logits,
    float* signature_scalar,
    std::int64_t rows,
    std::int64_t buckets,
    void* cuda_stream) {
    if (sig_logits == nullptr || signature_scalar == nullptr ||
        rows <= 0 || buckets <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_signature_scalar_float32(
        sig_logits, signature_scalar, rows, buckets, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_semantic_vec_append_signature_float32(
    const float* topic_vec,
    const float* signature_scalar,
    float* semantic_vec,
    std::int64_t rows,
    std::int64_t topic_dims,
    void* cuda_stream) {
    if (topic_vec == nullptr || signature_scalar == nullptr || semantic_vec == nullptr ||
        rows <= 0 || topic_dims <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_vec_append_signature_float32(
        topic_vec, signature_scalar, semantic_vec, rows, topic_dims, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_semantic_vec_split_signature_grad_float32(
    const float* grad_semantic_vec,
    float* grad_topic_vec,
    float* grad_signature_scalar,
    std::int64_t rows,
    std::int64_t topic_dims,
    void* cuda_stream) {
    if (grad_semantic_vec == nullptr || grad_topic_vec == nullptr ||
        grad_signature_scalar == nullptr || rows <= 0 || topic_dims <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_vec_split_signature_grad_float32(
        grad_semantic_vec, grad_topic_vec, grad_signature_scalar, rows, topic_dims,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_semantic_signature_scalar_backward_float32(
    const float* sig_logits,
    const float* signature_scalar,
    const float* grad_signature_scalar,
    float* grad_sig_logits,
    std::int64_t rows,
    std::int64_t buckets,
    void* cuda_stream) {
    if (sig_logits == nullptr || signature_scalar == nullptr ||
        grad_signature_scalar == nullptr || grad_sig_logits == nullptr ||
        rows <= 0 || buckets <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_signature_scalar_backward_float32(
        sig_logits, signature_scalar, grad_signature_scalar, grad_sig_logits,
        rows, buckets, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (semantic_vec == nullptr || free_weight == nullptr || route_logits == nullptr ||
        rows <= 0 || experts <= 0 || semantic_vocab_dims <= 0 || semantic_vec_dims <= 0 ||
        semantic_shared_experts < 0 || semantic_free_experts <= 0 ||
        weight_stride <= 0 ||
        semantic_shared_experts + semantic_vocab_dims + semantic_free_experts > experts) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_free_expert_projection_float32(
        semantic_vec, free_weight, route_logits, rows, experts, semantic_vocab_dims,
        semantic_vec_dims, semantic_shared_experts, semantic_free_experts, weight_stride,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_semantic_shared_expert_projection_float32(
    const float* semantic_vec,
    const float* shared_weight,
    float* route_logits,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t semantic_vocab_dims,
    std::int64_t semantic_shared_experts,
    std::int64_t weight_stride,
    void* cuda_stream) {
    if (semantic_vec == nullptr || shared_weight == nullptr || route_logits == nullptr ||
        rows <= 0 || experts <= 0 || semantic_vocab_dims <= 0 ||
        semantic_shared_experts <= 0 || weight_stride <= 0 ||
        semantic_shared_experts > experts) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_shared_expert_projection_float32(
        semantic_vec, shared_weight, route_logits, rows, experts, semantic_vocab_dims,
        semantic_shared_experts, weight_stride, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (semantic_vec == nullptr || free_weight == nullptr || grad_route_logits == nullptr ||
        grad_semantic_vec == nullptr || grad_free_weight == nullptr ||
        rows <= 0 || experts <= 0 || semantic_vocab_dims <= 0 || semantic_vec_dims <= 0 ||
        semantic_shared_experts < 0 || semantic_free_experts <= 0 ||
        weight_stride <= 0 ||
        semantic_shared_experts + semantic_vocab_dims + semantic_free_experts > experts) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_free_expert_projection_backward_float32(
        semantic_vec, free_weight, grad_route_logits, grad_semantic_vec, grad_free_weight,
        rows, experts, semantic_vocab_dims, semantic_vec_dims, semantic_shared_experts, semantic_free_experts,
        weight_stride, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (semantic_vec == nullptr || shared_weight == nullptr || grad_route_logits == nullptr ||
        grad_semantic_vec == nullptr || grad_shared_weight == nullptr ||
        rows <= 0 || experts <= 0 || semantic_vocab_dims <= 0 ||
        semantic_shared_experts <= 0 || weight_stride <= 0 ||
        semantic_shared_experts > experts) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_shared_expert_projection_backward_float32(
        semantic_vec, shared_weight, grad_route_logits, grad_semantic_vec, grad_shared_weight,
        rows, experts, semantic_vocab_dims, semantic_shared_experts, weight_stride,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_semantic_router_bias_add_float32(
    float* route_logits,
    const float* shared_logits,
    const float* free_bias,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t semantic_vocab_dims,
    std::int64_t semantic_shared_experts,
    std::int64_t semantic_free_experts,
    void* cuda_stream) {
    if (route_logits == nullptr || rows <= 0 || experts <= 0 || semantic_vocab_dims <= 0 ||
        semantic_shared_experts < 0 || semantic_free_experts < 0 ||
        semantic_shared_experts + semantic_vocab_dims + semantic_free_experts > experts ||
        (semantic_shared_experts > 0 && shared_logits == nullptr) ||
        (semantic_free_experts > 0 && free_bias == nullptr)) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_router_bias_add_float32(
        route_logits, shared_logits, free_bias, rows, experts, semantic_vocab_dims,
        semantic_shared_experts, semantic_free_experts, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_semantic_router_bias_backward_float32(
    const float* grad_route_logits,
    float* grad_shared_logits,
    float* grad_free_bias,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t semantic_vocab_dims,
    std::int64_t semantic_shared_experts,
    std::int64_t semantic_free_experts,
    void* cuda_stream) {
    if (grad_route_logits == nullptr || rows <= 0 || experts <= 0 || semantic_vocab_dims <= 0 ||
        semantic_shared_experts < 0 || semantic_free_experts < 0 ||
        semantic_shared_experts + semantic_vocab_dims + semantic_free_experts > experts ||
        (semantic_shared_experts > 0 && grad_shared_logits == nullptr) ||
        (semantic_free_experts > 0 && grad_free_bias == nullptr)) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_router_bias_backward_float32(
        grad_route_logits, grad_shared_logits, grad_free_bias, rows, experts, semantic_vocab_dims,
        semantic_shared_experts, semantic_free_experts, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_semantic_targets_from_matrix_int64(
    const std::int64_t* semantic_matrix,
    const std::int64_t* lm_targets,
    std::int64_t* semantic_targets,
    std::uint8_t* semantic_target_valid,
    std::int64_t rows,
    std::int64_t semantic_dims,
    std::int64_t semantic_vocab_dims,
    void* cuda_stream) {
    if (semantic_matrix == nullptr || lm_targets == nullptr || semantic_targets == nullptr ||
        semantic_target_valid == nullptr || rows <= 0 || semantic_dims <= 0 ||
        semantic_vocab_dims <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_targets_from_matrix_int64(
        semantic_matrix, lm_targets, semantic_targets, semantic_target_valid,
        rows, semantic_dims, semantic_vocab_dims, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_semantic_targets_from_tokens_u16_int64(
    const std::uint16_t* tokens,
    const std::int64_t* lm_targets,
    std::int64_t* semantic_targets,
    std::uint8_t* semantic_target_valid,
    std::int64_t rows,
    std::int64_t semantic_dims,
    std::int64_t semantic_terms,
    std::int64_t semantic_vocab_dims,
    void* cuda_stream) {
    if (tokens == nullptr || lm_targets == nullptr || semantic_targets == nullptr ||
        semantic_target_valid == nullptr || rows <= 0 || semantic_dims <= 0 ||
        semantic_terms <= 0 || semantic_vocab_dims <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_targets_from_tokens_u16_int64(
        tokens, lm_targets, semantic_targets, semantic_target_valid,
        rows, semantic_dims, semantic_terms, semantic_vocab_dims, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_semantic_target_matrix_from_tokens_u16_int64(
    const std::uint16_t* tokens,
    std::int64_t* semantic_matrix,
    const std::int64_t* term_counts,
    std::int64_t rows,
    std::int64_t semantic_dims,
    std::int64_t ignore_index,
    void* cuda_stream) {
    if (tokens == nullptr || semantic_matrix == nullptr || term_counts == nullptr ||
        rows <= 0 || semantic_dims <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_target_matrix_from_tokens_u16_int64(
        tokens, semantic_matrix, term_counts, rows, semantic_dims, ignore_index,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (tokens <= 0 || dim <= 0 || hidden_dim <= 0 || experts <= 0 || top_k <= 0 || top_k > experts) {
        return 1;
    }
    neuralfn::tile_cuda::launch_moe_swiglu_forward_float32(
        x, route_weights, route_indices, w1, w2, w3, out, tokens, dim, hidden_dim, experts, top_k, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (tokens <= 0 || dim <= 0 || hidden_dim <= 0 || experts <= 0 || top_k <= 0 || top_k > experts) {
        return 1;
    }
    neuralfn::tile_cuda::launch_moe_swiglu_backward_float32(
        x,
        route_weights,
        route_indices,
        w1,
        w2,
        w3,
        grad_out,
        grad_x,
        grad_w1,
        grad_w2,
        grad_w3,
        nullptr,
        tokens,
        dim,
        hidden_dim,
        experts,
        top_k,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (tokens <= 0 || dim <= 0 || hidden_dim <= 0 || experts <= 0 || top_k <= 0 || top_k > experts ||
        grad_route_weights == nullptr) {
        return 1;
    }
    neuralfn::tile_cuda::launch_moe_swiglu_backward_float32(
        x,
        route_weights,
        route_indices,
        w1,
        w2,
        w3,
        grad_out,
        grad_x,
        grad_w1,
        grad_w2,
        grad_w3,
        grad_route_weights,
        tokens,
        dim,
        hidden_dim,
        experts,
        top_k,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (tokens <= 0 || dim <= 0 || hidden_dim <= 0 || experts <= 0 ||
        top_k <= 0 || top_k > experts ||
        quantization_kind <= 0 || quantization_kind > 3) {
        return 1;
    }
    neuralfn::tile_cuda::launch_moe_swiglu_forward_quantized_float32(
        x, route_weights, route_indices, w1, w2, w3, out, tokens, dim, hidden_dim,
        experts, top_k, static_cast<int>(quantization_kind), as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (tokens <= 0 || dim <= 0 || hidden_dim <= 0 || experts <= 0 ||
        top_k <= 0 || top_k > experts || grad_route_weights == nullptr ||
        quantization_kind <= 0 || quantization_kind > 3) {
        return 1;
    }
    neuralfn::tile_cuda::launch_moe_swiglu_backward_quantized_float32(
        x,
        route_weights,
        route_indices,
        w1,
        w2,
        w3,
        grad_out,
        grad_x,
        grad_w1,
        grad_w2,
        grad_w3,
        grad_route_weights,
        tokens,
        dim,
        hidden_dim,
        experts,
        top_k,
        static_cast<int>(quantization_kind),
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_semantic_hash_int64(
    const float* sem_vec,
    const float* proj,
    std::int64_t* out,
    std::int64_t batch,
    std::int64_t dim,
    std::int64_t tables,
    std::int64_t planes,
    void* cuda_stream) {
    if (batch <= 0 || dim <= 0 || tables <= 0 || planes <= 0 || planes > 62) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_hash_int64(
        sem_vec, proj, out, batch, dim, tables, planes, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (batch <= 0 || residual_dim <= 0 || vocab_size <= 0 || n_buckets <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_attentionless_decoder_float32(
        bucket_indices, expert_output, bucket_embed, out_weight, out, batch, residual_dim, vocab_size, n_buckets, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_expert_bias_add_float32(
    const float* logits,
    const float* bias,
    float* out,
    std::int64_t n,
    std::int64_t experts,
    void* cuda_stream) {
    if (n <= 0 || experts <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_expert_bias_add_float32(
        logits, bias, out, n, experts, as_stream(cuda_stream));
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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_adamw_step_many_with_device_scale_hyper_float32(
        params,
        grads,
        grad_scale,
        exp_avgs,
        exp_avg_sqs,
        elements,
        weight_decays,
        hyper,
        buffer_count,
        max_elements,
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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_adamw_step_many_with_device_scale_bf16_shadow_hyper_float32(
        params,
        grads,
        grad_scale,
        exp_avgs,
        exp_avg_sqs,
        elements,
        weight_decays,
        bf16_shadow_offsets,
        bf16_shadow_bits,
        hyper,
        buffer_count,
        max_elements,
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
    void* cuda_stream) {
    if (kind < 1 || kind > 3) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_linear_quantized_float32(
        x, weight, bias, out, rows, input_dim, output_dim, has_bias, kind, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_input_quantized_float32(
    const float* grad_out,
    const float* weight,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    int kind,
    void* cuda_stream) {
    if (kind < 1 || kind > 3) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neuralfn::tile_cuda::launch_linear_backward_input_quantized_float32(
        grad_out, weight, grad_x, rows, input_dim, output_dim, kind, as_stream(cuda_stream));
    return launch_status();
}

__attribute__((visibility("default"), used))
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
    void* cuda_stream) {
    const cudaStream_t stream = as_stream(cuda_stream);
    neuralfn::tile_cuda::launch_linear_float32(
        x, q_weight, nullptr, q_projection, batch * seq_len, model_dim, model_dim, false, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_linear_float32(
        x, k_weight, nullptr, k_projection, batch * seq_len, model_dim, kv_heads * head_dim, false, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_linear_float32(
        x, v_weight, nullptr, v_projection, batch * seq_len, model_dim, kv_heads * head_dim, false, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_reshape_heads_float32(
        q_projection, q, batch, seq_len, heads, head_dim, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_reshape_heads_float32(
        k_projection, k, batch, seq_len, kv_heads, head_dim, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_reshape_heads_float32(
        v_projection, v, batch, seq_len, kv_heads, head_dim, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_rotary_embedding_float32(
        q, inv_freq, q_rope, batch, heads, seq_len, head_dim, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_rotary_embedding_float32(
        k, inv_freq, k_rope, batch, kv_heads, seq_len, head_dim, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_scaled_dot_product_attention_float32(
        q_rope, k_rope, v, attention, batch * model_dim, heads, kv_heads, seq_len, seq_len,
        head_dim, head_dim, scale, true, false, false, 0, 0, 0, 0, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_merge_heads_float32(
        attention, attention_flat, batch, heads, seq_len, head_dim, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_linear_float32(
        attention_flat, out_weight, nullptr, output, batch * seq_len, model_dim, model_dim, false, stream);
    return launch_status();
}

__attribute__((visibility("default"), used))
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
    void* cuda_stream) {
    const cudaStream_t stream = as_stream(cuda_stream);
    neuralfn::tile_cuda::launch_linear_backward_input_float32(
        grad_output, out_weight, grad_attention_flat, batch * seq_len, model_dim, model_dim, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_linear_backward_weight_float32(
        attention_flat, grad_output, grad_out_weight, batch * seq_len, model_dim, model_dim, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_reshape_heads_float32(
        grad_attention_flat, grad_attention, batch, seq_len, heads, head_dim, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_scaled_dot_product_attention_backward_float32(
        q_rope, k_rope, v, grad_attention, grad_q_rope, grad_k_rope, grad_v,
        batch, heads, kv_heads, seq_len, seq_len, head_dim, head_dim, scale, true, false, false,
        0, 0, 0, 0, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_rotary_embedding_backward_float32(
        grad_q_rope, inv_freq, grad_q, batch * heads * seq_len * head_dim, heads, seq_len, head_dim, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_rotary_embedding_backward_float32(
        grad_k_rope, inv_freq, grad_k, batch * kv_heads * seq_len * head_dim, kv_heads, seq_len, head_dim, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_merge_heads_float32(
        grad_q, grad_q_projection, batch, heads, seq_len, head_dim, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_merge_heads_float32(
        grad_k, grad_k_projection, batch, kv_heads, seq_len, head_dim, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_merge_heads_float32(
        grad_v, grad_v_projection, batch, kv_heads, seq_len, head_dim, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_linear_backward_input_float32(
        grad_q_projection, q_weight, grad_q_input, batch * seq_len, model_dim, model_dim, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_linear_backward_input_float32(
        grad_k_projection, k_weight, grad_k_input, batch * seq_len, model_dim, kv_heads * head_dim, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_linear_backward_input_float32(
        grad_v_projection, v_weight, grad_v_input, batch * seq_len, model_dim, kv_heads * head_dim, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_linear_backward_weight_float32(
        x, grad_q_projection, grad_q_weight, batch * seq_len, model_dim, model_dim, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_linear_backward_weight_float32(
        x, grad_k_projection, grad_k_weight, batch * seq_len, model_dim, kv_heads * head_dim, stream);
    if (launch_status() != 0) return launch_status();
    neuralfn::tile_cuda::launch_linear_backward_weight_float32(
        x, grad_v_projection, grad_v_weight, batch * seq_len, model_dim, kv_heads * head_dim, stream);
    return launch_status();
}

int nfn_native_tile_split_last_dim_float32(
    const float* input,
    float* first,
    float* second,
    std::int64_t rows,
    std::int64_t dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_split_last_dim_float32(
        input, first, second, rows, dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_merge_last_dim_float32(
    const float* first,
    const float* second,
    float* output,
    std::int64_t rows,
    std::int64_t half_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_merge_last_dim_float32(
        first, second, output, rows, half_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_split_at_last_dim_float32(
    const float* input,
    float* first,
    float* second,
    std::int64_t rows,
    std::int64_t first_dim,
    std::int64_t second_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_split_at_last_dim_float32(
        input, first, second, rows, first_dim, second_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_concat_last_dim_float32(
    const float* first,
    const float* second,
    float* output,
    std::int64_t rows,
    std::int64_t first_dim,
    std::int64_t second_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_concat_last_dim_float32(
        first, second, output, rows, first_dim, second_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_differential_combine_float32(
    const float* first,
    const float* second,
    float* output,
    std::int64_t elements,
    float lambda,
    float output_scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_differential_combine_float32(
        first, second, output, elements, lambda, output_scale, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_differential_backward_float32(
    const float* grad_output,
    float* grad_first,
    float* grad_second,
    std::int64_t elements,
    float lambda,
    float output_scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_differential_backward_float32(
        grad_output, grad_first, grad_second, elements, lambda, output_scale, as_stream(cuda_stream));
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

int nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_strided_float32(
    const std::uint16_t* grad_out_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    std::int64_t grad_out_row_stride,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_input_bf16_bits_weight_bf16_strided_float32(
        grad_out_bf16_bits,
        weight_bf16_bits,
        grad_x,
        rows,
        input_dim,
        output_dim,
        grad_out_row_stride,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_input_bf16_bits_weight_bf16_strided_cublaslt_float32(
    const std::uint16_t* grad_out_bf16_bits,
    const std::uint16_t* weight_bf16_bits,
    float* grad_x,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    std::int64_t grad_out_row_stride,
    void* cuda_stream) {
    const bool launched =
        neuralfn::tile_cuda::cublaslt_linear_backward_input_bf16_bits_weight_bf16_strided_float32(
            grad_out_bf16_bits,
            weight_bf16_bits,
            grad_x,
            rows,
            input_dim,
            output_dim,
            grad_out_row_stride,
            as_stream(cuda_stream));
    if (!launched) {
        return static_cast<int>(cudaErrorNotSupported);
    }
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

int nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_float32_beta(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    std::int64_t grad_out_row_stride,
    float beta,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_float32_beta(
        x_bf16_bits,
        grad_out_bf16_bits,
        grad_weight,
        rows,
        input_dim,
        output_dim,
        grad_out_row_stride,
        beta,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_cublaslt_float32_beta(
    const std::uint16_t* x_bf16_bits,
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    std::int64_t grad_out_row_stride,
    float beta,
    void* cuda_stream) {
    const bool launched =
        neuralfn::tile_cuda::cublaslt_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_float32_beta(
            x_bf16_bits,
            grad_out_bf16_bits,
            grad_weight,
            rows,
            input_dim,
            output_dim,
            grad_out_row_stride,
            beta,
            as_stream(cuda_stream));
    if (!launched) {
        return static_cast<int>(cudaErrorNotSupported);
    }
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

int nfn_native_tile_linear_backward_bias_accumulate_bf16_bits_float32(
    const std::uint16_t* grad_out_bf16_bits,
    float* grad_bias,
    std::int64_t rows,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_backward_bias_accumulate_bf16_bits_float32(
        grad_out_bf16_bits, grad_bias, rows, output_dim, as_stream(cuda_stream));
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

int nfn_native_tile_moa_add_bias_float32(
    const float* x,
    const float* bias,
    float* biased_out,
    float* activation_out,
    std::int64_t rows,
    std::int64_t output_dim,
    int activation_kind,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_moa_add_bias_float32(
        x, bias, biased_out, activation_out, rows, output_dim, activation_kind, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_moa_add_bias_bf16_act_float32(
    const float* x,
    const float* bias,
    float* biased_out,
    float* activation_out,
    std::uint16_t* activation_bf16_bits,
    std::int64_t rows,
    std::int64_t output_dim,
    int activation_kind,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_moa_add_bias_bf16_act_float32(
        x,
        bias,
        biased_out,
        activation_out,
        activation_bf16_bits,
        rows,
        output_dim,
        activation_kind,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_swiglu_float32(
    const float* gate,
    const float* up,
    float* out,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_swiglu_float32(gate, up, out, n, as_stream(cuda_stream));
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

int nfn_native_tile_linear_bias_residual_add_bf16_linear_bf16_residual_float32(
    const float* residual,
    const std::uint16_t* linear_out_bf16_bits,
    const float* bias,
    const float* residual_scale,
    float* out,
    std::uint16_t* residual_bf16_out,
    std::int64_t rows,
    std::int64_t output_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_linear_bias_residual_add_bf16_linear_bf16_residual_float32(
        residual,
        linear_out_bf16_bits,
        bias,
        residual_scale,
        out,
        residual_bf16_out,
        rows,
        output_dim,
        as_stream(cuda_stream));
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

int nfn_native_tile_swiglu_backward_float32(
    const float* gate,
    const float* up,
    const float* grad_out,
    float* grad_gate,
    float* grad_up,
    std::int64_t n,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_swiglu_backward_float32(
        gate, up, grad_out, grad_gate, grad_up, n, as_stream(cuda_stream));
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

int nfn_native_tile_moa_backward_inplace_float32(
    const float* x,
    float* grad,
    std::int64_t n,
    int activation_kind,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_moa_backward_inplace_float32(
        x, grad, n, activation_kind, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_moa_backward_inplace_bf16_bits_float32(
    const std::uint16_t* x_bf16_bits,
    float* grad,
    std::int64_t n,
    int activation_kind,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_moa_backward_inplace_bf16_bits_float32(
        x_bf16_bits, grad, n, activation_kind, as_stream(cuda_stream));
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

int nfn_native_tile_token_position_embedding_residual_u16_bf16_weight_float32(
    const std::uint16_t* token_weight_bf16,
    const std::uint16_t* token_ids,
    const float* position_weight,
    const float* scale,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_position_embedding_residual_u16_bf16_weight_float32(
        token_weight_bf16,
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

int nfn_native_tile_random_timesteps_float32(
    float* out,
    std::int64_t batch,
    std::int64_t counter,
    void* cuda_stream) {
    if (batch <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_random_timesteps_float32(out, batch, counter, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_mask_scheduler_int64(
    const std::int64_t* tokens,
    const float* timesteps,
    std::int64_t* out,
    std::int64_t n,
    std::int64_t seq_len,
    std::int64_t mask_token_id,
    std::int64_t counter,
    void* cuda_stream) {
    if (n <= 0 || seq_len <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_mask_scheduler_int64(
        tokens, timesteps, out, n, seq_len, mask_token_id, counter, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_rotary_embedding_float32(
    const float* x,
    const float* inv_freq,
    float* out,
    std::int64_t n,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    void* cuda_stream) {
    if (head_dim <= 0 || (head_dim % 2) != 0 || heads <= 0 || seq_len <= 0) {
        return 1;
    }
    const std::int64_t denominator = heads * seq_len * head_dim;
    if (n <= 0 || denominator <= 0 || (n % denominator) != 0) {
        return 1;
    }
    const std::int64_t batch = n / denominator;
    neuralfn::tile_cuda::launch_rotary_embedding_float32(
        x, inv_freq, out, batch, heads, seq_len, head_dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_rotary_embedding_backward_float32(
    const float* grad_out,
    const float* inv_freq,
    float* grad_x,
    std::int64_t n,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    void* cuda_stream) {
    if (head_dim <= 0 || (head_dim % 2) != 0 || heads <= 0 || seq_len <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_rotary_embedding_backward_float32(
        grad_out, inv_freq, grad_x, n, heads, seq_len, head_dim, as_stream(cuda_stream));
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

int nfn_native_tile_token_cross_entropy_z_partials_strided_bf16_bits_u16_targets(
    const std::uint16_t* logits_bf16_bits,
    const std::uint16_t* targets,
    float* partials,
    float* z_partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_cross_entropy_z_partials_strided_bf16_bits_u16_targets(
        logits_bf16_bits, targets, partials, z_partials, rows, vocab, row_stride,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_cross_entropy_variant_bf16_u16(
        logits_bf16_bits, targets, row_losses, rows, vocab, row_stride, loss_scale,
        z_loss_coef, logit_softcap, write_gradient, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_qk_rms_norm_packed_bf16_forward(
    std::uint16_t* packed_qkv_bits,
    float* rstd,
    std::int64_t rows,
    std::int64_t heads,
    std::int64_t head_dim,
    float eps,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_qk_rms_norm_packed_bf16_forward(
        packed_qkv_bits, rstd, rows, heads, head_dim, eps, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_qk_rms_norm_packed_bf16_backward(
    const std::uint16_t* normalized_qkv_bits,
    const float* rstd,
    float* grad_qkv_float,
    std::uint16_t* grad_qkv_bf16_bits,
    std::int64_t rows,
    std::int64_t heads,
    std::int64_t head_dim,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_qk_rms_norm_packed_bf16_backward(
        normalized_qkv_bits, rstd, grad_qkv_float, grad_qkv_bf16_bits,
        rows, heads, head_dim, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    return neuralfn::tile_cuda::launch_differential_packed_attention_forward_bf16(
        qkv_bf16_bits, out_bf16_bits, batch, heads, seq_len, head_dim,
        lambda, output_scale, eps, as_stream(cuda_stream));
}

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
    void* cuda_stream) {
    return neuralfn::tile_cuda::launch_differential_packed_attention_backward_bf16(
        out_bf16_bits, grad_out, grad_qkv_bf16_bits,
        batch, heads, seq_len, head_dim, lambda, output_scale,
        as_stream(cuda_stream));
}

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
    void* cuda_stream) {
    return neuralfn::tile_cuda::launch_differential_packed_attention_forward_learned_lambda_bf16(
        qkv_bf16_bits, out_bf16_bits, batch, heads, seq_len, head_dim,
        lambda, output_scale, eps, as_stream(cuda_stream));
}

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
    void* cuda_stream) {
    return neuralfn::tile_cuda::launch_differential_packed_attention_backward_learned_lambda_bf16(
        qkv_bf16_bits, out_bf16_bits, grad_out, grad_qkv_bf16_bits,
        batch, heads, seq_len, head_dim, lambda, output_scale, eps,
        grad_lambda, as_stream(cuda_stream));
}

int nfn_native_tile_differential_packed_attention_release_workspaces() {
    return neuralfn::tile_cuda::release_differential_packed_attention_workspaces();
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

int nfn_native_tile_latent_mse_loss_float32(
    const float* pred,
    const float* target,
    float* partials,
    std::int64_t n,
    void* cuda_stream) {
    if (n <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_latent_mse_partials_float32(
        pred, target, partials, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_act_weighted_sum_float32(
    const float* states,
    const float* weights,
    float* out,
    std::int64_t batch,
    std::int64_t steps,
    std::int64_t inner,
    void* cuda_stream) {
    if (batch <= 0 || steps <= 0 || inner <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_act_weighted_sum_float32(
        states, weights, out, batch, steps, inner, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_act_pack_step_float32(
    const float* state_step,
    const float* halt_logits_step,
    float* state_stack,
    float* halt_logits_stack,
    std::int64_t rows,
    std::int64_t steps,
    std::int64_t inner,
    std::int64_t step,
    void* cuda_stream) {
    if (rows <= 0 || steps <= 0 || inner <= 0 || step < 0 || step >= steps) {
        return 1;
    }
    neuralfn::tile_cuda::launch_act_pack_step_float32(
        state_step, halt_logits_step, state_stack, halt_logits_stack,
        rows, steps, inner, step, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_act_prepare_weights_float32(
    const float* halt_logits_stack,
    const std::int64_t* targets,
    float* halt_targets,
    float* halt_weights,
    std::int64_t rows,
    std::int64_t steps,
    float halt_epsilon,
    void* cuda_stream) {
    if (rows <= 0 || steps <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_act_prepare_weights_float32(
        halt_logits_stack, targets, halt_targets, halt_weights,
        rows, steps, halt_epsilon, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (rows <= 0 || steps <= 0 || inner <= 0 || step < 0 || step >= steps) {
        return 1;
    }
    neuralfn::tile_cuda::launch_act_unpack_step_grad_float32(
        grad_act, halt_weights, grad_halt_stack, grad_state_step, grad_halt_step,
        rows, steps, inner, step, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_act_halting_bce_grad_float32(
    const float* logits,
    const float* targets,
    float* partials,
    float* grad_logits,
    float* probs_out,
    std::int64_t n,
    void* cuda_stream) {
    if (n <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_act_halting_bce_grad_float32(
        logits, targets, partials, grad_logits, probs_out, n, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_latent_pool_float32(
    const float* x,
    const float* mask_values,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t dim,
    void* cuda_stream) {
    if (batch <= 0 || seq_len <= 0 || dim <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_latent_pool_float32(
        x, mask_values, out, batch, seq_len, dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_latent_pool_backward_float32(
    const float* grad_pooled,
    const float* mask_values,
    float* grad_x,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t dim,
    void* cuda_stream) {
    if (batch <= 0 || seq_len <= 0 || dim <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_latent_pool_backward_float32(
        grad_pooled, mask_values, grad_x, batch, seq_len, dim, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_native_family_jepa_mask_float32(
    float* mask_values,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t masked_span,
    float mask_ratio,
    int strategy,
    void* cuda_stream) {
    if (batch <= 0 || seq_len <= 0 || masked_span < 0 || masked_span > seq_len ||
        mask_ratio < 0.0f || mask_ratio > 1.0f) {
        return 1;
    }
    neuralfn::tile_cuda::launch_native_family_jepa_mask_float32(
        mask_values, batch, seq_len, masked_span, mask_ratio, strategy, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_native_family_jepa_mask_u16_float32(
    const std::uint16_t* tokens,
    std::uint16_t* masked_tokens,
    float* mask_values,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t masked_span,
    float mask_ratio,
    int strategy,
    void* cuda_stream) {
    if (tokens == nullptr || masked_tokens == nullptr || mask_values == nullptr ||
        batch <= 0 || seq_len <= 0 || masked_span < 0 || masked_span > seq_len ||
        mask_ratio < 0.0f || mask_ratio > 1.0f) {
        return 1;
    }
    neuralfn::tile_cuda::launch_native_family_jepa_mask_u16_float32(
        tokens, masked_tokens, mask_values, batch, seq_len, masked_span, mask_ratio,
        strategy, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (n <= 0 || dims <= 0 || terms <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_alignment_loss_items_float32(
        logits, targets, term_counts, losses, counts, n, dims, terms, ignore_index, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (logits == nullptr || targets == nullptr || term_counts == nullptr ||
        term_offsets == nullptr || losses == nullptr || counts == nullptr ||
        grad_logits == nullptr || n <= 0 || dims <= 0 || total_terms <= 0) {
        return 1;
    }
    neuralfn::tile_cuda::launch_semantic_alignment_packed_loss_backward_float32(
        logits, targets, term_counts, term_offsets, losses, counts, grad_logits,
        n, dims, total_terms, ignore_index, grad_scale, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_route_balance_density_float32(
    const float* route_logits,
    float* density,
    std::int64_t rows,
    std::int64_t experts,
    void* cuda_stream) {
    if (rows <= 0 || rows > 1024 || experts <= 0 || experts > 1024) {
        return 1;
    }
    neuralfn::tile_cuda::launch_route_balance_density_float32(
        route_logits, density, rows, experts, as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    if (rows <= 0 || seq_len <= 0 || experts <= 0 || num_vocab_dims <= 0 ||
        shared_experts < 0 || shared_experts + num_vocab_dims > experts) {
        return 1;
    }
    const std::int64_t n = rows * seq_len * num_vocab_dims;
    neuralfn::tile_cuda::launch_route_selection_loss_partials_float32(
        route_logits,
        sem_targets,
        loss_partials,
        count_partials,
        n,
        seq_len,
        experts,
        num_vocab_dims,
        shared_experts,
        ignore_index,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_route_balance_loss_float32(
    const float* density,
    float* out,
    std::int64_t experts,
    void* cuda_stream) {
    if (experts <= 0 || experts > 1024) {
        return 1;
    }
    neuralfn::tile_cuda::launch_route_balance_loss_float32(
        density, out, experts, as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_moe_router_aux_loss_backward_float32(
    const float* router_logits,
    float* density_workspace,
    float* weighted_loss_accumulator,
    float* grad_router_logits,
    std::int64_t rows,
    std::int64_t experts,
    float coefficient,
    void* cuda_stream) {
    if (rows <= 0 || experts <= 0 ||
        rows > std::numeric_limits<std::int64_t>::max() / experts ||
        rows > std::numeric_limits<int>::max() ||
        experts > std::numeric_limits<int>::max() ||
        !std::isfinite(coefficient) || coefficient < 0.0f) {
        return 1;
    }
    if (coefficient == 0.0f) {
        return 0;
    }
    if (router_logits == nullptr || density_workspace == nullptr ||
        weighted_loss_accumulator == nullptr || grad_router_logits == nullptr) {
        return 1;
    }
    neuralfn::tile_cuda::launch_moe_router_aux_loss_backward_float32(
        router_logits,
        density_workspace,
        weighted_loss_accumulator,
        grad_router_logits,
        rows,
        experts,
        coefficient,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_softmax_distillation_partials_float32(
    const float* teacher_logits,
    const float* student_logits,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    void* cuda_stream) {
    if (rows <= 0 || vocab <= 0 || vocab > 1024) {
        return 1;
    }
    neuralfn::tile_cuda::launch_softmax_distillation_partials_float32(
        teacher_logits, student_logits, partials, rows, vocab, as_stream(cuda_stream));
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

int nfn_native_tile_token_cross_entropy_backward_inplace_strided_no_pad_zero_with_workspace_float32(
    float* logits,
    const std::int64_t* targets,
    float* row_max_workspace,
    float* row_denom_workspace,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_cross_entropy_backward_inplace_strided_no_pad_zero_with_workspace_float32(
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

int nfn_native_tile_token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_with_workspace(
    std::uint16_t* logits,
    const std::int64_t* targets,
    float* row_max_workspace,
    float* row_denom_workspace,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_with_workspace(
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

int nfn_native_tile_token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* row_max_workspace,
    float* row_denom_workspace,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_cross_entropy_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace(
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

int nfn_native_tile_token_cross_entropy_backward_loss_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* loss_total,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_token_cross_entropy_backward_loss_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
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

int nfn_native_tile_lm_head_classifier_backward_loss_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* loss_total,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_lm_head_classifier_backward_loss_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
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

int nfn_native_tile_lm_head_classifier_backward_row_losses_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* row_losses,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_lm_head_classifier_backward_row_losses_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
        logits,
        targets,
        row_losses,
        rows,
        vocab,
        row_stride,
        loss_scale,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_lm_head_classifier_backward_loss_bins_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* loss_bins,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    std::int64_t loss_bin_count,
    float loss_scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_lm_head_classifier_backward_loss_bins_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
        logits,
        targets,
        loss_bins,
        rows,
        vocab,
        row_stride,
        loss_bin_count,
        loss_scale,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_lm_head_classifier_backward_prob_only_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_lm_head_classifier_backward_prob_only_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
        logits,
        targets,
        rows,
        vocab,
        row_stride,
        loss_scale,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_lm_head_classifier_backward_prob_only_ce_target_correction_bf16_bits(
        logits,
        targets,
        token_weight_bf16,
        hidden_bf16,
        grad_hidden,
        grad_weight,
        rows,
        vocab,
        row_stride,
        hidden_dim,
        token_weight_row_stride,
        grad_weight_row_stride,
        loss_scale,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_lm_head_prob_only_dhidden_target_correction_bf16_bits(
    const std::uint16_t* targets,
    const std::uint16_t* token_weight_bf16,
    float* grad_hidden,
    std::int64_t rows,
    std::int64_t hidden_dim,
    std::int64_t token_weight_row_stride,
    float loss_scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_lm_head_prob_only_dhidden_target_correction_bf16_bits(
        targets,
        token_weight_bf16,
        grad_hidden,
        rows,
        hidden_dim,
        token_weight_row_stride,
        loss_scale,
        as_stream(cuda_stream));
    return launch_status();
}

int nfn_native_tile_lm_head_prob_only_dweight_target_correction_bf16_bits(
    const std::uint16_t* targets,
    const std::uint16_t* hidden_bf16,
    float* grad_weight,
    std::int64_t rows,
    std::int64_t hidden_dim,
    std::int64_t grad_weight_row_stride,
    float loss_scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_lm_head_prob_only_dweight_target_correction_bf16_bits(
        targets,
        hidden_bf16,
        grad_weight,
        rows,
        hidden_dim,
        grad_weight_row_stride,
        loss_scale,
        as_stream(cuda_stream));
    return launch_status();
}

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
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_lm_head_prob_only_combined_target_correction_bf16_bits(
        targets,
        token_weight_bf16,
        hidden_bf16,
        grad_hidden,
        grad_weight,
        rows,
        hidden_dim,
        token_weight_row_stride,
        grad_weight_row_stride,
        loss_scale,
        as_stream(cuda_stream));
    return launch_status();
}

static int run_lm_head_classifier_backward_cooperative_sequence_bf16_u16(
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
    void* cuda_stream,
    bool schedule_dhidden_dweight_concurrently) {
    (void)hidden_float;
    (void)token_weight_float;
    const bool no_loss = (flags & kLmHeadCooperativeFlagNoLoss) != 0;
    if (logits_bf16 == nullptr ||
        targets_u16 == nullptr ||
        (!no_loss && row_losses == nullptr) ||
        hidden_bf16 == nullptr ||
        token_weight_bf16 == nullptr ||
        grad_hidden == nullptr ||
        grad_weight == nullptr ||
        rows <= 0 ||
        hidden_dim <= 0 ||
        vocab <= 0 ||
        row_stride < vocab) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    g_lm_head_cooperative_sequence_launch_count.fetch_add(1, std::memory_order_relaxed);
    if (schedule_dhidden_dweight_concurrently) {
        g_lm_head_cooperative_sequence_concurrent_count.fetch_add(1, std::memory_order_relaxed);
    }
    cudaStream_t stream = as_stream(cuda_stream);
    if ((flags & kLmHeadCooperativeFlagLossBins) != 0) {
        g_lm_head_cooperative_sequence_loss_bin_count.fetch_add(1, std::memory_order_relaxed);
        neuralfn::tile_cuda::launch_lm_head_classifier_backward_loss_bins_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
            logits_bf16,
            targets_u16,
            row_losses,
            rows,
            vocab,
            row_stride,
            lm_head_cooperative_loss_bin_count_from_flags(flags, rows),
            loss_scale,
            stream);
    } else if (no_loss) {
        neuralfn::tile_cuda::launch_lm_head_classifier_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace(
            logits_bf16,
            targets_u16,
            nullptr,
            nullptr,
            rows,
            vocab,
            row_stride,
            loss_scale,
            stream);
    } else {
        neuralfn::tile_cuda::launch_lm_head_classifier_backward_row_losses_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
            logits_bf16,
            targets_u16,
            row_losses,
            rows,
            vocab,
            row_stride,
            loss_scale,
            stream);
    }
    g_lm_head_cooperative_sequence_ce_launch_count.fetch_add(1, std::memory_order_relaxed);
    int status = launch_status();
    if (status != 0) {
        return status;
    }
    if (schedule_dhidden_dweight_concurrently) {
        LmHeadCooperativeStreams& cooperative_streams = lm_head_cooperative_streams();
        if (cooperative_streams.status != 0) {
            return cooperative_streams.status;
        }
        status = static_cast<int>(cudaEventRecord(cooperative_streams.ce_done, stream));
        if (status != 0) {
            return status;
        }
        status = static_cast<int>(cudaStreamWaitEvent(
            cooperative_streams.dhidden,
            cooperative_streams.ce_done,
            0));
        if (status != 0) {
            return status;
        }
        status = static_cast<int>(cudaStreamWaitEvent(
            cooperative_streams.dweight,
            cooperative_streams.ce_done,
            0));
        if (status != 0) {
            return status;
        }
        neuralfn::tile_cuda::launch_linear_backward_input_bf16_bits_weight_bf16_float32(
            logits_bf16,
            token_weight_bf16,
            grad_hidden,
            rows,
            hidden_dim,
            row_stride,
            cooperative_streams.dhidden);
        g_lm_head_cooperative_sequence_dhidden_launch_count.fetch_add(1, std::memory_order_relaxed);
        status = launch_status();
        if (status != 0) {
            return status;
        }
        neuralfn::tile_cuda::launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta(
            hidden_bf16,
            logits_bf16,
            grad_weight,
            rows,
            hidden_dim,
            row_stride,
            dweight_beta,
            cooperative_streams.dweight);
        g_lm_head_cooperative_sequence_dweight_launch_count.fetch_add(1, std::memory_order_relaxed);
        status = launch_status();
        if (status != 0) {
            return status;
        }
        status = static_cast<int>(cudaEventRecord(
            cooperative_streams.dhidden_done,
            cooperative_streams.dhidden));
        if (status != 0) {
            return status;
        }
        status = static_cast<int>(cudaEventRecord(
            cooperative_streams.dweight_done,
            cooperative_streams.dweight));
        if (status != 0) {
            return status;
        }
        status = static_cast<int>(cudaStreamWaitEvent(
            stream,
            cooperative_streams.dhidden_done,
            0));
        if (status != 0) {
            return status;
        }
        return static_cast<int>(cudaStreamWaitEvent(
            stream,
            cooperative_streams.dweight_done,
            0));
    }
    neuralfn::tile_cuda::launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta(
        hidden_bf16,
        logits_bf16,
        grad_weight,
        rows,
        hidden_dim,
        row_stride,
        dweight_beta,
        stream);
    g_lm_head_cooperative_sequence_dweight_launch_count.fetch_add(1, std::memory_order_relaxed);
    status = launch_status();
    if (status != 0) {
        return status;
    }
    neuralfn::tile_cuda::launch_linear_backward_input_bf16_bits_weight_bf16_float32(
        logits_bf16,
        token_weight_bf16,
        grad_hidden,
        rows,
        hidden_dim,
        row_stride,
        stream);
    g_lm_head_cooperative_sequence_dhidden_launch_count.fetch_add(1, std::memory_order_relaxed);
    return launch_status();
}

static int run_lm_head_classifier_backward_cooperative_sequence_bf16_u16_legacy_order(
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
    void* cuda_stream) {
    (void)hidden_float;
    (void)token_weight_float;
    const bool no_loss = (flags & kLmHeadCooperativeFlagNoLoss) != 0;
    if (logits_bf16 == nullptr ||
        targets_u16 == nullptr ||
        (!no_loss && row_losses == nullptr) ||
        hidden_bf16 == nullptr ||
        token_weight_bf16 == nullptr ||
        grad_hidden == nullptr ||
        grad_weight == nullptr ||
        rows <= 0 ||
        hidden_dim <= 0 ||
        vocab <= 0 ||
        row_stride < vocab) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    g_lm_head_cooperative_sequence_launch_count.fetch_add(1, std::memory_order_relaxed);
    g_lm_head_cooperative_sequence_legacy_count.fetch_add(1, std::memory_order_relaxed);
    cudaStream_t stream = as_stream(cuda_stream);
    if ((flags & kLmHeadCooperativeFlagLossBins) != 0) {
        g_lm_head_cooperative_sequence_loss_bin_count.fetch_add(1, std::memory_order_relaxed);
        neuralfn::tile_cuda::launch_lm_head_classifier_backward_loss_bins_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
            logits_bf16,
            targets_u16,
            row_losses,
            rows,
            vocab,
            row_stride,
            lm_head_cooperative_loss_bin_count_from_flags(flags, rows),
            loss_scale,
            stream);
    } else if (no_loss) {
        neuralfn::tile_cuda::launch_lm_head_classifier_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace(
            logits_bf16,
            targets_u16,
            nullptr,
            nullptr,
            rows,
            vocab,
            row_stride,
            loss_scale,
            stream);
    } else {
        neuralfn::tile_cuda::launch_lm_head_classifier_backward_row_losses_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
            logits_bf16,
            targets_u16,
            row_losses,
            rows,
            vocab,
            row_stride,
            loss_scale,
            stream);
    }
    g_lm_head_cooperative_sequence_ce_launch_count.fetch_add(1, std::memory_order_relaxed);
    int status = launch_status();
    if (status != 0) {
        return status;
    }
    neuralfn::tile_cuda::launch_linear_backward_input_bf16_bits_weight_bf16_float32(
        logits_bf16,
        token_weight_bf16,
        grad_hidden,
        rows,
        hidden_dim,
        row_stride,
        stream);
    g_lm_head_cooperative_sequence_dhidden_launch_count.fetch_add(1, std::memory_order_relaxed);
    status = launch_status();
    if (status != 0) {
        return status;
    }
    neuralfn::tile_cuda::launch_linear_backward_weight_accumulate_bf16_bits_bf16_bits_float32_beta(
        hidden_bf16,
        logits_bf16,
        grad_weight,
        rows,
        hidden_dim,
        row_stride,
        dweight_beta,
        stream);
    g_lm_head_cooperative_sequence_dweight_launch_count.fetch_add(1, std::memory_order_relaxed);
    return launch_status();
}

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
    void* cuda_stream) {
    return run_lm_head_classifier_backward_cooperative_sequence_bf16_u16_legacy_order(
        logits_bf16,
        targets_u16,
        row_losses,
        hidden_bf16,
        hidden_float,
        token_weight_bf16,
        token_weight_float,
        grad_hidden,
        grad_weight,
        rows,
        hidden_dim,
        vocab,
        row_stride,
        loss_scale,
        dweight_beta,
        flags,
        cuda_stream);
}

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
    void* cuda_stream) {
    return run_lm_head_classifier_backward_cooperative_sequence_bf16_u16(
        logits_bf16,
        targets_u16,
        row_losses,
        hidden_bf16,
        hidden_float,
        token_weight_bf16,
        token_weight_float,
        grad_hidden,
        grad_weight,
        rows,
        hidden_dim,
        vocab,
        row_stride,
        loss_scale,
        dweight_beta,
        flags,
        cuda_stream,
        true);
}

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
    void* cuda_stream) {
    (void)hidden_float;
    (void)token_weight_float;
    const bool no_loss = (flags & kLmHeadCooperativeFlagNoLoss) != 0;
    if (logits_bf16 == nullptr ||
        targets_u16 == nullptr ||
        (!no_loss && row_losses == nullptr) ||
        hidden_bf16 == nullptr ||
        token_weight_bf16 == nullptr ||
        grad_hidden == nullptr ||
        grad_weight == nullptr ||
        rows <= 0 ||
        hidden_dim <= 0 ||
        vocab <= 0 ||
        row_stride < vocab) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaStream_t stream = as_stream(cuda_stream);
    g_lm_head_cooperative_sequence_launch_count.fetch_add(1, std::memory_order_relaxed);
    if ((flags & kLmHeadCooperativeFlagLossBins) != 0) {
        g_lm_head_cooperative_sequence_loss_bin_count.fetch_add(1, std::memory_order_relaxed);
        neuralfn::tile_cuda::launch_lm_head_classifier_backward_loss_bins_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
            logits_bf16,
            targets_u16,
            row_losses,
            rows,
            vocab,
            row_stride,
            lm_head_cooperative_loss_bin_count_from_flags(flags, rows),
            loss_scale,
            stream);
    } else if (no_loss) {
        neuralfn::tile_cuda::launch_lm_head_classifier_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace(
            logits_bf16,
            targets_u16,
            nullptr,
            nullptr,
            rows,
            vocab,
            row_stride,
            loss_scale,
            stream);
    } else {
        neuralfn::tile_cuda::launch_lm_head_classifier_backward_row_losses_inplace_strided_no_pad_zero_bf16_bits_u16_targets(
            logits_bf16,
            targets_u16,
            row_losses,
            rows,
            vocab,
            row_stride,
            loss_scale,
            stream);
    }
    g_lm_head_cooperative_sequence_ce_launch_count.fetch_add(1, std::memory_order_relaxed);
    int status = launch_status();
    if (status != 0) {
        return status;
    }

    if (!neuralfn::tile_cuda::cublaslt_linear_backward_input_bf16_bits_weight_bf16_strided_float32(
            logits_bf16,
            token_weight_bf16,
            grad_hidden,
            rows,
            hidden_dim,
            vocab,
            row_stride,
            stream)) {
        return static_cast<int>(cudaErrorNotSupported);
    }
    g_lm_head_cooperative_sequence_dhidden_launch_count.fetch_add(1, std::memory_order_relaxed);
    status = launch_status();
    if (status != 0) {
        return status;
    }

    if (!neuralfn::tile_cuda::cublaslt_linear_backward_weight_accumulate_bf16_bits_bf16_bits_strided_float32_beta(
            hidden_bf16,
            logits_bf16,
            grad_weight,
            rows,
            hidden_dim,
            vocab,
            row_stride,
            dweight_beta,
            stream)) {
        return static_cast<int>(cudaErrorNotSupported);
    }
    g_lm_head_cooperative_sequence_dweight_launch_count.fetch_add(1, std::memory_order_relaxed);
    return launch_status();
}

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
    void* cuda_stream) {
    const bool no_loss = (flags & kLmHeadCooperativeFlagNoLoss) != 0;
    if (logits_bf16 == nullptr ||
        targets_u16 == nullptr ||
        (!no_loss && row_losses == nullptr) ||
        hidden_bf16 == nullptr ||
        token_weight_bf16 == nullptr ||
        grad_hidden == nullptr ||
        grad_weight == nullptr ||
        rows <= 0 ||
        hidden_dim <= 0 ||
        vocab <= 0 ||
        row_stride < vocab) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (lm_head_true_fused_cooperative_enabled()) {
        const cudaError_t status =
            neuralfn::tile_cuda::launch_lm_head_classifier_backward_true_fused_cooperative_bf16_bits_u16(
            logits_bf16,
            targets_u16,
            row_losses,
            hidden_bf16,
            token_weight_bf16,
            grad_hidden,
            grad_weight,
            rows,
            hidden_dim,
            vocab,
            row_stride,
            loss_scale,
            dweight_beta,
            no_loss,
            as_stream(cuda_stream));
        return status == cudaSuccess ? launch_status() : static_cast<int>(status);
    }
    const LmHeadBackwardGraphKey key{
        logits_bf16,
        targets_u16,
        row_losses,
        hidden_bf16,
        token_weight_bf16,
        grad_hidden,
        grad_weight,
        rows,
        hidden_dim,
        vocab,
        row_stride,
        lm_head_cooperative_loss_bin_count_from_flags(flags, rows),
        loss_scale,
        dweight_beta,
        flags,
    };
    const int graph_status =
        run_lm_head_classifier_backward_graph_bf16_u16(key, as_stream(cuda_stream));
    if (graph_status == 0) {
        return 0;
    }
    lm_head_graph_local_stats().fallback_count += 1;
    return run_lm_head_classifier_backward_cooperative_sequence_bf16_u16(
        logits_bf16,
        targets_u16,
        row_losses,
        hidden_bf16,
        hidden_float,
        token_weight_bf16,
        token_weight_float,
        grad_hidden,
        grad_weight,
        rows,
        hidden_dim,
        vocab,
        row_stride,
        loss_scale,
        dweight_beta,
        flags,
        cuda_stream,
        true);
}

int nfn_native_tile_lm_head_classifier_backward_fused_kernel_is_true_fused() {
    return lm_head_true_fused_cooperative_enabled() ? 1 : 0;
}

const char* nfn_native_tile_lm_head_classifier_backward_fused_kernel_path_class() {
    if (lm_head_true_fused_cooperative_enabled()) {
        return "strict-true-fused-tile-kernel";
    }
    return lm_head_graph_body_serial_enabled()
               ? "diagnostic-cuda-graph-wrapper-serial-body"
               : "diagnostic-cuda-graph-wrapper";
}

const char* nfn_native_tile_lm_head_classifier_backward_fused_kernel_implementation_class() {
    if (lm_head_true_fused_cooperative_enabled()) {
#if defined(NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_WMMA) && \
    NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_MAT_TILE == 16
#if defined(NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_THREADS) && \
    NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_THREADS == 128
#if defined(NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_CE_EXP2) && \
    NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_CE_EXP2
        return "wmma-bf16-cooperative-tile-warp128-exp2-ce-experimental";
#else
        return "wmma-bf16-cooperative-tile-warp128-experimental";
#endif
#elif defined(NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_THREADS) && \
    NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_THREADS == 32
#if defined(NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_CE_EXP2) && \
    NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_CE_EXP2
        return "wmma-bf16-cooperative-tile-warp32-exp2-ce-experimental";
#else
        return "wmma-bf16-cooperative-tile-warp32-experimental";
#endif
#else
#if defined(NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_CE_EXP2) && \
    NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_CE_EXP2
        return "wmma-bf16-cooperative-tile-exp2-ce-experimental";
#else
        return "wmma-bf16-cooperative-tile-experimental";
#endif
#endif
#else
#if defined(NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_CE_EXP2) && \
    NFN_TILE_CUDA_LM_HEAD_TRUE_FUSED_CE_EXP2
        return "scalar-cooperative-tile-exp2-ce-diagnostic";
#else
        return "scalar-cooperative-tile-diagnostic";
#endif
#endif
    }
    return lm_head_graph_body_serial_enabled()
               ? "diagnostic-cuda-graph-wrapper-serial-body"
               : "diagnostic-cuda-graph-wrapper";
}

int nfn_native_tile_lm_head_classifier_backward_fused_kernel_graph_body_node_count() {
    return 3;
}

int nfn_native_tile_lm_head_classifier_backward_fused_kernel_graph_body_ce_node_count() {
    return 1;
}

int nfn_native_tile_lm_head_classifier_backward_fused_kernel_graph_body_dhidden_node_count() {
    return 1;
}

int nfn_native_tile_lm_head_classifier_backward_fused_kernel_graph_body_dweight_node_count() {
    return 1;
}

int nfn_native_tile_lm_head_classifier_backward_llmk_classifier_matmul_parity() {
    return 1;
}

int nfn_native_tile_lm_head_classifier_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace(
    std::uint16_t* logits,
    const std::uint16_t* targets,
    float* row_max_workspace,
    float* row_denom_workspace,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t row_stride,
    float loss_scale,
    void* cuda_stream) {
    neuralfn::tile_cuda::launch_lm_head_classifier_backward_inplace_strided_no_pad_zero_bf16_bits_u16_targets_with_workspace(
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
    const int validation_status = validate_raw_sparse_attention_key_sequence_length(
        use_sparse_rules, seq_k);
    if (validation_status != 0) {
        return validation_status;
    }
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
    const int validation_status = validate_raw_sparse_attention_key_sequence_length(
        use_sparse_rules, seq_k);
    if (validation_status != 0) {
        return validation_status;
    }
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
    const int validation_status = validate_raw_sparse_attention_key_sequence_length(
        use_sparse_rules, seq_k);
    if (validation_status != 0) {
        return validation_status;
    }
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
