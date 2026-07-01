#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

namespace neuralfn::tile_cuda {

bool extension_loaded() {
  return true;
}

void launch_unary_float32(const float* x, float* out, std::int64_t n, int op, cudaStream_t stream);
void launch_binary_float32(
    const float* lhs,
    const float* rhs,
    float* out,
    std::int64_t n,
    int op,
    cudaStream_t stream);
void launch_binary_pair_float32(
    const float* lhs,
    const float* rhs,
    float* out0,
    float* out1,
    std::int64_t n,
    int op,
    cudaStream_t stream);
void launch_scalar_unary_float32(
    const float* x,
    float* out,
    std::int64_t n,
    float value,
    int op,
    cudaStream_t stream);
void launch_scalar_binary_float32(
    const float* lhs,
    const float* rhs,
    float* out,
    std::int64_t n,
    float value,
    int op,
    cudaStream_t stream);
void launch_ema_update_float32(
    float* target,
    const float* source,
    std::int64_t n,
    float decay,
    cudaStream_t stream);
void launch_gradient_accumulate_float32(
    float* buffer,
    const float* grad,
    std::int64_t n,
    float scale,
    cudaStream_t stream);
void launch_sumsq_partials_float32(
    const float* values,
    float* partials,
    std::int64_t n,
    cudaStream_t stream);
void launch_scale_inplace_float32(
    float* values,
    std::int64_t n,
    float scale,
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
void launch_scalar_ternary_float32(
    const float* a,
    const float* b,
    const float* c,
    float* out,
    std::int64_t n,
    float value,
    int op,
    cudaStream_t stream);
void launch_vector_binary_float32(
    const float* lhs,
    const float* rhs,
    const float* scale0,
    const float* scale1,
    float* out,
    std::int64_t n,
    std::int64_t dim,
    int op,
    cudaStream_t stream);
void launch_qk_gain_float32(
    const float* q,
    const float* gain,
    float* out,
    std::int64_t n,
    std::int64_t heads,
    std::int64_t inner,
    cudaStream_t stream);
void launch_dyt_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t n,
    std::int64_t dim,
    float alpha,
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
void launch_byte_patch_merge_float32(
    const float* x,
    float* out,
    std::int64_t batch,
    std::int64_t source_len,
    std::int64_t target_len,
    std::int64_t dim,
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
void launch_latent_mse_partials_float32(
    const float* pred,
    const float* target,
    float* partials,
    std::int64_t n,
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
void launch_sum_partials_float32(
    const float* values,
    float* partials,
    std::int64_t n,
    cudaStream_t stream);
void launch_scale_float32(const float* x, float* out, std::int64_t n, float value, cudaStream_t stream);
void launch_kv_cache_read_float32(
    const float* k,
    const float* v,
    const float* cache_k,
    const float* cache_v,
    float* out_k,
    float* out_v,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t cache_seq,
    std::int64_t current_seq,
    std::int64_t head_dim,
    cudaStream_t stream);
void launch_kv_quant_pack_float32(
    const float* k,
    const float* v,
    float* out,
    std::int64_t rows,
    std::int64_t head_dim,
    cudaStream_t stream);
void launch_kv_quant_unpack_float32(
    const float* packed,
    float* out_k,
    float* out_v,
    std::int64_t rows,
    std::int64_t head_dim,
    cudaStream_t stream);
void launch_absolute_position_embedding_float32(
    const float* weight,
    float* out,
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
void launch_rotary_embedding_float32(
    const float* x,
    const float* inv_freq,
    float* out,
    std::int64_t batch,
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
void launch_layer_norm_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    cudaStream_t stream);
void launch_softmax_lastdim_float32(
    const float* x,
    float* out,
    std::int64_t rows,
    std::int64_t dim,
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
void launch_group_norm_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t dim,
    std::int64_t num_groups,
    float eps,
    cudaStream_t stream);
void launch_scaled_residual_add_float32(
    const float* lhs,
    const float* rhs,
    const float* scale,
    float* out,
    std::int64_t n,
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
void launch_act_weighted_sum_float32(
    const float* states,
    const float* weights,
    float* out,
    std::int64_t batch,
    std::int64_t steps,
    std::int64_t inner,
    cudaStream_t stream);
void launch_latent_pool_float32(
    const float* x,
    const float* mask_values,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t dim,
    cudaStream_t stream);
void launch_token_cross_entropy_partials_float32(
    const float* logits,
    const std::int64_t* targets,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
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
void launch_sequence_logp_float32(
    const float* logits,
    const std::int64_t* targets,
    const float* loss_mask,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t vocab,
    std::int64_t ignore_index,
    cudaStream_t stream);
void launch_preference_bce_partials_float32(
    const float* reward_chosen,
    const float* reward_rejected,
    float* partials,
    std::int64_t n,
    cudaStream_t stream);
void launch_ppo_clipped_loss_partials_float32(
    const float* logp_new,
    const float* logp_old,
    const float* advantages,
    const float* value_new,
    const float* value_old,
    const float* returns,
    float* policy_partials,
    float* value_partials,
    std::int64_t n,
    float clip_range,
    cudaStream_t stream);
void launch_gae_compute_float32(
    const float* rewards,
    const float* values,
    float* advantages,
    float* returns,
    std::int64_t batch,
    std::int64_t seq_len,
    float gamma,
    float lambda_value,
    cudaStream_t stream);
void launch_dpo_pairwise_partials_float32(
    const float* policy_logp_chosen,
    const float* policy_logp_rejected,
    const float* ref_logp_chosen,
    const float* ref_logp_rejected,
    float* partials,
    float* chosen_reward_out,
    float* rejected_reward_out,
    std::int64_t n,
    float beta,
    float label_smoothing,
    int loss_type,
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
void launch_softmax_distillation_partials_float32(
    const float* teacher_logits,
    const float* student_logits,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
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
void launch_random_timesteps_float32(float* out, std::int64_t batch, std::int64_t counter, cudaStream_t stream);
void launch_mask_scheduler_int64(
    const std::int64_t* tokens,
    const float* timesteps,
    std::int64_t* out,
    std::int64_t n,
    std::int64_t seq_len,
    std::int64_t mask_token_id,
    std::int64_t counter,
    cudaStream_t stream);
void launch_jepa_mask_int64(
    const std::int64_t* tokens,
    std::int64_t* masked_tokens,
    float* mask_values,
    std::int64_t n,
    std::int64_t seq_len,
    float mask_ratio,
    std::int64_t mask_token_id,
    int strategy,
    std::int64_t num_blocks,
    float min_block_ratio,
    float max_block_ratio,
    std::int64_t counter,
    cudaStream_t stream);

torch::Tensor tile_unary(torch::Tensor x, std::int64_t op) {
  TORCH_CHECK(x.is_cuda(), "tile_unary expects a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "tile_unary only supports float32 tensors");
  TORCH_CHECK(x.is_contiguous(), "tile_unary expects a contiguous tensor");
  auto out = torch::empty_like(x);
  if (x.numel() == 0) {
    return out;
  }
  launch_unary_float32(
      x.data_ptr<float>(),
      out.data_ptr<float>(),
      x.numel(),
      static_cast<int>(op),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_binary(torch::Tensor lhs, torch::Tensor rhs, std::int64_t op) {
  TORCH_CHECK(lhs.is_cuda() && rhs.is_cuda(), "tile_binary expects CUDA tensors");
  TORCH_CHECK(
      lhs.scalar_type() == torch::kFloat32 && rhs.scalar_type() == torch::kFloat32,
      "tile_binary only supports float32 tensors");
  TORCH_CHECK(lhs.is_contiguous() && rhs.is_contiguous(), "tile_binary expects contiguous tensors");
  TORCH_CHECK(lhs.sizes() == rhs.sizes(), "tile_binary expects tensors with identical shapes");
  auto out = torch::empty_like(lhs);
  if (lhs.numel() == 0) {
    return out;
  }
  launch_binary_float32(
      lhs.data_ptr<float>(),
      rhs.data_ptr<float>(),
      out.data_ptr<float>(),
      lhs.numel(),
      static_cast<int>(op),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> tile_binary_pair(torch::Tensor lhs, torch::Tensor rhs, std::int64_t op) {
  TORCH_CHECK(lhs.is_cuda() && rhs.is_cuda(), "tile_binary_pair expects CUDA tensors");
  TORCH_CHECK(
      lhs.scalar_type() == torch::kFloat32 && rhs.scalar_type() == torch::kFloat32,
      "tile_binary_pair only supports float32 tensors");
  TORCH_CHECK(lhs.is_contiguous() && rhs.is_contiguous(), "tile_binary_pair expects contiguous tensors");
  TORCH_CHECK(lhs.sizes() == rhs.sizes(), "tile_binary_pair expects tensors with identical shapes");
  auto out0 = torch::empty_like(lhs);
  auto out1 = torch::empty_like(lhs);
  if (lhs.numel() == 0) {
    return {out0, out1};
  }
  launch_binary_pair_float32(
      lhs.data_ptr<float>(),
      rhs.data_ptr<float>(),
      out0.data_ptr<float>(),
      out1.data_ptr<float>(),
      lhs.numel(),
      static_cast<int>(op),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out0, out1};
}

torch::Tensor tile_scalar_unary(torch::Tensor x, double value, std::int64_t op) {
  TORCH_CHECK(x.is_cuda(), "tile_scalar_unary expects a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "tile_scalar_unary only supports float32 tensors");
  TORCH_CHECK(x.is_contiguous(), "tile_scalar_unary expects a contiguous tensor");
  auto out = torch::empty_like(x);
  if (x.numel() == 0) {
    return out;
  }
  launch_scalar_unary_float32(
      x.data_ptr<float>(),
      out.data_ptr<float>(),
      x.numel(),
      static_cast<float>(value),
      static_cast<int>(op),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_scalar_binary(torch::Tensor lhs, torch::Tensor rhs, double value, std::int64_t op) {
  TORCH_CHECK(lhs.is_cuda() && rhs.is_cuda(), "tile_scalar_binary expects CUDA tensors");
  TORCH_CHECK(
      lhs.scalar_type() == torch::kFloat32 && rhs.scalar_type() == torch::kFloat32,
      "tile_scalar_binary only supports float32 tensors");
  TORCH_CHECK(lhs.is_contiguous() && rhs.is_contiguous(), "tile_scalar_binary expects contiguous tensors");
  TORCH_CHECK(lhs.sizes() == rhs.sizes(), "tile_scalar_binary expects tensors with identical shapes");
  auto out = torch::empty_like(lhs);
  if (lhs.numel() == 0) {
    return out;
  }
  launch_scalar_binary_float32(
      lhs.data_ptr<float>(),
      rhs.data_ptr<float>(),
      out.data_ptr<float>(),
      lhs.numel(),
      static_cast<float>(value),
      static_cast<int>(op),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_scalar_ternary(torch::Tensor a, torch::Tensor b, torch::Tensor c, double value, std::int64_t op) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "tile_scalar_ternary expects CUDA tensors");
  TORCH_CHECK(
      a.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32 && c.scalar_type() == torch::kFloat32,
      "tile_scalar_ternary only supports float32 tensors");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(), "tile_scalar_ternary expects contiguous tensors");
  TORCH_CHECK(a.sizes() == b.sizes() && a.sizes() == c.sizes(), "tile_scalar_ternary expects tensors with identical shapes");
  auto out = torch::empty_like(a);
  if (a.numel() == 0) {
    return out;
  }
  launch_scalar_ternary_float32(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      c.data_ptr<float>(),
      out.data_ptr<float>(),
      a.numel(),
      static_cast<float>(value),
      static_cast<int>(op),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_vector_binary(
    torch::Tensor lhs,
    torch::Tensor rhs,
    torch::Tensor scale0,
    torch::Tensor scale1,
    std::int64_t op) {
  TORCH_CHECK(lhs.is_cuda() && rhs.is_cuda() && scale0.is_cuda(), "tile_vector_binary expects CUDA tensors");
  TORCH_CHECK(
      lhs.scalar_type() == torch::kFloat32 && rhs.scalar_type() == torch::kFloat32 && scale0.scalar_type() == torch::kFloat32,
      "tile_vector_binary only supports float32 tensors");
  TORCH_CHECK(lhs.is_contiguous() && rhs.is_contiguous() && scale0.is_contiguous(), "tile_vector_binary expects contiguous tensors");
  TORCH_CHECK(lhs.sizes() == rhs.sizes(), "tile_vector_binary expects data tensors with identical shapes");
  TORCH_CHECK(scale0.dim() == 1, "tile_vector_binary expects a 1D scale tensor");
  TORCH_CHECK(scale0.numel() > 0, "tile_vector_binary expects a non-empty scale tensor");
  TORCH_CHECK(lhs.numel() % scale0.numel() == 0, "tile_vector_binary expects scale length to divide the input numel");
  const float* scale1_ptr = nullptr;
  if (scale1.defined() && scale1.numel() > 0) {
    TORCH_CHECK(scale1.is_cuda(), "tile_vector_binary expects CUDA scale1 tensor");
    TORCH_CHECK(scale1.scalar_type() == torch::kFloat32, "tile_vector_binary only supports float32 scale1 tensor");
    TORCH_CHECK(scale1.is_contiguous(), "tile_vector_binary expects contiguous scale1 tensor");
    TORCH_CHECK(scale1.sizes() == scale0.sizes(), "tile_vector_binary expects scale tensors with identical shapes");
    scale1_ptr = scale1.data_ptr<float>();
  }
  auto out = torch::empty_like(lhs);
  if (lhs.numel() == 0) {
    return out;
  }
  launch_vector_binary_float32(
      lhs.data_ptr<float>(),
      rhs.data_ptr<float>(),
      scale0.data_ptr<float>(),
      scale1_ptr,
      out.data_ptr<float>(),
      lhs.numel(),
      scale0.numel(),
      static_cast<int>(op),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_qk_gain(torch::Tensor q, torch::Tensor gain) {
  TORCH_CHECK(q.is_cuda() && gain.is_cuda(), "tile_qk_gain expects CUDA tensors");
  TORCH_CHECK(q.scalar_type() == torch::kFloat32 && gain.scalar_type() == torch::kFloat32, "tile_qk_gain only supports float32 tensors");
  TORCH_CHECK(q.is_contiguous() && gain.is_contiguous(), "tile_qk_gain expects contiguous tensors");
  TORCH_CHECK(q.dim() >= 3, "tile_qk_gain expects q shaped [B, H, ...]");
  TORCH_CHECK(gain.dim() == 1, "tile_qk_gain expects a 1D gain tensor");
  TORCH_CHECK(q.size(1) == gain.numel(), "tile_qk_gain expects gain length to match q.size(1)");
  auto out = torch::empty_like(q);
  if (q.numel() == 0) {
    return out;
  }
  const auto inner = q.numel() / (q.size(0) * q.size(1));
  launch_qk_gain_float32(
      q.data_ptr<float>(),
      gain.data_ptr<float>(),
      out.data_ptr<float>(),
      q.numel(),
      gain.numel(),
      inner,
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_dyt(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor alpha) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda() && bias.is_cuda() && alpha.is_cuda(), "tile_dyt expects CUDA tensors");
  TORCH_CHECK(
      x.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32 &&
          bias.scalar_type() == torch::kFloat32 && alpha.scalar_type() == torch::kFloat32,
      "tile_dyt only supports float32 tensors");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && bias.is_contiguous(), "tile_dyt expects contiguous tensors");
  TORCH_CHECK(alpha.numel() == 1, "tile_dyt expects a scalar alpha tensor");
  TORCH_CHECK(weight.dim() == 1 && bias.dim() == 1 && weight.sizes() == bias.sizes(), "tile_dyt expects 1D weight and bias tensors with identical shapes");
  TORCH_CHECK(x.dim() >= 1 && x.size(-1) == weight.numel(), "tile_dyt expects weight length to match x.size(-1)");
  auto out = torch::empty_like(x);
  if (x.numel() == 0) {
    return out;
  }
  launch_dyt_float32(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias.data_ptr<float>(),
      out.data_ptr<float>(),
      x.numel(),
      weight.numel(),
      alpha.item<float>(),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_reshape_heads(torch::Tensor x, std::int64_t num_heads) {
  TORCH_CHECK(x.is_cuda(), "tile_reshape_heads expects a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "tile_reshape_heads only supports float32 tensors");
  TORCH_CHECK(x.is_contiguous(), "tile_reshape_heads expects a contiguous tensor");
  TORCH_CHECK(x.dim() == 3, "tile_reshape_heads expects input shaped [B, S, D]");
  TORCH_CHECK(num_heads > 0, "tile_reshape_heads expects num_heads > 0");
  TORCH_CHECK(x.size(2) % num_heads == 0, "tile_reshape_heads expects width divisible by num_heads");
  const auto batch = x.size(0);
  const auto seq_len = x.size(1);
  const auto head_dim = x.size(2) / num_heads;
  auto out = torch::empty({batch, num_heads, seq_len, head_dim}, x.options());
  if (x.numel() == 0) {
    return out;
  }
  launch_reshape_heads_float32(
      x.data_ptr<float>(),
      out.data_ptr<float>(),
      batch,
      seq_len,
      num_heads,
      head_dim,
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_merge_heads(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "tile_merge_heads expects a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "tile_merge_heads only supports float32 tensors");
  TORCH_CHECK(x.is_contiguous(), "tile_merge_heads expects a contiguous tensor");
  TORCH_CHECK(x.dim() == 4, "tile_merge_heads expects input shaped [B, H, S, D]");
  const auto batch = x.size(0);
  const auto heads = x.size(1);
  const auto seq_len = x.size(2);
  const auto head_dim = x.size(3);
  auto out = torch::empty({batch, seq_len, heads * head_dim}, x.options());
  if (x.numel() == 0) {
    return out;
  }
  launch_merge_heads_float32(
      x.data_ptr<float>(),
      out.data_ptr<float>(),
      batch,
      heads,
      seq_len,
      head_dim,
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_repeat_kv(torch::Tensor x, std::int64_t repeats) {
  TORCH_CHECK(x.is_cuda(), "tile_repeat_kv expects a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "tile_repeat_kv only supports float32 tensors");
  TORCH_CHECK(x.is_contiguous(), "tile_repeat_kv expects a contiguous tensor");
  TORCH_CHECK(x.dim() == 4, "tile_repeat_kv expects input shaped [B, Hkv, S, D]");
  TORCH_CHECK(repeats >= 1, "tile_repeat_kv expects repeats >= 1");
  const auto batch = x.size(0);
  const auto kv_heads = x.size(1);
  const auto seq_len = x.size(2);
  const auto head_dim = x.size(3);
  auto out = torch::empty({batch, kv_heads * repeats, seq_len, head_dim}, x.options());
  if (x.numel() == 0) {
    return out;
  }
  launch_repeat_kv_float32(
      x.data_ptr<float>(),
      out.data_ptr<float>(),
      batch,
      kv_heads,
      seq_len,
      head_dim,
      repeats,
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> tile_broadcast_expert_routes(torch::Tensor weights, torch::Tensor indices, std::int64_t seq_len) {
  TORCH_CHECK(weights.is_cuda() && indices.is_cuda(), "tile_broadcast_expert_routes expects CUDA tensors");
  TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "tile_broadcast_expert_routes only supports float32 weights");
  TORCH_CHECK(indices.scalar_type() == torch::kInt64, "tile_broadcast_expert_routes only supports int64 indices");
  TORCH_CHECK(weights.is_contiguous() && indices.is_contiguous(), "tile_broadcast_expert_routes expects contiguous tensors");
  TORCH_CHECK(weights.dim() == 3 && indices.sizes() == weights.sizes(), "tile_broadcast_expert_routes expects [B, route_seq, K] tensors");
  TORCH_CHECK(seq_len >= 0, "tile_broadcast_expert_routes expects seq_len >= 0");
  TORCH_CHECK(weights.size(1) == 1 || weights.size(1) == seq_len, "tile_broadcast_expert_routes route_seq must be 1 or seq_len");
  auto out_weights = torch::empty({weights.size(0), seq_len, weights.size(2)}, weights.options());
  auto out_indices = torch::empty({weights.size(0), seq_len, weights.size(2)}, indices.options());
  if (out_weights.numel() == 0) {
    return {out_weights, out_indices};
  }
  launch_broadcast_expert_routes_float32(
      weights.data_ptr<float>(),
      indices.data_ptr<std::int64_t>(),
      out_weights.data_ptr<float>(),
      out_indices.data_ptr<std::int64_t>(),
      weights.size(0),
      weights.size(1),
      seq_len,
      weights.size(2),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_weights, out_indices};
}

std::vector<torch::Tensor> tile_broadcast_chunk_routes(
    torch::Tensor weights,
    torch::Tensor indices,
    std::int64_t seq_len,
    std::int64_t chunk_size) {
  TORCH_CHECK(weights.is_cuda() && indices.is_cuda(), "tile_broadcast_chunk_routes expects CUDA tensors");
  TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "tile_broadcast_chunk_routes only supports float32 weights");
  TORCH_CHECK(indices.scalar_type() == torch::kInt64, "tile_broadcast_chunk_routes only supports int64 indices");
  TORCH_CHECK(weights.is_contiguous() && indices.is_contiguous(), "tile_broadcast_chunk_routes expects contiguous tensors");
  TORCH_CHECK(weights.dim() == 3 && indices.sizes() == weights.sizes(), "tile_broadcast_chunk_routes expects [B, chunks, K] tensors");
  TORCH_CHECK(seq_len >= 0, "tile_broadcast_chunk_routes expects seq_len >= 0");
  TORCH_CHECK(chunk_size >= 1, "tile_broadcast_chunk_routes expects chunk_size >= 1");
  auto out_weights = torch::empty({weights.size(0), seq_len, weights.size(2)}, weights.options());
  auto out_indices = torch::empty({weights.size(0), seq_len, weights.size(2)}, indices.options());
  if (out_weights.numel() == 0) {
    return {out_weights, out_indices};
  }
  launch_broadcast_chunk_routes_float32(
      weights.data_ptr<float>(),
      indices.data_ptr<std::int64_t>(),
      out_weights.data_ptr<float>(),
      out_indices.data_ptr<std::int64_t>(),
      weights.size(0),
      weights.size(1),
      seq_len,
      weights.size(2),
      chunk_size,
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_weights, out_indices};
}

torch::Tensor tile_byte_patch_merge(torch::Tensor x, std::int64_t target_len) {
  TORCH_CHECK(x.is_cuda(), "tile_byte_patch_merge expects a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "tile_byte_patch_merge only supports float32 tensors");
  TORCH_CHECK(x.is_contiguous(), "tile_byte_patch_merge expects a contiguous tensor");
  TORCH_CHECK(x.dim() == 3, "tile_byte_patch_merge expects input shaped [B, S, D]");
  TORCH_CHECK(target_len >= 0, "tile_byte_patch_merge expects target_len >= 0");
  auto out = torch::empty({x.size(0), target_len, x.size(2)}, x.options());
  if (out.numel() == 0) {
    return out;
  }
  launch_byte_patch_merge_float32(
      x.data_ptr<float>(),
      out.data_ptr<float>(),
      x.size(0),
      x.size(1),
      target_len,
      x.size(2),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_byte_patch_embed(
    torch::Tensor tokens,
    torch::Tensor embedding,
    torch::Tensor proj,
    std::int64_t patch_size,
    std::int64_t stride) {
  TORCH_CHECK(tokens.is_cuda() && embedding.is_cuda() && proj.is_cuda(), "tile_byte_patch_embed expects CUDA tensors");
  TORCH_CHECK(tokens.scalar_type() == torch::kInt64, "tile_byte_patch_embed expects int64 token ids");
  TORCH_CHECK(
      embedding.scalar_type() == torch::kFloat32 && proj.scalar_type() == torch::kFloat32,
      "tile_byte_patch_embed only supports float32 embedding/projection weights");
  TORCH_CHECK(tokens.is_contiguous() && embedding.is_contiguous() && proj.is_contiguous(), "tile_byte_patch_embed expects contiguous tensors");
  TORCH_CHECK(tokens.dim() == 2, "tile_byte_patch_embed expects tokens shaped [B,S]");
  TORCH_CHECK(embedding.dim() == 2, "tile_byte_patch_embed expects embedding shaped [V,D]");
  TORCH_CHECK(proj.dim() == 3, "tile_byte_patch_embed expects projection shaped [D,D,K]");
  TORCH_CHECK(patch_size >= 1 && stride >= 1, "tile_byte_patch_embed expects patch_size and stride >= 1");
  const auto batch = tokens.size(0);
  const auto seq_len = tokens.size(1);
  const auto vocab_size = embedding.size(0);
  const auto model_dim = embedding.size(1);
  TORCH_CHECK(seq_len > 0 && vocab_size > 0 && model_dim > 0, "tile_byte_patch_embed expects non-empty tokens and embedding");
  TORCH_CHECK(proj.size(0) == model_dim && proj.size(1) == model_dim && proj.size(2) == patch_size, "tile_byte_patch_embed projection shape must be [D,D,patch_size]");
  std::int64_t padded_len = seq_len;
  if (seq_len < patch_size) {
    padded_len = patch_size;
  } else {
    const auto remainder = (seq_len - patch_size) % stride;
    if (remainder != 0) {
      padded_len = seq_len + (stride - remainder);
    }
  }
  const auto out_len = (padded_len - patch_size) / stride + 1;
  auto out = torch::empty({batch, out_len, model_dim}, embedding.options());
  launch_byte_patch_embed_float32(
      tokens.data_ptr<std::int64_t>(),
      embedding.data_ptr<float>(),
      proj.data_ptr<float>(),
      out.data_ptr<float>(),
      batch,
      seq_len,
      model_dim,
      patch_size,
      stride,
      out_len,
      vocab_size,
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_causal_chunk_state(torch::Tensor hidden, std::int64_t chunk_size, std::string mode) {
  TORCH_CHECK(hidden.is_cuda(), "tile_causal_chunk_state expects a CUDA tensor");
  TORCH_CHECK(hidden.scalar_type() == torch::kFloat32, "tile_causal_chunk_state only supports float32 tensors");
  TORCH_CHECK(hidden.is_contiguous(), "tile_causal_chunk_state expects a contiguous tensor");
  TORCH_CHECK(hidden.dim() == 3, "tile_causal_chunk_state expects hidden shaped [B,S,D]");
  TORCH_CHECK(chunk_size >= 1, "tile_causal_chunk_state expects chunk_size >= 1");
  TORCH_CHECK(mode == "mean" || mode == "prefix", "tile_causal_chunk_state mode must be 'mean' or 'prefix'");
  const auto batch = hidden.size(0);
  const auto seq_len = hidden.size(1);
  const auto dim = hidden.size(2);
  TORCH_CHECK(seq_len > 0 && dim > 0, "tile_causal_chunk_state expects non-empty sequence and feature dimensions");
  const auto chunks = (seq_len + chunk_size - 1) / chunk_size;
  auto out = torch::empty({batch, chunks, dim}, hidden.options());
  launch_causal_chunk_state_float32(
      hidden.data_ptr<float>(),
      out.data_ptr<float>(),
      batch,
      seq_len,
      dim,
      chunk_size,
      chunks,
      mode == "mean" ? 0 : 1,
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_causal_chunk_state_backward(
    torch::Tensor grad_out,
    std::int64_t seq_len,
    std::int64_t chunk_size,
    std::string mode) {
  TORCH_CHECK(grad_out.is_cuda(), "tile_causal_chunk_state_backward expects a CUDA tensor");
  TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32, "tile_causal_chunk_state_backward only supports float32 tensors");
  TORCH_CHECK(grad_out.is_contiguous(), "tile_causal_chunk_state_backward expects a contiguous tensor");
  TORCH_CHECK(grad_out.dim() == 3, "tile_causal_chunk_state_backward expects grad_out shaped [B,C,D]");
  TORCH_CHECK(seq_len > 0, "tile_causal_chunk_state_backward expects seq_len > 0");
  TORCH_CHECK(chunk_size >= 1, "tile_causal_chunk_state_backward expects chunk_size >= 1");
  TORCH_CHECK(mode == "mean" || mode == "prefix", "tile_causal_chunk_state_backward mode must be 'mean' or 'prefix'");
  const auto batch = grad_out.size(0);
  const auto chunks = grad_out.size(1);
  const auto dim = grad_out.size(2);
  TORCH_CHECK(chunks > 0 && dim > 0, "tile_causal_chunk_state_backward expects non-empty chunk and feature dimensions");
  auto grad_hidden = torch::empty({batch, seq_len, dim}, grad_out.options());
  launch_causal_chunk_state_backward_float32(
      grad_out.data_ptr<float>(),
      grad_hidden.data_ptr<float>(),
      batch,
      seq_len,
      dim,
      chunk_size,
      chunks,
      mode == "mean" ? 0 : 1,
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_hidden;
}

torch::Tensor tile_latent_mse_loss(torch::Tensor pred, torch::Tensor target) {
  TORCH_CHECK(pred.is_cuda() && target.is_cuda(), "tile_latent_mse_loss expects CUDA tensors");
  TORCH_CHECK(pred.scalar_type() == torch::kFloat32 && target.scalar_type() == torch::kFloat32, "tile_latent_mse_loss only supports float32 tensors");
  TORCH_CHECK(pred.is_contiguous() && target.is_contiguous(), "tile_latent_mse_loss expects contiguous tensors");
  TORCH_CHECK(pred.sizes() == target.sizes(), "tile_latent_mse_loss expects same-shape tensors");
  TORCH_CHECK(pred.numel() > 0, "tile_latent_mse_loss expects non-empty tensors");
  auto stream = at::cuda::getCurrentCUDAStream();
  auto partials = torch::empty({(pred.numel() + 1023) / 1024}, pred.options());
  launch_latent_mse_partials_float32(pred.data_ptr<float>(), target.data_ptr<float>(), partials.data_ptr<float>(), pred.numel(), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  while (partials.numel() > 1) {
    auto next = torch::empty({(partials.numel() + 1023) / 1024}, pred.options());
    launch_sum_partials_float32(partials.data_ptr<float>(), next.data_ptr<float>(), partials.numel(), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    partials = next;
  }
  auto out = torch::empty({}, pred.options());
  launch_scale_float32(partials.data_ptr<float>(), out.data_ptr<float>(), 1, 1.0f / static_cast<float>(pred.numel()), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_semantic_alignment_loss(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor term_counts,
    std::int64_t ignore_index) {
  TORCH_CHECK(logits.is_cuda() && targets.is_cuda() && term_counts.is_cuda(), "tile_semantic_alignment_loss expects CUDA tensors");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32, "tile_semantic_alignment_loss expects float32 logits");
  TORCH_CHECK(targets.scalar_type() == torch::kInt64 && term_counts.scalar_type() == torch::kInt64, "tile_semantic_alignment_loss expects int64 targets and term_counts");
  TORCH_CHECK(logits.is_contiguous() && targets.is_contiguous() && term_counts.is_contiguous(), "tile_semantic_alignment_loss expects contiguous tensors");
  TORCH_CHECK(logits.dim() == 3, "tile_semantic_alignment_loss expects logits shaped [R,D,T]");
  TORCH_CHECK(targets.dim() == 2, "tile_semantic_alignment_loss expects targets shaped [R,D]");
  TORCH_CHECK(term_counts.dim() == 1, "tile_semantic_alignment_loss expects term_counts shaped [D]");
  TORCH_CHECK(logits.size(0) == targets.size(0) && logits.size(1) == targets.size(1), "tile_semantic_alignment_loss row/dim axes must match");
  TORCH_CHECK(term_counts.numel() == logits.size(1), "tile_semantic_alignment_loss term_counts must match dims");
  TORCH_CHECK(logits.size(0) > 0 && logits.size(1) > 0 && logits.size(2) > 0, "tile_semantic_alignment_loss expects non-empty logits");
  const auto rows = logits.size(0);
  const auto dims = logits.size(1);
  const auto terms = logits.size(2);
  const auto n = rows * dims;
  auto losses = torch::empty({rows, dims}, logits.options());
  auto counts = torch::empty_like(losses);
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_semantic_alignment_loss_items_float32(
      logits.data_ptr<float>(),
      targets.data_ptr<std::int64_t>(),
      term_counts.data_ptr<std::int64_t>(),
      losses.data_ptr<float>(),
      counts.data_ptr<float>(),
      n,
      dims,
      terms,
      ignore_index,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto loss_sum_by_dim = losses.sum(0);
  auto count_by_dim = counts.sum(0).clamp_min(1.0);
  return (loss_sum_by_dim / count_by_dim).mean();
}

std::vector<torch::Tensor> tile_kv_cache_read(
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor cache_k,
    torch::Tensor cache_v) {
  TORCH_CHECK(k.is_cuda() && v.is_cuda() && cache_k.is_cuda() && cache_v.is_cuda(), "tile_kv_cache_read expects CUDA tensors");
  TORCH_CHECK(
      k.scalar_type() == torch::kFloat32 && v.scalar_type() == torch::kFloat32 &&
          cache_k.scalar_type() == torch::kFloat32 && cache_v.scalar_type() == torch::kFloat32,
      "tile_kv_cache_read only supports float32 tensors");
  TORCH_CHECK(k.is_contiguous() && v.is_contiguous() && cache_k.is_contiguous() && cache_v.is_contiguous(), "tile_kv_cache_read expects contiguous tensors");
  TORCH_CHECK(k.dim() == 4 && v.sizes() == k.sizes(), "tile_kv_cache_read expects current K/V shaped [B,H,S,D]");
  TORCH_CHECK(cache_k.dim() == 4 && cache_v.sizes() == cache_k.sizes(), "tile_kv_cache_read expects cache K/V shaped [B,H,S,D]");
  TORCH_CHECK(cache_k.size(0) == k.size(0) && cache_k.size(1) == k.size(1) && cache_k.size(3) == k.size(3), "tile_kv_cache_read cache and current dimensions must match except sequence");
  auto out_k = torch::empty({k.size(0), k.size(1), cache_k.size(2) + k.size(2), k.size(3)}, k.options());
  auto out_v = torch::empty_like(out_k);
  if (out_k.numel() == 0) {
    return {out_k, out_v};
  }
  launch_kv_cache_read_float32(
      k.data_ptr<float>(),
      v.data_ptr<float>(),
      cache_k.data_ptr<float>(),
      cache_v.data_ptr<float>(),
      out_k.data_ptr<float>(),
      out_v.data_ptr<float>(),
      k.size(0),
      k.size(1),
      cache_k.size(2),
      k.size(2),
      k.size(3),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_k, out_v};
}

torch::Tensor tile_kv_quant_pack(torch::Tensor k, torch::Tensor v) {
  TORCH_CHECK(k.is_cuda() && v.is_cuda(), "tile_kv_quant_pack expects CUDA tensors");
  TORCH_CHECK(k.scalar_type() == torch::kFloat32 && v.scalar_type() == torch::kFloat32, "tile_kv_quant_pack only supports float32 tensors");
  TORCH_CHECK(k.is_contiguous() && v.is_contiguous(), "tile_kv_quant_pack expects contiguous tensors");
  TORCH_CHECK(k.sizes() == v.sizes(), "tile_kv_quant_pack expects same-shaped K/V tensors");
  TORCH_CHECK(k.dim() >= 1, "tile_kv_quant_pack expects at least one dimension");
  TORCH_CHECK(k.size(-1) > 0 && k.size(-1) <= 512, "tile_kv_quant_pack expects head_dim in 1..512");
  auto out_sizes = k.sizes().vec();
  out_sizes.back() = k.size(-1) * 2 + 1;
  auto out = torch::empty(out_sizes, k.options());
  if (out.numel() == 0) {
    return out;
  }
  const auto rows = k.numel() / k.size(-1);
  launch_kv_quant_pack_float32(
      k.data_ptr<float>(),
      v.data_ptr<float>(),
      out.data_ptr<float>(),
      rows,
      k.size(-1),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> tile_kv_quant_unpack(torch::Tensor packed, std::int64_t head_dim) {
  TORCH_CHECK(packed.is_cuda(), "tile_kv_quant_unpack expects a CUDA tensor");
  TORCH_CHECK(packed.scalar_type() == torch::kFloat32, "tile_kv_quant_unpack only supports float32 tensors");
  TORCH_CHECK(packed.is_contiguous(), "tile_kv_quant_unpack expects a contiguous tensor");
  TORCH_CHECK(packed.dim() >= 1, "tile_kv_quant_unpack expects at least one dimension");
  TORCH_CHECK(head_dim > 0 && head_dim <= 512, "tile_kv_quant_unpack expects head_dim in 1..512");
  TORCH_CHECK(packed.size(-1) == head_dim * 2 + 1, "tile_kv_quant_unpack expects packed last dim to equal 2*head_dim+1");
  auto out_sizes = packed.sizes().vec();
  out_sizes.back() = head_dim;
  auto out_k = torch::empty(out_sizes, packed.options());
  auto out_v = torch::empty(out_sizes, packed.options());
  if (packed.numel() == 0) {
    return {out_k, out_v};
  }
  const auto rows = packed.numel() / packed.size(-1);
  launch_kv_quant_unpack_float32(
      packed.data_ptr<float>(),
      out_k.data_ptr<float>(),
      out_v.data_ptr<float>(),
      rows,
      head_dim,
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_k, out_v};
}

torch::Tensor tile_absolute_position_embedding(torch::Tensor weight, std::int64_t batch, std::int64_t seq_len) {
  TORCH_CHECK(weight.is_cuda(), "tile_absolute_position_embedding expects a CUDA tensor");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "tile_absolute_position_embedding only supports float32 tensors");
  TORCH_CHECK(weight.is_contiguous(), "tile_absolute_position_embedding expects contiguous weight");
  TORCH_CHECK(weight.dim() == 2, "tile_absolute_position_embedding expects weight shaped [max_seq_len, model_dim]");
  TORCH_CHECK(batch >= 0 && seq_len >= 0, "tile_absolute_position_embedding expects non-negative batch and seq_len");
  TORCH_CHECK(seq_len <= weight.size(0), "tile_absolute_position_embedding seq_len exceeds weight length");
  auto out = torch::empty({batch, seq_len, weight.size(1)}, weight.options());
  if (out.numel() == 0) {
    return out;
  }
  launch_absolute_position_embedding_float32(
      weight.data_ptr<float>(),
      out.data_ptr<float>(),
      batch,
      seq_len,
      weight.size(1),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_token_embedding(torch::Tensor weight, torch::Tensor token_ids) {
  TORCH_CHECK(weight.is_cuda() && token_ids.is_cuda(), "tile_token_embedding expects CUDA tensors");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "tile_token_embedding only supports float32 weights");
  TORCH_CHECK(token_ids.scalar_type() == torch::kInt64, "tile_token_embedding only supports int64 token ids");
  TORCH_CHECK(weight.is_contiguous() && token_ids.is_contiguous(), "tile_token_embedding expects contiguous tensors");
  TORCH_CHECK(weight.dim() == 2, "tile_token_embedding expects weight shaped [vocab_size, model_dim]");
  auto out_sizes = token_ids.sizes().vec();
  out_sizes.push_back(weight.size(1));
  auto out = torch::empty(out_sizes, weight.options());
  if (out.numel() == 0) {
    return out;
  }
  launch_token_embedding_float32(
      weight.data_ptr<float>(),
      token_ids.data_ptr<std::int64_t>(),
      out.data_ptr<float>(),
      token_ids.numel(),
      weight.size(1),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> tile_rotary_embedding(torch::Tensor q, torch::Tensor k, torch::Tensor inv_freq) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && inv_freq.is_cuda(), "tile_rotary_embedding expects CUDA tensors");
  TORCH_CHECK(
      q.scalar_type() == torch::kFloat32 && k.scalar_type() == torch::kFloat32 && inv_freq.scalar_type() == torch::kFloat32,
      "tile_rotary_embedding only supports float32 tensors");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && inv_freq.is_contiguous(), "tile_rotary_embedding expects contiguous tensors");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "tile_rotary_embedding expects Q/K shaped [B,H,S,D]");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(2) == k.size(2) && q.size(3) == k.size(3), "tile_rotary_embedding expects Q/K to match batch, sequence, and head_dim");
  TORCH_CHECK(q.size(3) % 2 == 0, "tile_rotary_embedding expects an even head_dim");
  TORCH_CHECK(inv_freq.dim() == 1 && inv_freq.numel() == q.size(3) / 2, "tile_rotary_embedding expects inv_freq length D/2");
  auto out_q = torch::empty_like(q);
  auto out_k = torch::empty_like(k);
  if (out_q.numel() > 0) {
    launch_rotary_embedding_float32(
        q.data_ptr<float>(),
        inv_freq.data_ptr<float>(),
        out_q.data_ptr<float>(),
        q.size(0),
        q.size(1),
        q.size(2),
        q.size(3),
        at::cuda::getCurrentCUDAStream());
  }
  if (out_k.numel() > 0) {
    launch_rotary_embedding_float32(
        k.data_ptr<float>(),
        inv_freq.data_ptr<float>(),
        out_k.data_ptr<float>(),
        k.size(0),
        k.size(1),
        k.size(2),
        k.size(3),
        at::cuda::getCurrentCUDAStream());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_q, out_k};
}

torch::Tensor tile_rms_norm(torch::Tensor x, double eps) {
  TORCH_CHECK(x.is_cuda(), "tile_rms_norm expects a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "tile_rms_norm only supports float32 tensors");
  TORCH_CHECK(x.is_contiguous(), "tile_rms_norm expects contiguous tensors");
  TORCH_CHECK(x.dim() >= 1, "tile_rms_norm expects at least one dimension");
  TORCH_CHECK(x.size(-1) > 0 && x.size(-1) <= 1024, "tile_rms_norm expects last dim in 1..1024");
  auto out = torch::empty_like(x);
  if (x.numel() == 0) {
    return out;
  }
  const auto dim = x.size(-1);
  const auto rows = x.numel() / dim;
  launch_rms_norm_float32(
      x.data_ptr<float>(),
      out.data_ptr<float>(),
      rows,
      dim,
      static_cast<float>(eps),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_layer_norm(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda() && bias.is_cuda(), "tile_layer_norm expects CUDA tensors");
  TORCH_CHECK(
      x.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32 && bias.scalar_type() == torch::kFloat32,
      "tile_layer_norm only supports float32 tensors");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && bias.is_contiguous(), "tile_layer_norm expects contiguous tensors");
  TORCH_CHECK(x.dim() >= 1, "tile_layer_norm expects at least one dimension");
  TORCH_CHECK(x.size(-1) > 0 && x.size(-1) <= 1024, "tile_layer_norm expects last dim in 1..1024");
  TORCH_CHECK(weight.dim() == 1 && bias.sizes() == weight.sizes() && weight.numel() == x.size(-1), "tile_layer_norm expects 1D affine parameters matching last dim");
  auto out = torch::empty_like(x);
  if (x.numel() == 0) {
    return out;
  }
  const auto dim = x.size(-1);
  const auto rows = x.numel() / dim;
  launch_layer_norm_float32(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias.data_ptr<float>(),
      out.data_ptr<float>(),
      rows,
      dim,
      static_cast<float>(eps),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_group_norm(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, std::int64_t num_groups, double eps) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda() && bias.is_cuda(), "tile_group_norm expects CUDA tensors");
  TORCH_CHECK(
      x.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32 && bias.scalar_type() == torch::kFloat32,
      "tile_group_norm only supports float32 tensors");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && bias.is_contiguous(), "tile_group_norm expects contiguous tensors");
  TORCH_CHECK(x.dim() == 3, "tile_group_norm expects input shaped [B,S,D]");
  TORCH_CHECK(x.size(1) > 0 && x.size(2) > 0, "tile_group_norm expects non-empty sequence and feature dimensions");
  TORCH_CHECK(num_groups >= 1, "tile_group_norm expects num_groups >= 1");
  TORCH_CHECK(x.size(2) % num_groups == 0, "tile_group_norm expects feature dim divisible by num_groups");
  TORCH_CHECK(weight.dim() == 1 && bias.sizes() == weight.sizes() && weight.numel() == x.size(2), "tile_group_norm expects 1D affine parameters matching feature dim");
  const auto group_elems = x.size(1) * (x.size(2) / num_groups);
  TORCH_CHECK(group_elems <= 1024, "tile_group_norm expects seq_len * group_dim <= 1024");
  auto out = torch::empty_like(x);
  if (x.numel() == 0) {
    return out;
  }
  launch_group_norm_float32(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias.data_ptr<float>(),
      out.data_ptr<float>(),
      x.size(0),
      x.size(1),
      x.size(2),
      num_groups,
      static_cast<float>(eps),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_softmax_lastdim(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "tile_softmax_lastdim expects a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "tile_softmax_lastdim only supports float32 tensors");
  TORCH_CHECK(x.is_contiguous(), "tile_softmax_lastdim expects a contiguous tensor");
  TORCH_CHECK(x.dim() >= 1, "tile_softmax_lastdim expects at least one dimension");
  TORCH_CHECK(x.size(-1) > 0 && x.size(-1) <= 1024, "tile_softmax_lastdim expects last dim in 1..1024");
  auto out = torch::empty_like(x);
  if (x.numel() == 0) {
    return out;
  }
  const auto dim = x.size(-1);
  const auto rows = x.numel() / dim;
  launch_softmax_lastdim_float32(
      x.data_ptr<float>(),
      out.data_ptr<float>(),
      rows,
      dim,
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_semantic_hash(torch::Tensor sem_vec, torch::Tensor proj) {
  TORCH_CHECK(sem_vec.is_cuda() && proj.is_cuda(), "tile_semantic_hash expects CUDA tensors");
  TORCH_CHECK(sem_vec.scalar_type() == torch::kFloat32 && proj.scalar_type() == torch::kFloat32, "tile_semantic_hash only supports float32 tensors");
  TORCH_CHECK(sem_vec.is_contiguous() && proj.is_contiguous(), "tile_semantic_hash expects contiguous tensors");
  TORCH_CHECK(sem_vec.dim() == 2, "tile_semantic_hash expects semantic vectors shaped [B,D]");
  TORCH_CHECK(proj.dim() == 3, "tile_semantic_hash expects projection shaped [tables,planes,D]");
  TORCH_CHECK(sem_vec.size(0) > 0 && sem_vec.size(1) > 0, "tile_semantic_hash expects non-empty semantic vectors");
  TORCH_CHECK(proj.size(0) > 0 && proj.size(1) > 0, "tile_semantic_hash expects non-empty hash tables and planes");
  TORCH_CHECK(proj.size(2) == sem_vec.size(1), "tile_semantic_hash projection dim must match semantic vector dim");
  TORCH_CHECK(proj.size(1) <= 62, "tile_semantic_hash supports up to 62 planes per table");
  auto out = torch::empty({sem_vec.size(0), proj.size(0)}, sem_vec.options().dtype(torch::kInt64));
  launch_semantic_hash_int64(
      sem_vec.data_ptr<float>(),
      proj.data_ptr<float>(),
      out.data_ptr<std::int64_t>(),
      sem_vec.size(0),
      sem_vec.size(1),
      proj.size(0),
      proj.size(1),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> tile_topk_route(torch::Tensor logits, std::int64_t top_k) {
  TORCH_CHECK(logits.is_cuda(), "tile_topk_route expects a CUDA tensor");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32, "tile_topk_route only supports float32 tensors");
  TORCH_CHECK(logits.is_contiguous(), "tile_topk_route expects a contiguous tensor");
  TORCH_CHECK(logits.dim() >= 1, "tile_topk_route expects at least one dimension");
  TORCH_CHECK(logits.size(-1) > 0, "tile_topk_route expects a non-empty expert dimension");
  TORCH_CHECK(top_k >= 1 && top_k <= logits.size(-1), "tile_topk_route expects 1 <= top_k <= experts");
  TORCH_CHECK(top_k <= 64, "tile_topk_route supports top_k <= 64");
  const auto experts = logits.size(-1);
  const auto rows = logits.numel() / experts;
  auto out_shape = logits.sizes().vec();
  out_shape.back() = top_k;
  auto weights = torch::empty(out_shape, logits.options());
  auto indices = torch::empty(out_shape, logits.options().dtype(torch::kInt64));
  if (logits.numel() == 0) {
    return {weights, indices};
  }
  launch_topk_route_float32(
      logits.data_ptr<float>(),
      weights.data_ptr<float>(),
      indices.data_ptr<std::int64_t>(),
      rows,
      experts,
      top_k,
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {weights, indices};
}

torch::Tensor tile_attentionless_decoder(
    torch::Tensor bucket_indices,
    torch::Tensor expert_output,
    torch::Tensor bucket_embed,
    torch::Tensor out_weight) {
  TORCH_CHECK(bucket_indices.is_cuda() && expert_output.is_cuda() && bucket_embed.is_cuda() && out_weight.is_cuda(), "tile_attentionless_decoder expects CUDA tensors");
  TORCH_CHECK(bucket_indices.scalar_type() == torch::kInt64, "tile_attentionless_decoder expects int64 bucket indices");
  TORCH_CHECK(
      expert_output.scalar_type() == torch::kFloat32 && bucket_embed.scalar_type() == torch::kFloat32 && out_weight.scalar_type() == torch::kFloat32,
      "tile_attentionless_decoder only supports float32 tensors");
  TORCH_CHECK(bucket_indices.is_contiguous() && expert_output.is_contiguous() && bucket_embed.is_contiguous() && out_weight.is_contiguous(), "tile_attentionless_decoder expects contiguous tensors");
  TORCH_CHECK(bucket_indices.dim() == 1, "tile_attentionless_decoder expects primary bucket indices shaped [B]");
  TORCH_CHECK(expert_output.dim() == 2, "tile_attentionless_decoder expects expert output shaped [B,R]");
  TORCH_CHECK(bucket_embed.dim() == 2 && out_weight.dim() == 2, "tile_attentionless_decoder expects embedding [N,R] and output weight [V,R]");
  TORCH_CHECK(bucket_indices.size(0) == expert_output.size(0), "tile_attentionless_decoder batch dimensions must match");
  TORCH_CHECK(bucket_embed.size(0) > 0 && bucket_embed.size(1) > 0, "tile_attentionless_decoder expects non-empty bucket embedding");
  TORCH_CHECK(expert_output.size(1) == bucket_embed.size(1) && out_weight.size(1) == bucket_embed.size(1), "tile_attentionless_decoder residual dimensions must match");
  TORCH_CHECK(out_weight.size(0) > 0, "tile_attentionless_decoder expects non-empty vocabulary");
  auto out = torch::empty({expert_output.size(0), 1, out_weight.size(0)}, expert_output.options());
  if (expert_output.numel() == 0) {
    return out;
  }
  launch_attentionless_decoder_float32(
      bucket_indices.data_ptr<std::int64_t>(),
      expert_output.data_ptr<float>(),
      bucket_embed.data_ptr<float>(),
      out_weight.data_ptr<float>(),
      out.data_ptr<float>(),
      expert_output.size(0),
      expert_output.size(1),
      out_weight.size(0),
      bucket_embed.size(0),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_expert_bias_add(torch::Tensor logits, torch::Tensor bias) {
  TORCH_CHECK(logits.is_cuda() && bias.is_cuda(), "tile_expert_bias_add expects CUDA tensors");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32 && bias.scalar_type() == torch::kFloat32, "tile_expert_bias_add only supports float32 tensors");
  TORCH_CHECK(logits.is_contiguous() && bias.is_contiguous(), "tile_expert_bias_add expects contiguous tensors");
  TORCH_CHECK(logits.dim() >= 1 && bias.dim() == 1, "tile_expert_bias_add expects logits [...,E] and bias [E]");
  TORCH_CHECK(logits.size(-1) == bias.numel() && bias.numel() > 0, "tile_expert_bias_add expert dimensions must match");
  auto out = torch::empty_like(logits);
  if (logits.numel() == 0) {
    return out;
  }
  launch_expert_bias_add_float32(
      logits.data_ptr<float>(),
      bias.data_ptr<float>(),
      out.data_ptr<float>(),
      logits.numel(),
      bias.numel(),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_scaled_residual_add(torch::Tensor lhs, torch::Tensor rhs, torch::Tensor scale) {
  TORCH_CHECK(lhs.is_cuda() && rhs.is_cuda() && scale.is_cuda(), "tile_scaled_residual_add expects CUDA tensors");
  TORCH_CHECK(
      lhs.scalar_type() == torch::kFloat32 && rhs.scalar_type() == torch::kFloat32 && scale.scalar_type() == torch::kFloat32,
      "tile_scaled_residual_add only supports float32 tensors");
  TORCH_CHECK(lhs.is_contiguous() && rhs.is_contiguous() && scale.is_contiguous(), "tile_scaled_residual_add expects contiguous tensors");
  TORCH_CHECK(lhs.sizes() == rhs.sizes(), "tile_scaled_residual_add expects same-shaped inputs");
  TORCH_CHECK(scale.numel() == 1, "tile_scaled_residual_add expects a scalar scale tensor");
  auto out = torch::empty_like(lhs);
  if (lhs.numel() == 0) {
    return out;
  }
  launch_scaled_residual_add_float32(
      lhs.data_ptr<float>(),
      rhs.data_ptr<float>(),
      scale.data_ptr<float>(),
      out.data_ptr<float>(),
      lhs.numel(),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_linear(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, bool has_bias) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "tile_linear expects CUDA input and weight tensors");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32, "tile_linear only supports float32 input and weight");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "tile_linear expects contiguous input and weight");
  TORCH_CHECK(x.dim() >= 1 && weight.dim() == 2, "tile_linear expects x [..., input_dim] and weight [output_dim, input_dim]");
  TORCH_CHECK(x.size(-1) == weight.size(1), "tile_linear input dim must match weight");
  const float* bias_ptr = nullptr;
  if (has_bias) {
    TORCH_CHECK(bias.is_cuda(), "tile_linear expects CUDA bias tensor");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "tile_linear only supports float32 bias");
    TORCH_CHECK(bias.is_contiguous(), "tile_linear expects contiguous bias");
    TORCH_CHECK(bias.dim() == 1 && bias.numel() == weight.size(0), "tile_linear bias must match output dim");
    bias_ptr = bias.data_ptr<float>();
  }
  auto out_sizes = x.sizes().vec();
  out_sizes.back() = weight.size(0);
  auto out = torch::empty(out_sizes, x.options());
  if (out.numel() == 0) {
    return out;
  }
  const auto rows = x.numel() / x.size(-1);
  launch_linear_float32(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias_ptr,
      out.data_ptr<float>(),
      rows,
      x.size(-1),
      weight.size(0),
      has_bias,
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_act_weighted_sum(torch::Tensor states, torch::Tensor weights) {
  TORCH_CHECK(states.is_cuda() && weights.is_cuda(), "tile_act_weighted_sum expects CUDA tensors");
  TORCH_CHECK(states.scalar_type() == torch::kFloat32 && weights.scalar_type() == torch::kFloat32, "tile_act_weighted_sum only supports float32 tensors");
  TORCH_CHECK(states.is_contiguous() && weights.is_contiguous(), "tile_act_weighted_sum expects contiguous tensors");
  TORCH_CHECK(states.dim() >= 3 && weights.dim() == 2, "tile_act_weighted_sum expects states [B,steps,...] and weights [B,steps]");
  TORCH_CHECK(states.size(0) == weights.size(0) && states.size(1) == weights.size(1), "tile_act_weighted_sum batch/step axes must match");
  auto out_sizes = states.sizes().vec();
  out_sizes.erase(out_sizes.begin() + 1);
  auto out = torch::empty(out_sizes, states.options());
  if (out.numel() == 0) {
    return out;
  }
  const auto inner = states.numel() / (states.size(0) * states.size(1));
  launch_act_weighted_sum_float32(
      states.data_ptr<float>(),
      weights.data_ptr<float>(),
      out.data_ptr<float>(),
      states.size(0),
      states.size(1),
      inner,
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_latent_pool(torch::Tensor x, torch::Tensor mask_values) {
  TORCH_CHECK(x.is_cuda() && mask_values.is_cuda(), "tile_latent_pool expects CUDA tensors");
  TORCH_CHECK(
      x.scalar_type() == torch::kFloat32 && mask_values.scalar_type() == torch::kFloat32,
      "tile_latent_pool only supports float32 tensors");
  TORCH_CHECK(x.is_contiguous() && mask_values.is_contiguous(), "tile_latent_pool expects contiguous tensors");
  TORCH_CHECK(x.dim() == 3 && mask_values.dim() == 2, "tile_latent_pool expects x [B,S,D] and mask [B,S]");
  TORCH_CHECK(x.size(0) == mask_values.size(0) && x.size(1) == mask_values.size(1), "tile_latent_pool batch/sequence axes must match");
  TORCH_CHECK(x.numel() > 0, "tile_latent_pool expects non-empty x");
  auto out = torch::empty({x.size(0), x.size(2)}, x.options());
  launch_latent_pool_float32(
      x.data_ptr<float>(),
      mask_values.data_ptr<float>(),
      out.data_ptr<float>(),
      x.size(0),
      x.size(1),
      x.size(2),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_token_cross_entropy(torch::Tensor logits, torch::Tensor target_ids) {
  TORCH_CHECK(logits.is_cuda() && target_ids.is_cuda(), "tile_token_cross_entropy expects CUDA tensors");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32 && target_ids.scalar_type() == torch::kInt64, "tile_token_cross_entropy expects float32 logits and int64 targets");
  TORCH_CHECK(logits.is_contiguous() && target_ids.is_contiguous(), "tile_token_cross_entropy expects contiguous tensors");
  TORCH_CHECK(logits.dim() >= 2, "tile_token_cross_entropy expects logits with a vocab dimension");
  auto expected_target_sizes = logits.sizes().vec();
  expected_target_sizes.pop_back();
  TORCH_CHECK(target_ids.sizes().vec() == expected_target_sizes, "tile_token_cross_entropy target shape must match logits without vocab");
  TORCH_CHECK(logits.numel() > 0, "tile_token_cross_entropy expects non-empty logits");
  const auto vocab = logits.size(-1);
  const auto rows = logits.numel() / vocab;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto partials = torch::empty({(rows + 1023) / 1024}, logits.options());
  launch_token_cross_entropy_partials_float32(
      logits.data_ptr<float>(),
      target_ids.data_ptr<std::int64_t>(),
      partials.data_ptr<float>(),
      rows,
      vocab,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  while (partials.numel() > 1) {
    auto next = torch::empty({(partials.numel() + 1023) / 1024}, logits.options());
    launch_sum_partials_float32(partials.data_ptr<float>(), next.data_ptr<float>(), partials.numel(), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    partials = next;
  }
  auto out = torch::empty({}, logits.options());
  launch_scale_float32(partials.data_ptr<float>(), out.data_ptr<float>(), 1, 1.0f / static_cast<float>(rows), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_masked_token_cross_entropy(
    torch::Tensor logits,
    torch::Tensor target_ids,
    torch::Tensor loss_mask,
    std::int64_t ignore_index) {
  TORCH_CHECK(logits.is_cuda() && target_ids.is_cuda() && loss_mask.is_cuda(), "tile_masked_token_cross_entropy expects CUDA tensors");
  TORCH_CHECK(
      logits.scalar_type() == torch::kFloat32 && target_ids.scalar_type() == torch::kInt64 &&
          loss_mask.scalar_type() == torch::kFloat32,
      "tile_masked_token_cross_entropy expects float32 logits, int64 targets, and float32 mask");
  TORCH_CHECK(
      logits.is_contiguous() && target_ids.is_contiguous() && loss_mask.is_contiguous(),
      "tile_masked_token_cross_entropy expects contiguous tensors");
  TORCH_CHECK(logits.dim() >= 2, "tile_masked_token_cross_entropy expects logits with a vocab dimension");
  auto expected_target_sizes = logits.sizes().vec();
  expected_target_sizes.pop_back();
  TORCH_CHECK(target_ids.sizes().vec() == expected_target_sizes, "tile_masked_token_cross_entropy target shape must match logits without vocab");
  TORCH_CHECK(loss_mask.sizes() == target_ids.sizes(), "tile_masked_token_cross_entropy mask shape must match targets");
  TORCH_CHECK(logits.numel() > 0, "tile_masked_token_cross_entropy expects non-empty logits");
  const auto vocab = logits.size(-1);
  const auto rows = logits.numel() / vocab;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto loss_partials = torch::empty({(rows + 1023) / 1024}, logits.options());
  auto mask_partials = torch::empty_like(loss_partials);
  launch_masked_token_cross_entropy_partials_float32(
      logits.data_ptr<float>(),
      target_ids.data_ptr<std::int64_t>(),
      loss_mask.data_ptr<float>(),
      loss_partials.data_ptr<float>(),
      mask_partials.data_ptr<float>(),
      rows,
      vocab,
      ignore_index,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  while (loss_partials.numel() > 1) {
    auto next_loss = torch::empty({(loss_partials.numel() + 1023) / 1024}, logits.options());
    auto next_mask = torch::empty_like(next_loss);
    launch_sum_partials_float32(loss_partials.data_ptr<float>(), next_loss.data_ptr<float>(), loss_partials.numel(), stream);
    launch_sum_partials_float32(mask_partials.data_ptr<float>(), next_mask.data_ptr<float>(), mask_partials.numel(), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    loss_partials = next_loss;
    mask_partials = next_mask;
  }
  auto denom = torch::clamp_min(mask_partials, 1.0);
  return (loss_partials / denom).squeeze(0);
}

torch::Tensor tile_sequence_logp(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor loss_mask,
    std::int64_t ignore_index) {
  TORCH_CHECK(logits.is_cuda() && targets.is_cuda() && loss_mask.is_cuda(), "tile_sequence_logp expects CUDA tensors");
  TORCH_CHECK(
      logits.scalar_type() == torch::kFloat32 && targets.scalar_type() == torch::kInt64 &&
          loss_mask.scalar_type() == torch::kFloat32,
      "tile_sequence_logp expects float32 logits, int64 targets, and float32 mask");
  TORCH_CHECK(logits.is_contiguous() && targets.is_contiguous() && loss_mask.is_contiguous(), "tile_sequence_logp expects contiguous tensors");
  TORCH_CHECK(logits.dim() == 3, "tile_sequence_logp expects logits [B,S,V]");
  std::vector<std::int64_t> expected_target_sizes = {logits.size(0), logits.size(1)};
  TORCH_CHECK(targets.sizes().vec() == expected_target_sizes, "tile_sequence_logp targets must be [B,S]");
  TORCH_CHECK(loss_mask.sizes().vec() == expected_target_sizes, "tile_sequence_logp mask must be [B,S]");
  TORCH_CHECK(logits.size(0) > 0 && logits.size(0) <= 1024, "tile_sequence_logp supports batch 1..1024");
  auto out = torch::empty({logits.size(0)}, logits.options());
  launch_sequence_logp_float32(
      logits.data_ptr<float>(),
      targets.data_ptr<std::int64_t>(),
      loss_mask.data_ptr<float>(),
      out.data_ptr<float>(),
      logits.size(0),
      logits.size(1),
      logits.size(2),
      ignore_index,
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_preference_bce_loss(torch::Tensor reward_chosen, torch::Tensor reward_rejected) {
  TORCH_CHECK(reward_chosen.is_cuda() && reward_rejected.is_cuda(), "tile_preference_bce_loss expects CUDA tensors");
  TORCH_CHECK(
      reward_chosen.scalar_type() == torch::kFloat32 && reward_rejected.scalar_type() == torch::kFloat32,
      "tile_preference_bce_loss only supports float32 tensors");
  TORCH_CHECK(reward_chosen.is_contiguous() && reward_rejected.is_contiguous(), "tile_preference_bce_loss expects contiguous tensors");
  TORCH_CHECK(reward_chosen.sizes() == reward_rejected.sizes(), "tile_preference_bce_loss expects same-shape tensors");
  TORCH_CHECK(reward_chosen.numel() > 0, "tile_preference_bce_loss expects non-empty tensors");
  auto stream = at::cuda::getCurrentCUDAStream();
  auto partials = torch::empty({(reward_chosen.numel() + 1023) / 1024}, reward_chosen.options());
  launch_preference_bce_partials_float32(
      reward_chosen.data_ptr<float>(),
      reward_rejected.data_ptr<float>(),
      partials.data_ptr<float>(),
      reward_chosen.numel(),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  while (partials.numel() > 1) {
    auto next = torch::empty({(partials.numel() + 1023) / 1024}, reward_chosen.options());
    launch_sum_partials_float32(partials.data_ptr<float>(), next.data_ptr<float>(), partials.numel(), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    partials = next;
  }
  auto out = torch::empty({}, reward_chosen.options());
  launch_scale_float32(partials.data_ptr<float>(), out.data_ptr<float>(), 1, 1.0f / static_cast<float>(reward_chosen.numel()), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> tile_ppo_clipped_loss(
    torch::Tensor logp_new,
    torch::Tensor logp_old,
    torch::Tensor advantages,
    torch::Tensor value_new,
    torch::Tensor value_old,
    torch::Tensor returns,
    double clip_range,
    double vf_coef) {
  TORCH_CHECK(
      logp_new.is_cuda() && logp_old.is_cuda() && advantages.is_cuda() && value_new.is_cuda() &&
          value_old.is_cuda() && returns.is_cuda(),
      "tile_ppo_clipped_loss expects CUDA tensors");
  TORCH_CHECK(
      logp_new.scalar_type() == torch::kFloat32 && logp_old.scalar_type() == torch::kFloat32 &&
          advantages.scalar_type() == torch::kFloat32 && value_new.scalar_type() == torch::kFloat32 &&
          value_old.scalar_type() == torch::kFloat32 && returns.scalar_type() == torch::kFloat32,
      "tile_ppo_clipped_loss only supports float32 tensors");
  TORCH_CHECK(
      logp_new.is_contiguous() && logp_old.is_contiguous() && advantages.is_contiguous() &&
          value_new.is_contiguous() && value_old.is_contiguous() && returns.is_contiguous(),
      "tile_ppo_clipped_loss expects contiguous tensors");
  TORCH_CHECK(
      logp_old.sizes() == logp_new.sizes() && advantages.sizes() == logp_new.sizes() &&
          value_new.sizes() == logp_new.sizes() && value_old.sizes() == logp_new.sizes() &&
          returns.sizes() == logp_new.sizes(),
      "tile_ppo_clipped_loss expects same-shape tensors");
  TORCH_CHECK(logp_new.numel() > 0, "tile_ppo_clipped_loss expects non-empty tensors");
  auto stream = at::cuda::getCurrentCUDAStream();
  auto policy_partials = torch::empty({(logp_new.numel() + 1023) / 1024}, logp_new.options());
  auto value_partials = torch::empty_like(policy_partials);
  launch_ppo_clipped_loss_partials_float32(
      logp_new.data_ptr<float>(),
      logp_old.data_ptr<float>(),
      advantages.data_ptr<float>(),
      value_new.data_ptr<float>(),
      value_old.data_ptr<float>(),
      returns.data_ptr<float>(),
      policy_partials.data_ptr<float>(),
      value_partials.data_ptr<float>(),
      logp_new.numel(),
      static_cast<float>(clip_range),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  while (policy_partials.numel() > 1) {
    auto next_policy = torch::empty({(policy_partials.numel() + 1023) / 1024}, logp_new.options());
    auto next_value = torch::empty_like(next_policy);
    launch_sum_partials_float32(policy_partials.data_ptr<float>(), next_policy.data_ptr<float>(), policy_partials.numel(), stream);
    launch_sum_partials_float32(value_partials.data_ptr<float>(), next_value.data_ptr<float>(), value_partials.numel(), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    policy_partials = next_policy;
    value_partials = next_value;
  }
  auto policy_loss = torch::empty({}, logp_new.options());
  auto value_loss = torch::empty({}, logp_new.options());
  launch_scale_float32(policy_partials.data_ptr<float>(), policy_loss.data_ptr<float>(), 1, 1.0f / static_cast<float>(logp_new.numel()), stream);
  launch_scale_float32(value_partials.data_ptr<float>(), value_loss.data_ptr<float>(), 1, 1.0f / static_cast<float>(logp_new.numel()), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto loss = policy_loss + static_cast<float>(vf_coef) * value_loss;
  return {policy_loss, value_loss, loss};
}

std::vector<torch::Tensor> tile_gae_compute(
    torch::Tensor rewards,
    torch::Tensor values,
    double gamma,
    double lambda_value) {
  TORCH_CHECK(rewards.is_cuda() && values.is_cuda(), "tile_gae_compute expects CUDA tensors");
  TORCH_CHECK(rewards.scalar_type() == torch::kFloat32 && values.scalar_type() == torch::kFloat32, "tile_gae_compute only supports float32 tensors");
  TORCH_CHECK(rewards.is_contiguous() && values.is_contiguous(), "tile_gae_compute expects contiguous tensors");
  TORCH_CHECK(rewards.sizes() == values.sizes(), "tile_gae_compute expects same-shape tensors");
  TORCH_CHECK(rewards.dim() == 2, "tile_gae_compute expects rewards and values shaped [B,S]");
  TORCH_CHECK(rewards.size(0) > 0 && rewards.size(1) > 0, "tile_gae_compute expects non-empty tensors");
  auto advantages = torch::empty_like(rewards);
  auto returns = torch::empty_like(rewards);
  launch_gae_compute_float32(
      rewards.data_ptr<float>(),
      values.data_ptr<float>(),
      advantages.data_ptr<float>(),
      returns.data_ptr<float>(),
      rewards.size(0),
      rewards.size(1),
      static_cast<float>(gamma),
      static_cast<float>(lambda_value),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {advantages, returns};
}

std::vector<torch::Tensor> tile_dpo_pairwise_loss(
    torch::Tensor policy_logp_chosen,
    torch::Tensor policy_logp_rejected,
    torch::Tensor ref_logp_chosen,
    torch::Tensor ref_logp_rejected,
    double beta,
    double label_smoothing,
    std::int64_t loss_type) {
  TORCH_CHECK(
      policy_logp_chosen.is_cuda() && policy_logp_rejected.is_cuda() && ref_logp_chosen.is_cuda() &&
          ref_logp_rejected.is_cuda(),
      "tile_dpo_pairwise_loss expects CUDA tensors");
  TORCH_CHECK(
      policy_logp_chosen.scalar_type() == torch::kFloat32 && policy_logp_rejected.scalar_type() == torch::kFloat32 &&
          ref_logp_chosen.scalar_type() == torch::kFloat32 && ref_logp_rejected.scalar_type() == torch::kFloat32,
      "tile_dpo_pairwise_loss only supports float32 tensors");
  TORCH_CHECK(
      policy_logp_chosen.is_contiguous() && policy_logp_rejected.is_contiguous() && ref_logp_chosen.is_contiguous() &&
          ref_logp_rejected.is_contiguous(),
      "tile_dpo_pairwise_loss expects contiguous tensors");
  TORCH_CHECK(
      policy_logp_chosen.sizes() == policy_logp_rejected.sizes() &&
          policy_logp_chosen.sizes() == ref_logp_chosen.sizes() &&
          policy_logp_chosen.sizes() == ref_logp_rejected.sizes(),
      "tile_dpo_pairwise_loss expects same-shape tensors");
  TORCH_CHECK(policy_logp_chosen.numel() > 0, "tile_dpo_pairwise_loss expects non-empty tensors");
  TORCH_CHECK(loss_type >= 0 && loss_type <= 2, "tile_dpo_pairwise_loss loss_type must be 0, 1, or 2");
  float beta_value = static_cast<float>(beta);
  if (loss_type == 2 && std::abs(beta_value) < 1.0e-8f) {
    beta_value = 1.0e-8f;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  auto partials = torch::empty({(policy_logp_chosen.numel() + 1023) / 1024}, policy_logp_chosen.options());
  auto chosen_reward = torch::empty_like(policy_logp_chosen);
  auto rejected_reward = torch::empty_like(policy_logp_rejected);
  launch_dpo_pairwise_partials_float32(
      policy_logp_chosen.data_ptr<float>(),
      policy_logp_rejected.data_ptr<float>(),
      ref_logp_chosen.data_ptr<float>(),
      ref_logp_rejected.data_ptr<float>(),
      partials.data_ptr<float>(),
      chosen_reward.data_ptr<float>(),
      rejected_reward.data_ptr<float>(),
      policy_logp_chosen.numel(),
      beta_value,
      static_cast<float>(label_smoothing),
      static_cast<int>(loss_type),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  while (partials.numel() > 1) {
    auto next = torch::empty({(partials.numel() + 1023) / 1024}, policy_logp_chosen.options());
    launch_sum_partials_float32(partials.data_ptr<float>(), next.data_ptr<float>(), partials.numel(), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    partials = next;
  }
  auto loss = torch::empty({}, policy_logp_chosen.options());
  launch_scale_float32(partials.data_ptr<float>(), loss.data_ptr<float>(), 1, 1.0f / static_cast<float>(policy_logp_chosen.numel()), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {loss, chosen_reward, rejected_reward};
}

torch::Tensor tile_route_selection_loss(
    torch::Tensor route_logits,
    torch::Tensor sem_targets,
    std::int64_t num_vocab_dims,
    std::int64_t shared_experts,
    std::int64_t ignore_index) {
  TORCH_CHECK(route_logits.is_cuda() && sem_targets.is_cuda(), "tile_route_selection_loss expects CUDA tensors");
  TORCH_CHECK(route_logits.scalar_type() == torch::kFloat32 && sem_targets.scalar_type() == torch::kInt64, "tile_route_selection_loss expects float32 logits and int64 targets");
  TORCH_CHECK(route_logits.is_contiguous() && sem_targets.is_contiguous(), "tile_route_selection_loss expects contiguous tensors");
  TORCH_CHECK(route_logits.dim() == 3, "tile_route_selection_loss expects route_logits shaped [B,S,E]");
  TORCH_CHECK(sem_targets.dim() == 2, "tile_route_selection_loss expects sem_targets shaped [B,D]");
  TORCH_CHECK(route_logits.size(0) == sem_targets.size(0), "tile_route_selection_loss batch dimensions must match");
  TORCH_CHECK(num_vocab_dims > 0, "tile_route_selection_loss expects num_vocab_dims > 0");
  TORCH_CHECK(sem_targets.size(1) >= num_vocab_dims, "tile_route_selection_loss sem_targets must include all semantic dims");
  TORCH_CHECK(route_logits.size(2) >= shared_experts + num_vocab_dims, "tile_route_selection_loss route logits must include semantic expert slice");
  TORCH_CHECK(route_logits.numel() > 0, "tile_route_selection_loss expects non-empty logits");
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto n = route_logits.size(0) * route_logits.size(1) * num_vocab_dims;
  auto partials = torch::empty({(n + 1023) / 1024}, route_logits.options());
  auto counts = torch::empty_like(partials);
  launch_route_selection_loss_partials_float32(
      route_logits.data_ptr<float>(),
      sem_targets.data_ptr<std::int64_t>(),
      partials.data_ptr<float>(),
      counts.data_ptr<float>(),
      n,
      route_logits.size(1),
      route_logits.size(2),
      num_vocab_dims,
      shared_experts,
      ignore_index,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  while (partials.numel() > 1) {
    auto next_partials = torch::empty({(partials.numel() + 1023) / 1024}, route_logits.options());
    auto next_counts = torch::empty_like(next_partials);
    launch_sum_partials_float32(partials.data_ptr<float>(), next_partials.data_ptr<float>(), partials.numel(), stream);
    launch_sum_partials_float32(counts.data_ptr<float>(), next_counts.data_ptr<float>(), counts.numel(), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    partials = next_partials;
    counts = next_counts;
  }
  auto denom = torch::clamp_min(counts, 1.0);
  return (partials / denom).squeeze(0);
}

torch::Tensor tile_route_balance_loss(torch::Tensor route_logits) {
  TORCH_CHECK(route_logits.is_cuda(), "tile_route_balance_loss expects a CUDA tensor");
  TORCH_CHECK(route_logits.scalar_type() == torch::kFloat32, "tile_route_balance_loss only supports float32 tensors");
  TORCH_CHECK(route_logits.is_contiguous(), "tile_route_balance_loss expects a contiguous tensor");
  TORCH_CHECK(route_logits.dim() >= 1, "tile_route_balance_loss expects logits with an expert dimension");
  TORCH_CHECK(route_logits.numel() > 0, "tile_route_balance_loss expects non-empty logits");
  const auto experts = route_logits.size(-1);
  const auto rows = route_logits.numel() / experts;
  TORCH_CHECK(experts > 0 && experts <= 1024, "tile_route_balance_loss supports 1..1024 experts");
  TORCH_CHECK(rows > 0 && rows <= 1024, "tile_route_balance_loss supports 1..1024 flattened rows");
  auto stream = at::cuda::getCurrentCUDAStream();
  auto density = torch::empty({experts}, route_logits.options());
  auto out = torch::empty({}, route_logits.options());
  launch_route_balance_density_float32(
      route_logits.data_ptr<float>(),
      density.data_ptr<float>(),
      rows,
      experts,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  launch_route_balance_loss_float32(density.data_ptr<float>(), out.data_ptr<float>(), experts, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_softmax_distillation_loss(torch::Tensor teacher_logits, torch::Tensor student_logits) {
  TORCH_CHECK(teacher_logits.is_cuda() && student_logits.is_cuda(), "tile_softmax_distillation_loss expects CUDA tensors");
  TORCH_CHECK(
      teacher_logits.scalar_type() == torch::kFloat32 && student_logits.scalar_type() == torch::kFloat32,
      "tile_softmax_distillation_loss only supports float32 tensors");
  TORCH_CHECK(
      teacher_logits.is_contiguous() && student_logits.is_contiguous(),
      "tile_softmax_distillation_loss expects contiguous tensors");
  TORCH_CHECK(teacher_logits.sizes() == student_logits.sizes(), "tile_softmax_distillation_loss expects same-shape tensors");
  TORCH_CHECK(teacher_logits.dim() >= 2, "tile_softmax_distillation_loss expects at least batch and vocab dimensions");
  TORCH_CHECK(teacher_logits.numel() > 0, "tile_softmax_distillation_loss expects non-empty logits");
  const auto vocab = teacher_logits.size(-1);
  const auto rows = teacher_logits.numel() / vocab;
  const auto batch = teacher_logits.size(0);
  TORCH_CHECK(vocab > 0 && vocab <= 1024, "tile_softmax_distillation_loss supports 1..1024 vocab width");
  auto stream = at::cuda::getCurrentCUDAStream();
  auto partials = torch::empty({rows}, teacher_logits.options());
  launch_softmax_distillation_partials_float32(
      teacher_logits.data_ptr<float>(),
      student_logits.data_ptr<float>(),
      partials.data_ptr<float>(),
      rows,
      vocab,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  while (partials.numel() > 1) {
    auto next = torch::empty({(partials.numel() + 1023) / 1024}, teacher_logits.options());
    launch_sum_partials_float32(partials.data_ptr<float>(), next.data_ptr<float>(), partials.numel(), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    partials = next;
  }
  auto out = torch::empty({}, teacher_logits.options());
  launch_scale_float32(partials.data_ptr<float>(), out.data_ptr<float>(), 1, 1.0f / static_cast<float>(batch), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_ema_update(torch::Tensor target, torch::Tensor source, double decay) {
  TORCH_CHECK(target.is_cuda() && source.is_cuda(), "tile_ema_update expects CUDA tensors");
  TORCH_CHECK(target.scalar_type() == torch::kFloat32 && source.scalar_type() == torch::kFloat32, "tile_ema_update only supports float32 tensors");
  TORCH_CHECK(target.is_contiguous() && source.is_contiguous(), "tile_ema_update expects contiguous tensors");
  TORCH_CHECK(target.sizes() == source.sizes(), "tile_ema_update expects same-shape tensors");
  TORCH_CHECK(target.numel() > 0, "tile_ema_update expects non-empty tensors");
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_ema_update_float32(
      target.data_ptr<float>(),
      source.data_ptr<float>(),
      target.numel(),
      static_cast<float>(decay),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return target;
}

torch::Tensor tile_gradient_accumulate(torch::Tensor buffer, torch::Tensor grad, double scale) {
  TORCH_CHECK(buffer.is_cuda() && grad.is_cuda(), "tile_gradient_accumulate expects CUDA tensors");
  TORCH_CHECK(buffer.scalar_type() == torch::kFloat32 && grad.scalar_type() == torch::kFloat32, "tile_gradient_accumulate only supports float32 tensors");
  TORCH_CHECK(buffer.is_contiguous() && grad.is_contiguous(), "tile_gradient_accumulate expects contiguous tensors");
  TORCH_CHECK(buffer.sizes() == grad.sizes(), "tile_gradient_accumulate expects same-shape tensors");
  TORCH_CHECK(buffer.numel() > 0, "tile_gradient_accumulate expects non-empty tensors");
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_gradient_accumulate_float32(
      buffer.data_ptr<float>(),
      grad.data_ptr<float>(),
      buffer.numel(),
      static_cast<float>(scale),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return buffer;
}

torch::Tensor tile_gradient_clip_norm(std::vector<torch::Tensor> grads, double max_norm, double eps) {
  TORCH_CHECK(!grads.empty(), "tile_gradient_clip_norm expects at least one tensor");
  auto options = grads[0].options();
  TORCH_CHECK(grads[0].is_cuda(), "tile_gradient_clip_norm expects CUDA tensors");
  TORCH_CHECK(grads[0].scalar_type() == torch::kFloat32, "tile_gradient_clip_norm only supports float32 tensors");
  auto stream = at::cuda::getCurrentCUDAStream();
  auto total = torch::zeros({}, options);
  for (const auto& grad : grads) {
    TORCH_CHECK(grad.is_cuda(), "tile_gradient_clip_norm expects CUDA tensors");
    TORCH_CHECK(grad.scalar_type() == torch::kFloat32, "tile_gradient_clip_norm only supports float32 tensors");
    TORCH_CHECK(grad.is_contiguous(), "tile_gradient_clip_norm expects contiguous tensors");
    TORCH_CHECK(grad.numel() > 0, "tile_gradient_clip_norm expects non-empty tensors");
    auto partials = torch::empty({(grad.numel() + 1023) / 1024}, grad.options());
    launch_sumsq_partials_float32(grad.data_ptr<float>(), partials.data_ptr<float>(), grad.numel(), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    while (partials.numel() > 1) {
      auto next = torch::empty({(partials.numel() + 1023) / 1024}, grad.options());
      launch_sum_partials_float32(partials.data_ptr<float>(), next.data_ptr<float>(), partials.numel(), stream);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      partials = next;
    }
    total = total + partials.squeeze(0);
  }
  auto total_norm = torch::sqrt(total);
  const float norm_value = total_norm.item<float>();
  const float scale = std::min(1.0f, static_cast<float>(max_norm) / (norm_value + static_cast<float>(eps)));
  if (scale < 1.0f) {
    for (auto& grad : grads) {
      launch_scale_inplace_float32(grad.data_ptr<float>(), grad.numel(), scale, stream);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
  return total_norm;
}

torch::Tensor tile_scaled_dot_product_attention(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool is_causal,
    bool right_align_causal,
    bool use_sparse_rules,
    std::int64_t window,
    std::int64_t num_sinks,
    std::int64_t block_size,
    std::int64_t compress_stride) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "tile_scaled_dot_product_attention expects CUDA tensors");
  TORCH_CHECK(
      q.scalar_type() == torch::kFloat32 && k.scalar_type() == torch::kFloat32 && v.scalar_type() == torch::kFloat32,
      "tile_scaled_dot_product_attention only supports float32 tensors");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "tile_scaled_dot_product_attention expects contiguous tensors");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "tile_scaled_dot_product_attention expects [B,H,S,D] tensors");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "tile_scaled_dot_product_attention batch dimensions must match");
  TORCH_CHECK(k.size(1) == v.size(1), "tile_scaled_dot_product_attention key/value head counts must match");
  TORCH_CHECK(q.size(1) % k.size(1) == 0, "tile_scaled_dot_product_attention query heads must be divisible by key heads");
  TORCH_CHECK(k.size(2) == v.size(2), "tile_scaled_dot_product_attention key/value sequence lengths must match");
  TORCH_CHECK(q.size(3) == k.size(3), "tile_scaled_dot_product_attention query/key head dimensions must match");
  TORCH_CHECK(k.size(2) > 0 && k.size(2) <= 1024, "tile_scaled_dot_product_attention supports 1 <= key sequence <= 1024");
  TORCH_CHECK(q.numel() > 0, "tile_scaled_dot_product_attention expects non-empty q");
  auto out = torch::empty({q.size(0), q.size(1), q.size(2), v.size(3)}, q.options());
  const float scale = 1.0f / std::sqrt(static_cast<float>(q.size(3)));
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_scaled_dot_product_attention_float32(
      q.data_ptr<float>(),
      k.data_ptr<float>(),
      v.data_ptr<float>(),
      out.data_ptr<float>(),
      out.numel(),
      q.size(1),
      k.size(1),
      q.size(2),
      k.size(2),
      q.size(3),
      v.size(3),
      scale,
      is_causal,
      right_align_causal,
      use_sparse_rules,
      window,
      num_sinks,
      block_size,
      compress_stride,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_random_timesteps(torch::Tensor tokens, std::int64_t counter) {
  TORCH_CHECK(tokens.is_cuda(), "tile_random_timesteps expects a CUDA tensor");
  TORCH_CHECK(tokens.dim() >= 1, "tile_random_timesteps expects tokens with a batch dimension");
  TORCH_CHECK(tokens.size(0) > 0, "tile_random_timesteps expects a non-empty batch");
  auto out = torch::empty({tokens.size(0)}, tokens.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_random_timesteps_float32(out.data_ptr<float>(), tokens.size(0), counter, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor tile_mask_scheduler(torch::Tensor tokens, torch::Tensor timesteps, std::int64_t mask_token_id, std::int64_t counter) {
  TORCH_CHECK(tokens.is_cuda() && timesteps.is_cuda(), "tile_mask_scheduler expects CUDA tensors");
  TORCH_CHECK(tokens.scalar_type() == torch::kInt64 && timesteps.scalar_type() == torch::kFloat32, "tile_mask_scheduler expects int64 tokens and float32 timesteps");
  TORCH_CHECK(tokens.is_contiguous() && timesteps.is_contiguous(), "tile_mask_scheduler expects contiguous tensors");
  TORCH_CHECK(tokens.dim() == 2 && timesteps.dim() == 1, "tile_mask_scheduler expects tokens [B,S] and timesteps [B]");
  TORCH_CHECK(tokens.size(0) == timesteps.size(0), "tile_mask_scheduler batch dimensions must match");
  auto out = torch::empty_like(tokens);
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_mask_scheduler_int64(
      tokens.data_ptr<std::int64_t>(),
      timesteps.data_ptr<float>(),
      out.data_ptr<std::int64_t>(),
      tokens.numel(),
      tokens.size(1),
      mask_token_id,
      counter,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> tile_jepa_mask(
    torch::Tensor tokens,
    double mask_ratio,
    std::int64_t mask_token_id,
    std::string mask_strategy,
    std::int64_t num_blocks,
    double min_block_ratio,
    double max_block_ratio,
    std::int64_t counter) {
  TORCH_CHECK(tokens.is_cuda(), "tile_jepa_mask expects a CUDA tensor");
  TORCH_CHECK(tokens.scalar_type() == torch::kInt64, "tile_jepa_mask expects int64 tokens");
  TORCH_CHECK(tokens.is_contiguous(), "tile_jepa_mask expects contiguous tokens");
  TORCH_CHECK(tokens.dim() == 2, "tile_jepa_mask expects tokens [B,S]");
  auto masked = torch::empty_like(tokens);
  auto mask = torch::empty(tokens.sizes(), tokens.options().dtype(torch::kFloat32));
  const int strategy = mask_strategy == "block" ? 1 : 0;
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_jepa_mask_int64(
      tokens.data_ptr<std::int64_t>(),
      masked.data_ptr<std::int64_t>(),
      mask.data_ptr<float>(),
      tokens.numel(),
      tokens.size(1),
      static_cast<float>(mask_ratio),
      mask_token_id,
      strategy,
      num_blocks,
      static_cast<float>(min_block_ratio),
      static_cast<float>(max_block_ratio),
      counter,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {masked, mask};
}

std::vector<torch::Tensor> tile_adamw_step(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    std::int64_t step) {
  TORCH_CHECK(param.is_cuda() && grad.is_cuda() && exp_avg.is_cuda() && exp_avg_sq.is_cuda(), "tile_adamw_step expects CUDA tensors");
  TORCH_CHECK(
      param.scalar_type() == torch::kFloat32 && grad.scalar_type() == torch::kFloat32 &&
      exp_avg.scalar_type() == torch::kFloat32 && exp_avg_sq.scalar_type() == torch::kFloat32,
      "tile_adamw_step only supports float32 tensors");
  TORCH_CHECK(
      param.is_contiguous() && grad.is_contiguous() && exp_avg.is_contiguous() && exp_avg_sq.is_contiguous(),
      "tile_adamw_step expects contiguous tensors");
  TORCH_CHECK(param.sizes() == grad.sizes() && param.sizes() == exp_avg.sizes() && param.sizes() == exp_avg_sq.sizes(), "tile_adamw_step expects same-shape tensors");
  TORCH_CHECK(param.numel() > 0, "tile_adamw_step expects non-empty tensors");
  TORCH_CHECK(step > 0, "tile_adamw_step expects step > 0");
  const float bias_correction1 = 1.0f - std::pow(static_cast<float>(beta1), static_cast<float>(step));
  const float bias_correction2 = 1.0f - std::pow(static_cast<float>(beta2), static_cast<float>(step));
  const float sqrt_bias_correction2 = std::sqrt(bias_correction2);
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_adamw_step_float32(
      param.data_ptr<float>(),
      grad.data_ptr<float>(),
      exp_avg.data_ptr<float>(),
      exp_avg_sq.data_ptr<float>(),
      param.numel(),
      static_cast<float>(lr),
      static_cast<float>(beta1),
      static_cast<float>(beta2),
      static_cast<float>(eps),
      static_cast<float>(weight_decay),
      bias_correction1,
      sqrt_bias_correction2,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {param, exp_avg, exp_avg_sq};
}

std::vector<torch::Tensor> tile_adamw_step_batch(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    std::int64_t step) {
  TORCH_CHECK(!params.empty(), "tile_adamw_step_batch expects at least one tensor");
  TORCH_CHECK(
      params.size() == grads.size() && params.size() == exp_avgs.size() && params.size() == exp_avg_sqs.size(),
      "tile_adamw_step_batch expects equally sized tensor lists");
  TORCH_CHECK(step > 0, "tile_adamw_step_batch expects step > 0");
  const float bias_correction1 = 1.0f - std::pow(static_cast<float>(beta1), static_cast<float>(step));
  const float bias_correction2 = 1.0f - std::pow(static_cast<float>(beta2), static_cast<float>(step));
  const float sqrt_bias_correction2 = std::sqrt(bias_correction2);
  auto stream = at::cuda::getCurrentCUDAStream();
  for (std::size_t i = 0; i < params.size(); ++i) {
    auto& param = params[i];
    auto& grad = grads[i];
    auto& exp_avg = exp_avgs[i];
    auto& exp_avg_sq = exp_avg_sqs[i];
    TORCH_CHECK(param.is_cuda() && grad.is_cuda() && exp_avg.is_cuda() && exp_avg_sq.is_cuda(), "tile_adamw_step_batch expects CUDA tensors");
    TORCH_CHECK(
        param.scalar_type() == torch::kFloat32 && grad.scalar_type() == torch::kFloat32 &&
        exp_avg.scalar_type() == torch::kFloat32 && exp_avg_sq.scalar_type() == torch::kFloat32,
        "tile_adamw_step_batch only supports float32 tensors");
    TORCH_CHECK(
        param.is_contiguous() && grad.is_contiguous() && exp_avg.is_contiguous() && exp_avg_sq.is_contiguous(),
        "tile_adamw_step_batch expects contiguous tensors");
    TORCH_CHECK(
        param.sizes() == grad.sizes() && param.sizes() == exp_avg.sizes() && param.sizes() == exp_avg_sq.sizes(),
        "tile_adamw_step_batch expects same-shape tensors");
    TORCH_CHECK(param.numel() > 0, "tile_adamw_step_batch expects non-empty tensors");
    launch_adamw_step_float32(
        param.data_ptr<float>(),
        grad.data_ptr<float>(),
        exp_avg.data_ptr<float>(),
        exp_avg_sq.data_ptr<float>(),
        param.numel(),
        static_cast<float>(lr),
        static_cast<float>(beta1),
        static_cast<float>(beta2),
        static_cast<float>(eps),
        static_cast<float>(weight_decay),
        bias_correction1,
        sqrt_bias_correction2,
        stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return params;
}

}  // namespace neuralfn::tile_cuda

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("extension_loaded", &neuralfn::tile_cuda::extension_loaded);
  m.def("tile_unary", &neuralfn::tile_cuda::tile_unary);
  m.def("tile_binary", &neuralfn::tile_cuda::tile_binary);
  m.def("tile_binary_pair", &neuralfn::tile_cuda::tile_binary_pair);
  m.def("tile_scalar_unary", &neuralfn::tile_cuda::tile_scalar_unary);
  m.def("tile_scalar_binary", &neuralfn::tile_cuda::tile_scalar_binary);
  m.def("tile_scalar_ternary", &neuralfn::tile_cuda::tile_scalar_ternary);
  m.def("tile_vector_binary", &neuralfn::tile_cuda::tile_vector_binary);
  m.def("tile_qk_gain", &neuralfn::tile_cuda::tile_qk_gain);
  m.def("tile_dyt", &neuralfn::tile_cuda::tile_dyt);
  m.def("tile_reshape_heads", &neuralfn::tile_cuda::tile_reshape_heads);
  m.def("tile_merge_heads", &neuralfn::tile_cuda::tile_merge_heads);
  m.def("tile_repeat_kv", &neuralfn::tile_cuda::tile_repeat_kv);
  m.def("tile_broadcast_expert_routes", &neuralfn::tile_cuda::tile_broadcast_expert_routes);
  m.def("tile_broadcast_chunk_routes", &neuralfn::tile_cuda::tile_broadcast_chunk_routes);
  m.def("tile_byte_patch_merge", &neuralfn::tile_cuda::tile_byte_patch_merge);
  m.def("tile_byte_patch_embed", &neuralfn::tile_cuda::tile_byte_patch_embed);
  m.def("tile_causal_chunk_state", &neuralfn::tile_cuda::tile_causal_chunk_state);
  m.def("tile_causal_chunk_state_backward", &neuralfn::tile_cuda::tile_causal_chunk_state_backward);
  m.def("tile_latent_mse_loss", &neuralfn::tile_cuda::tile_latent_mse_loss);
  m.def("tile_semantic_alignment_loss", &neuralfn::tile_cuda::tile_semantic_alignment_loss);
  m.def("tile_kv_cache_read", &neuralfn::tile_cuda::tile_kv_cache_read);
  m.def("tile_kv_quant_pack", &neuralfn::tile_cuda::tile_kv_quant_pack);
  m.def("tile_kv_quant_unpack", &neuralfn::tile_cuda::tile_kv_quant_unpack);
  m.def("tile_absolute_position_embedding", &neuralfn::tile_cuda::tile_absolute_position_embedding);
  m.def("tile_token_embedding", &neuralfn::tile_cuda::tile_token_embedding);
  m.def("tile_rotary_embedding", &neuralfn::tile_cuda::tile_rotary_embedding);
  m.def("tile_rms_norm", &neuralfn::tile_cuda::tile_rms_norm);
  m.def("tile_layer_norm", &neuralfn::tile_cuda::tile_layer_norm);
  m.def("tile_group_norm", &neuralfn::tile_cuda::tile_group_norm);
  m.def("tile_softmax_lastdim", &neuralfn::tile_cuda::tile_softmax_lastdim);
  m.def("tile_semantic_hash", &neuralfn::tile_cuda::tile_semantic_hash);
  m.def("tile_topk_route", &neuralfn::tile_cuda::tile_topk_route);
  m.def("tile_attentionless_decoder", &neuralfn::tile_cuda::tile_attentionless_decoder);
  m.def("tile_expert_bias_add", &neuralfn::tile_cuda::tile_expert_bias_add);
  m.def("tile_scaled_residual_add", &neuralfn::tile_cuda::tile_scaled_residual_add);
  m.def("tile_linear", &neuralfn::tile_cuda::tile_linear);
  m.def("tile_act_weighted_sum", &neuralfn::tile_cuda::tile_act_weighted_sum);
  m.def("tile_latent_pool", &neuralfn::tile_cuda::tile_latent_pool);
  m.def("tile_token_cross_entropy", &neuralfn::tile_cuda::tile_token_cross_entropy);
  m.def("tile_masked_token_cross_entropy", &neuralfn::tile_cuda::tile_masked_token_cross_entropy);
  m.def("tile_sequence_logp", &neuralfn::tile_cuda::tile_sequence_logp);
  m.def("tile_dpo_pairwise_loss", &neuralfn::tile_cuda::tile_dpo_pairwise_loss);
  m.def("tile_preference_bce_loss", &neuralfn::tile_cuda::tile_preference_bce_loss);
  m.def("tile_ppo_clipped_loss", &neuralfn::tile_cuda::tile_ppo_clipped_loss);
  m.def("tile_gae_compute", &neuralfn::tile_cuda::tile_gae_compute);
  m.def("tile_route_selection_loss", &neuralfn::tile_cuda::tile_route_selection_loss);
  m.def("tile_route_balance_loss", &neuralfn::tile_cuda::tile_route_balance_loss);
  m.def("tile_softmax_distillation_loss", &neuralfn::tile_cuda::tile_softmax_distillation_loss);
  m.def("tile_scaled_dot_product_attention", &neuralfn::tile_cuda::tile_scaled_dot_product_attention);
  m.def("tile_random_timesteps", &neuralfn::tile_cuda::tile_random_timesteps);
  m.def("tile_mask_scheduler", &neuralfn::tile_cuda::tile_mask_scheduler);
  m.def("tile_jepa_mask", &neuralfn::tile_cuda::tile_jepa_mask);
  m.def("tile_ema_update", &neuralfn::tile_cuda::tile_ema_update);
  m.def("tile_gradient_accumulate", &neuralfn::tile_cuda::tile_gradient_accumulate);
  m.def("tile_gradient_clip_norm", &neuralfn::tile_cuda::tile_gradient_clip_norm);
  m.def("tile_adamw_step", &neuralfn::tile_cuda::tile_adamw_step);
  m.def("tile_adamw_step_batch", &neuralfn::tile_cuda::tile_adamw_step_batch);
}
