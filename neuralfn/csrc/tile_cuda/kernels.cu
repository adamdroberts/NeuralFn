#include <cuda_tile.h>

#include <cmath>
#include <cstdint>

namespace neuralfn::tile_cuda {

namespace {

constexpr int kTileSize = 1024;

__tile_global__ void unary_float32_kernel(const float* __restrict__ x, float* __restrict__ out, std::int64_t n, int op) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  auto x_tile = ct::partition_view{ct::tensor_span{x, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto zero = ct::full<decltype(x_tile)>(0.0f);
  auto one = ct::full<decltype(x_tile)>(1.0f);
  auto result = x_tile;

  if (op == 0) {
    result = x_tile;
  } else if (op == 1) {
    result = -x_tile;
  } else if (op == 2) {
    result = ct::select(x_tile > zero, x_tile, zero);
  } else if (op == 3) {
    result = one / (one + ct::exp(-x_tile));
  } else if (op == 4) {
    result = ct::tanh(x_tile);
  } else if (op == 5) {
    result = ct::select(x_tile >= zero, x_tile, x_tile * ct::full<decltype(x_tile)>(0.01f));
  } else if (op == 6) {
    auto sigmoid = one / (one + ct::exp(-x_tile));
    result = x_tile * sigmoid;
  } else if (op == 7) {
    result = ct::log(one + ct::exp(x_tile));
  } else if (op == 8) {
    auto neg_one = ct::full<decltype(x_tile)>(-1.0f);
    result = ct::select(x_tile > one, one, ct::select(x_tile < neg_one, neg_one, x_tile));
  } else if (op == 9) {
    result = ct::exp(-(x_tile * x_tile));
  } else if (op == 10) {
    auto floor = ct::full<decltype(x_tile)>(1.0e-7f);
    result = ct::log(ct::select(x_tile > floor, x_tile, floor));
  } else if (op == 11) {
    result = ct::select(x_tile >= zero, x_tile, x_tile * ct::full<decltype(x_tile)>(0.25f));
  } else if (op == 12) {
    auto six = ct::full<decltype(x_tile)>(6.0f);
    result = ct::select(x_tile > six, six, ct::select(x_tile < zero, zero, x_tile));
  } else if (op == 13) {
    result = ct::select(x_tile >= zero, x_tile, ct::exp(x_tile) - one);
  } else if (op == 14) {
    auto alpha = ct::full<decltype(x_tile)>(1.6732632423543772f);
    auto scale = ct::full<decltype(x_tile)>(1.0507009873554805f);
    auto inner = ct::select(x_tile >= zero, x_tile, alpha * (ct::exp(x_tile) - one));
    result = scale * inner;
  } else if (op == 15) {
    auto softplus = ct::log(one + ct::exp(x_tile));
    result = x_tile * ct::tanh(softplus);
  } else if (op == 16) {
    result = x_tile / (one + ct::abs(x_tile));
  } else if (op == 17) {
    auto raw = x_tile / ct::full<decltype(x_tile)>(6.0f) + ct::full<decltype(x_tile)>(0.5f);
    result = ct::select(raw > one, one, ct::select(raw < zero, zero, raw));
  } else if (op == 18) {
    auto raw = x_tile / ct::full<decltype(x_tile)>(6.0f) + ct::full<decltype(x_tile)>(0.5f);
    auto hard_sigmoid = ct::select(raw > one, one, ct::select(raw < zero, zero, raw));
    result = x_tile * hard_sigmoid;
  } else if (op == 19) {
    result = ct::select(x_tile >= zero, one, zero);
  } else if (op == 20) {
    auto inv_sqrt2 = ct::full<decltype(x_tile)>(0.7071067811865476f);
    auto z = x_tile * inv_sqrt2;
    auto sign = ct::select(z < zero, ct::full<decltype(z)>(-1.0f), one);
    auto abs_z = ct::abs(z);
    auto t = one / (one + ct::full<decltype(z)>(0.3275911f) * abs_z);
    auto a1 = ct::full<decltype(z)>(0.254829592f);
    auto a2 = ct::full<decltype(z)>(-0.284496736f);
    auto a3 = ct::full<decltype(z)>(1.421413741f);
    auto a4 = ct::full<decltype(z)>(-1.453152027f);
    auto a5 = ct::full<decltype(z)>(1.061405429f);
    auto poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t;
    auto erf_approx = sign * (one - poly * ct::exp(-(abs_z * abs_z)));
    result = ct::full<decltype(x_tile)>(0.5f) * x_tile * (one + erf_approx);
  }

  auto out_view = ct::partition_view{ct::tensor_span{out, ct::extents{n}}, ct::shape{1024_ic}};
  out_view.store_masked(result, bx);
}

__tile_global__ void binary_float32_kernel(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ out,
    std::int64_t n,
    int op) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  lhs = ct::assume_aligned(lhs, 16_ic);
  rhs = ct::assume_aligned(rhs, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  auto lhs_tile = ct::partition_view{ct::tensor_span{lhs, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto rhs_tile = ct::partition_view{ct::tensor_span{rhs, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto result = op == 0 ? lhs_tile + rhs_tile : lhs_tile * rhs_tile;
  auto out_view = ct::partition_view{ct::tensor_span{out, ct::extents{n}}, ct::shape{1024_ic}};
  out_view.store_masked(result, bx);
}

__tile_global__ void binary_pair_float32_kernel(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ out0,
    float* __restrict__ out1,
    std::int64_t n,
    int op) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  lhs = ct::assume_aligned(lhs, 16_ic);
  rhs = ct::assume_aligned(rhs, 16_ic);
  out0 = ct::assume_aligned(out0, 16_ic);
  out1 = ct::assume_aligned(out1, 16_ic);

  const int bx = ct::bid().x;
  auto lhs_tile = ct::partition_view{ct::tensor_span{lhs, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto rhs_tile = ct::partition_view{ct::tensor_span{rhs, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto max_tile = ct::select(lhs_tile > rhs_tile, lhs_tile, rhs_tile);
  auto lhs_exp = ct::exp(lhs_tile - max_tile);
  auto rhs_exp = ct::exp(rhs_tile - max_tile);
  auto denom = lhs_exp + rhs_exp;
  auto result0 = lhs_exp / denom;
  auto result1 = rhs_exp / denom;
  if (op == 1) {
    auto lse = max_tile + ct::log(denom);
    result0 = lhs_tile - lse;
    result1 = rhs_tile - lse;
  }
  auto out0_view = ct::partition_view{ct::tensor_span{out0, ct::extents{n}}, ct::shape{1024_ic}};
  auto out1_view = ct::partition_view{ct::tensor_span{out1, ct::extents{n}}, ct::shape{1024_ic}};
  out0_view.store_masked(result0, bx);
  out1_view.store_masked(result1, bx);
}

__tile_global__ void scalar_unary_float32_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    std::int64_t n,
    float value,
    int op) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  auto x_tile = ct::partition_view{ct::tensor_span{x, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto scalar = ct::full<decltype(x_tile)>(value);
  auto result = op == 0 ? scalar * x_tile : scalar * ct::tanh(x_tile / scalar);
  auto out_view = ct::partition_view{ct::tensor_span{out, ct::extents{n}}, ct::shape{1024_ic}};
  out_view.store_masked(result, bx);
}

__tile_global__ void scalar_binary_float32_kernel(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ out,
    std::int64_t n,
    float value,
    int op) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  lhs = ct::assume_aligned(lhs, 16_ic);
  rhs = ct::assume_aligned(rhs, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  auto lhs_tile = ct::partition_view{ct::tensor_span{lhs, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto rhs_tile = ct::partition_view{ct::tensor_span{rhs, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto scalar = ct::full<decltype(lhs_tile)>(value);
  auto result = op == 0 ? lhs_tile + scalar * rhs_tile : lhs_tile - scalar * rhs_tile;
  auto out_view = ct::partition_view{ct::tensor_span{out, ct::extents{n}}, ct::shape{1024_ic}};
  out_view.store_masked(result, bx);
}

__tile_global__ void ema_update_float32_kernel(
    float* __restrict__ target,
    const float* __restrict__ source,
    std::int64_t n,
    float decay) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  target = ct::assume_aligned(target, 16_ic);
  source = ct::assume_aligned(source, 16_ic);

  const int bx = ct::bid().x;
  auto target_tile = ct::partition_view{ct::tensor_span{target, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto source_tile = ct::partition_view{ct::tensor_span{source, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto decay_tile = ct::full<decltype(target_tile)>(decay);
  auto one_minus_decay = ct::full<decltype(target_tile)>(1.0f - decay);
  auto result = decay_tile * target_tile + one_minus_decay * source_tile;
  auto target_view = ct::partition_view{ct::tensor_span{target, ct::extents{n}}, ct::shape{1024_ic}};
  target_view.store_masked(result, bx);
}

__tile_global__ void gradient_accumulate_float32_kernel(
    float* __restrict__ buffer,
    const float* __restrict__ grad,
    std::int64_t n,
    float scale) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  buffer = ct::assume_aligned(buffer, 16_ic);
  grad = ct::assume_aligned(grad, 16_ic);

  const int bx = ct::bid().x;
  auto buffer_tile = ct::partition_view{ct::tensor_span{buffer, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto grad_tile = ct::partition_view{ct::tensor_span{grad, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto result = buffer_tile + ct::full<decltype(buffer_tile)>(scale) * grad_tile;
  auto buffer_view = ct::partition_view{ct::tensor_span{buffer, ct::extents{n}}, ct::shape{1024_ic}};
  buffer_view.store_masked(result, bx);
}

__tile_global__ void sumsq_partials_float32_kernel(
    const float* __restrict__ values,
    float* __restrict__ partials,
    std::int64_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  values = ct::assume_aligned(values, 16_ic);
  partials = ct::assume_aligned(partials, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto tile = ct::load_masked(values + idx, mask);
  auto zero = ct::full<decltype(tile)>(0.0f);
  auto squared = ct::select(mask, tile * tile, zero);
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(bx));
  ct::store(partials + out_idx, ct::sum(squared, 0_ic));
}

__tile_global__ void scale_inplace_float32_kernel(
    float* __restrict__ values,
    std::int64_t n,
    float scale) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  values = ct::assume_aligned(values, 16_ic);

  const int bx = ct::bid().x;
  auto view = ct::partition_view{ct::tensor_span{values, ct::extents{n}}, ct::shape{1024_ic}};
  auto tile = view.load_masked(bx);
  view.store_masked(tile * ct::full<decltype(tile)>(scale), bx);
}

__tile_global__ void adamw_step_float32_kernel(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    std::int64_t n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float sqrt_bias_correction2) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  param = ct::assume_aligned(param, 16_ic);
  grad = ct::assume_aligned(grad, 16_ic);
  exp_avg = ct::assume_aligned(exp_avg, 16_ic);
  exp_avg_sq = ct::assume_aligned(exp_avg_sq, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto p = ct::load_masked(param + idx, mask);
  auto g = ct::load_masked(grad + idx, mask);
  auto m = ct::load_masked(exp_avg + idx, mask);
  auto v = ct::load_masked(exp_avg_sq + idx, mask);
  auto one = ct::full<decltype(p)>(1.0f);
  auto beta1_tile = ct::full<decltype(p)>(beta1);
  auto beta2_tile = ct::full<decltype(p)>(beta2);
  auto next_m = beta1_tile * m + (one - beta1_tile) * g;
  auto next_v = beta2_tile * v + (one - beta2_tile) * g * g;
  auto decayed = p * (one - ct::full<decltype(p)>(lr * weight_decay));
  auto denom = ct::sqrt(next_v) / ct::full<decltype(p)>(sqrt_bias_correction2) + ct::full<decltype(p)>(eps);
  auto step_size = ct::full<decltype(p)>(lr / bias_correction1);
  auto next_p = decayed - step_size * next_m / denom;
  ct::store_masked(param + idx, next_p, mask);
  ct::store_masked(exp_avg + idx, next_m, mask);
  ct::store_masked(exp_avg_sq + idx, next_v, mask);
}

__tile_global__ void scalar_ternary_float32_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    float* __restrict__ out,
    std::int64_t n,
    float value,
    int op) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  a = ct::assume_aligned(a, 16_ic);
  b = ct::assume_aligned(b, 16_ic);
  c = ct::assume_aligned(c, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  auto a_tile = ct::partition_view{ct::tensor_span{a, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto b_tile = ct::partition_view{ct::tensor_span{b, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto c_tile = ct::partition_view{ct::tensor_span{c, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto scalar = ct::full<decltype(a_tile)>(value);
  auto result = op == 0 ? c_tile - scalar * (a_tile - b_tile) : a_tile + scalar * b_tile + c_tile;
  auto out_view = ct::partition_view{ct::tensor_span{out, ct::extents{n}}, ct::shape{1024_ic}};
  out_view.store_masked(result, bx);
}

__tile_global__ void vector_binary_float32_kernel(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    const float* __restrict__ scale0,
    const float* __restrict__ scale1,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t dim,
    int op) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  lhs = ct::assume_aligned(lhs, 16_ic);
  rhs = ct::assume_aligned(rhs, 16_ic);
  scale0 = ct::assume_aligned(scale0, 16_ic);
  out = ct::assume_aligned(out, 16_ic);
  if (scale1 != nullptr) {
    scale1 = ct::assume_aligned(scale1, 16_ic);
  }

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<decltype(idx)>(n);
  auto lhs_ptrs = lhs + idx;
  auto rhs_ptrs = rhs + idx;
  auto lane = idx % ct::full<decltype(idx)>(dim);
  auto scale0_ptrs = scale0 + lane;
  auto lhs_tile = ct::load_masked(lhs_ptrs, mask);
  auto rhs_tile = ct::load_masked(rhs_ptrs, mask);
  auto scale0_tile = ct::load_masked(scale0_ptrs, mask);
  auto result = lhs_tile + scale0_tile * rhs_tile;
  if (op == 1 && scale1 != nullptr) {
    auto scale1_ptrs = scale1 + lane;
    auto scale1_tile = ct::load_masked(scale1_ptrs, mask);
    result = scale0_tile * lhs_tile + scale1_tile * rhs_tile;
  } else if (op == 2) {
    auto one = ct::full<decltype(scale0_tile)>(1.0f);
    auto beta = one / (one + ct::exp(-scale0_tile));
    auto alpha = ct::sqrt(one - beta * beta);
    result = alpha * lhs_tile + beta * rhs_tile;
  }
  ct::store_masked(out + idx, result, mask);
}

__tile_global__ void qk_gain_float32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ gain,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t heads,
    std::int64_t inner) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  q = ct::assume_aligned(q, 16_ic);
  gain = ct::assume_aligned(gain, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto head = (idx / ct::full<IndexTile>(inner)) % ct::full<IndexTile>(heads);
  auto q_tile = ct::load_masked(q + idx, mask);
  auto gain_tile = ct::load_masked(gain + head, mask);
  ct::store_masked(out + idx, q_tile * gain_tile, mask);
}

__tile_global__ void dyt_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t dim,
    float alpha) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  weight = ct::assume_aligned(weight, 16_ic);
  bias = ct::assume_aligned(bias, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto lane = idx % ct::full<IndexTile>(dim);
  auto x_tile = ct::load_masked(x + idx, mask);
  auto weight_tile = ct::load_masked(weight + lane, mask);
  auto bias_tile = ct::load_masked(bias + lane, mask);
  auto alpha_tile = ct::full<decltype(x_tile)>(alpha);
  auto t = ct::tanh(alpha_tile * x_tile);
  ct::store_masked(out + idx, weight_tile * t + bias_tile, mask);
}

__tile_global__ void reshape_heads_float32_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t seq_len,
    std::int64_t heads,
    std::int64_t head_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto head_dim_tile = ct::full<IndexTile>(head_dim);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto heads_tile = ct::full<IndexTile>(heads);
  auto d = idx % head_dim_tile;
  auto s = (idx / head_dim_tile) % seq_tile;
  auto h = (idx / (head_dim_tile * seq_tile)) % heads_tile;
  auto b = idx / (head_dim_tile * seq_tile * heads_tile);
  auto src = (b * seq_tile + s) * (heads_tile * head_dim_tile) + h * head_dim_tile + d;
  auto value = ct::load_masked(x + src, mask);
  ct::store_masked(out + idx, value, mask);
}

__tile_global__ void merge_heads_float32_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto head_dim_tile = ct::full<IndexTile>(head_dim);
  auto heads_tile = ct::full<IndexTile>(heads);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto d = idx % head_dim_tile;
  auto h = (idx / head_dim_tile) % heads_tile;
  auto s = (idx / (head_dim_tile * heads_tile)) % seq_tile;
  auto b = idx / (head_dim_tile * heads_tile * seq_tile);
  auto src = ((b * heads_tile + h) * seq_tile + s) * head_dim_tile + d;
  auto value = ct::load_masked(x + src, mask);
  ct::store_masked(out + idx, value, mask);
}

__tile_global__ void repeat_kv_float32_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t kv_heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    std::int64_t repeats) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto head_dim_tile = ct::full<IndexTile>(head_dim);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto kv_heads_tile = ct::full<IndexTile>(kv_heads);
  auto repeats_tile = ct::full<IndexTile>(repeats);
  auto d = idx % head_dim_tile;
  auto s = (idx / head_dim_tile) % seq_tile;
  auto h = (idx / (head_dim_tile * seq_tile)) % (kv_heads_tile * repeats_tile);
  auto b = idx / (head_dim_tile * seq_tile * kv_heads_tile * repeats_tile);
  auto src_h = h / repeats_tile;
  auto src = ((b * kv_heads_tile + src_h) * seq_tile + s) * head_dim_tile + d;
  auto value = ct::load_masked(x + src, mask);
  ct::store_masked(out + idx, value, mask);
}

__tile_global__ void broadcast_expert_routes_float32_kernel(
    const float* __restrict__ weights,
    const std::int64_t* __restrict__ indices,
    float* __restrict__ out_weights,
    std::int64_t* __restrict__ out_indices,
    std::int64_t n,
    std::int64_t route_seq,
    std::int64_t seq_len,
    std::int64_t route_width) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  weights = ct::assume_aligned(weights, 16_ic);
  indices = ct::assume_aligned(indices, 16_ic);
  out_weights = ct::assume_aligned(out_weights, 16_ic);
  out_indices = ct::assume_aligned(out_indices, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto width_tile = ct::full<IndexTile>(route_width);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto route_seq_tile = ct::full<IndexTile>(route_seq);
  auto k = idx % width_tile;
  auto s = (idx / width_tile) % seq_tile;
  auto b = idx / (width_tile * seq_tile);
  auto src_s = ct::select(route_seq_tile == ct::full<IndexTile>(1), ct::full<IndexTile>(0), s);
  auto src = (b * route_seq_tile + src_s) * width_tile + k;
  auto weight = ct::load_masked(weights + src, mask);
  auto index = ct::load_masked(indices + src, mask);
  ct::store_masked(out_weights + idx, weight, mask);
  ct::store_masked(out_indices + idx, index, mask);
}

__tile_global__ void broadcast_chunk_routes_float32_kernel(
    const float* __restrict__ weights,
    const std::int64_t* __restrict__ indices,
    float* __restrict__ out_weights,
    std::int64_t* __restrict__ out_indices,
    std::int64_t n,
    std::int64_t chunks,
    std::int64_t seq_len,
    std::int64_t route_width,
    std::int64_t chunk_size) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  weights = ct::assume_aligned(weights, 16_ic);
  indices = ct::assume_aligned(indices, 16_ic);
  out_weights = ct::assume_aligned(out_weights, 16_ic);
  out_indices = ct::assume_aligned(out_indices, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto width_tile = ct::full<IndexTile>(route_width);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto chunks_tile = ct::full<IndexTile>(chunks);
  auto chunk_size_tile = ct::full<IndexTile>(chunk_size);
  auto k = idx % width_tile;
  auto s = (idx / width_tile) % seq_tile;
  auto b = idx / (width_tile * seq_tile);
  auto raw_chunk = s / chunk_size_tile;
  auto chunk = ct::select(raw_chunk >= chunks_tile, chunks_tile - ct::full<IndexTile>(1), raw_chunk);
  auto src = (b * chunks_tile + chunk) * width_tile + k;
  auto weight = ct::load_masked(weights + src, mask);
  auto index = ct::load_masked(indices + src, mask);
  ct::store_masked(out_weights + idx, weight, mask);
  ct::store_masked(out_indices + idx, index, mask);
}

__tile_global__ void byte_patch_merge_float32_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t source_len,
    std::int64_t target_len,
    std::int64_t dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto dim_tile = ct::full<IndexTile>(dim);
  auto target_tile = ct::full<IndexTile>(target_len);
  auto source_tile = ct::full<IndexTile>(source_len);
  auto d = idx % dim_tile;
  auto t = (idx / dim_tile) % target_tile;
  auto b = idx / (dim_tile * target_tile);
  auto src_t = (t * source_tile) / target_tile;
  auto src = (b * source_tile + src_t) * dim_tile + d;
  auto value = ct::load_masked(x + src, mask);
  ct::store_masked(out + idx, value, mask);
}

__tile_global__ void byte_patch_embed_float32_kernel(
    const std::int64_t* __restrict__ tokens,
    const float* __restrict__ embedding,
    const float* __restrict__ proj,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t seq_len,
    std::int64_t model_dim,
    std::int64_t patch_size,
    std::int64_t stride,
    std::int64_t out_len,
    std::int64_t vocab_size) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  tokens = ct::assume_aligned(tokens, 16_ic);
  embedding = ct::assume_aligned(embedding, 16_ic);
  proj = ct::assume_aligned(proj, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto model_dim_tile = ct::full<IndexTile>(model_dim);
  auto out_len_tile = ct::full<IndexTile>(out_len);
  auto out_c = idx % model_dim_tile;
  auto patch_idx = (idx / model_dim_tile) % out_len_tile;
  auto batch_idx = idx / (model_dim_tile * out_len_tile);
  auto acc = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  auto seq_len_tile = ct::full<IndexTile>(seq_len);
  auto vocab_tile = ct::full<IndexTile>(vocab_size);
  auto zero_i = ct::full<IndexTile>(0);
  for (std::int64_t k = 0; k < patch_size; ++k) {
    auto token_pos = patch_idx * ct::full<IndexTile>(stride) + ct::full<IndexTile>(k);
    auto valid_token = mask & (token_pos < seq_len_tile);
    auto token_raw = ct::load_masked(tokens + batch_idx * seq_len_tile + token_pos, valid_token);
    auto token_hi = ct::select(token_raw >= vocab_tile, vocab_tile - ct::full<IndexTile>(1), token_raw);
    auto token_id = ct::select(token_hi < zero_i, zero_i, token_hi);
    for (std::int64_t in_c = 0; in_c < model_dim; ++in_c) {
      auto embed = ct::load_masked(embedding + token_id * model_dim_tile + ct::full<IndexTile>(in_c), valid_token);
      auto weight = ct::load_masked(
          proj + (out_c * model_dim_tile + ct::full<IndexTile>(in_c)) * ct::full<IndexTile>(patch_size) + ct::full<IndexTile>(k),
          mask);
      acc = acc + ct::select(valid_token, embed * weight, ct::full<decltype(embed)>(0.0f));
    }
  }
  ct::store_masked(out + idx, acc, mask);
}

__tile_global__ void causal_chunk_state_float32_kernel(
    const float* __restrict__ hidden,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t seq_len,
    std::int64_t dim,
    std::int64_t chunk_size,
    std::int64_t chunks,
    int mode) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  hidden = ct::assume_aligned(hidden, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto dim_tile = ct::full<IndexTile>(dim);
  auto chunks_tile = ct::full<IndexTile>(chunks);
  auto d = idx % dim_tile;
  auto chunk = (idx / dim_tile) % chunks_tile;
  auto b = idx / (dim_tile * chunks_tile);
  auto chunk_size_tile = ct::full<IndexTile>(chunk_size);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto zero_i = ct::full<IndexTile>(0);
  auto one_i = ct::full<IndexTile>(1);
  auto start = chunk * chunk_size_tile;
  auto end_mean = start + chunk_size_tile;
  end_mean = ct::select(end_mean > seq_tile, seq_tile, end_mean);
  auto boundary = start - one_i;
  boundary = ct::select(boundary < zero_i, zero_i, boundary);
  boundary = ct::select(boundary >= seq_tile, seq_tile - one_i, boundary);

  auto sum = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  auto count = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t t = 0; t < seq_len; ++t) {
    auto t_tile = ct::full<IndexTile>(t);
    auto active = mode == 0 ? ((t_tile >= start) & (t_tile < end_mean)) : (t_tile <= boundary);
    active = active & mask;
    auto src = (b * seq_tile + t_tile) * dim_tile + d;
    auto value = ct::load_masked(hidden + src, active);
    sum = sum + ct::select(active, value, ct::full<decltype(value)>(0.0f));
    count = count + ct::select(active, ct::full<decltype(value)>(1.0f), ct::full<decltype(value)>(0.0f));
  }
  auto denom = ct::select(count > ct::full<decltype(count)>(1.0f), count, ct::full<decltype(count)>(1.0f));
  ct::store_masked(out + idx, sum / denom, mask);
}

__tile_global__ void latent_mse_partials_float32_kernel(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ partials,
    std::int64_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  pred = ct::assume_aligned(pred, 16_ic);
  target = ct::assume_aligned(target, 16_ic);
  partials = ct::assume_aligned(partials, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto pred_tile = ct::load_masked(pred + idx, mask);
  auto target_tile = ct::load_masked(target + idx, mask);
  auto zero = ct::full<decltype(pred_tile)>(0.0f);
  auto diff = ct::select(mask, pred_tile - target_tile, zero);
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(bx));
  ct::store(partials + out_idx, ct::sum(diff * diff, 0_ic));
}

__tile_global__ void semantic_alignment_loss_items_float32_kernel(
    const float* __restrict__ logits,
    const std::int64_t* __restrict__ targets,
    const std::int64_t* __restrict__ term_counts,
    float* __restrict__ losses,
    float* __restrict__ counts,
    std::int64_t n,
    std::int64_t dims,
    std::int64_t terms,
    std::int64_t ignore_index) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  logits = ct::assume_aligned(logits, 16_ic);
  targets = ct::assume_aligned(targets, 16_ic);
  term_counts = ct::assume_aligned(term_counts, 16_ic);
  losses = ct::assume_aligned(losses, 16_ic);
  counts = ct::assume_aligned(counts, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto dims_tile = ct::full<IndexTile>(dims);
  auto terms_tile = ct::full<IndexTile>(terms);
  auto dim_idx = idx % dims_tile;
  auto row_idx = idx / dims_tile;
  auto target = ct::load_masked(targets + row_idx * dims_tile + dim_idx, mask);
  auto term_count_raw = ct::load_masked(term_counts + dim_idx, mask);
  auto term_count = ct::select(term_count_raw > terms_tile, terms_tile, term_count_raw);
  auto valid = mask & (target != ct::full<IndexTile>(ignore_index)) & (target >= ct::full<IndexTile>(0)) & (target < term_count);
  auto base = (row_idx * dims_tile + dim_idx) * terms_tile;
  auto max_val = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(-3.4028234663852886e38f);
  for (std::int64_t term = 0; term < terms; ++term) {
    auto term_tile = ct::full<IndexTile>(term);
    auto active = valid & (term_tile < term_count);
    auto value = ct::load_masked(logits + base + term_tile, active);
    max_val = ct::select(active & (value > max_val), value, max_val);
  }
  auto sum_exp = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t term = 0; term < terms; ++term) {
    auto term_tile = ct::full<IndexTile>(term);
    auto active = valid & (term_tile < term_count);
    auto value = ct::load_masked(logits + base + term_tile, active);
    sum_exp = sum_exp + ct::select(active, ct::exp(value - max_val), ct::full<decltype(value)>(0.0f));
  }
  auto safe_target = ct::select(valid, target, ct::full<IndexTile>(0));
  auto target_logit = ct::load_masked(logits + base + safe_target, valid);
  auto loss = ct::log(sum_exp) + max_val - target_logit;
  ct::store_masked(losses + idx, ct::select(valid, loss, ct::full<decltype(loss)>(0.0f)), mask);
  ct::store_masked(counts + idx, ct::select(valid, ct::full<decltype(loss)>(1.0f), ct::full<decltype(loss)>(0.0f)), mask);
}

__tile_global__ void sum_partials_float32_kernel(
    const float* __restrict__ values,
    float* __restrict__ partials,
    std::int64_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  values = ct::assume_aligned(values, 16_ic);
  partials = ct::assume_aligned(partials, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto value_tile = ct::load_masked(values + idx, mask);
  auto zero = ct::full<decltype(value_tile)>(0.0f);
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(bx));
  ct::store(partials + out_idx, ct::sum(ct::select(mask, value_tile, zero), 0_ic));
}

__tile_global__ void kv_cache_read_float32_kernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ cache_k,
    const float* __restrict__ cache_v,
    float* __restrict__ out_k,
    float* __restrict__ out_v,
    std::int64_t n,
    std::int64_t heads,
    std::int64_t cache_seq,
    std::int64_t current_seq,
    std::int64_t head_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  k = ct::assume_aligned(k, 16_ic);
  v = ct::assume_aligned(v, 16_ic);
  cache_k = ct::assume_aligned(cache_k, 16_ic);
  cache_v = ct::assume_aligned(cache_v, 16_ic);
  out_k = ct::assume_aligned(out_k, 16_ic);
  out_v = ct::assume_aligned(out_v, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto head_dim_tile = ct::full<IndexTile>(head_dim);
  auto heads_tile = ct::full<IndexTile>(heads);
  auto cache_seq_tile = ct::full<IndexTile>(cache_seq);
  auto total_seq_tile = ct::full<IndexTile>(cache_seq + current_seq);
  auto d = idx % head_dim_tile;
  auto s = (idx / head_dim_tile) % total_seq_tile;
  auto h = (idx / (head_dim_tile * total_seq_tile)) % heads_tile;
  auto b = idx / (head_dim_tile * total_seq_tile * heads_tile);
  auto use_cache = s < cache_seq_tile;
  auto current_s = s - cache_seq_tile;
  auto cache_src = ((b * heads_tile + h) * cache_seq_tile + s) * head_dim_tile + d;
  auto current_src = ((b * heads_tile + h) * ct::full<IndexTile>(current_seq) + current_s) * head_dim_tile + d;
  auto cache_mask = mask & use_cache;
  auto current_mask = mask & (!use_cache);
  auto zero = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  auto k_value = ct::load_masked(cache_k + cache_src, cache_mask) + ct::load_masked(k + current_src, current_mask);
  auto v_value = ct::load_masked(cache_v + cache_src, cache_mask) + ct::load_masked(v + current_src, current_mask);
  k_value = ct::select(mask, k_value, zero);
  v_value = ct::select(mask, v_value, zero);
  ct::store_masked(out_k + idx, k_value, mask);
  ct::store_masked(out_v + idx, v_value, mask);
}

__tile_global__ void kv_quant_pack_float32_kernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    std::int64_t head_dim) {
  namespace ct = cuda::tiles;

  const int row = ct::bid().x;
  const std::int64_t kv_dim = head_dim * 2;
  const std::int64_t packed_dim = kv_dim + 1;
  const std::int64_t row_base = static_cast<std::int64_t>(row) * head_dim;
  const std::int64_t out_base = static_cast<std::int64_t>(row) * packed_dim;

  float amax = 0.0f;
  for (std::int64_t d = 0; d < head_dim; ++d) {
    const float kval = k[row_base + d];
    const float vval = v[row_base + d];
    const float kabs = kval < 0.0f ? -kval : kval;
    const float vabs = vval < 0.0f ? -vval : vval;
    amax = kabs > amax ? kabs : amax;
    amax = vabs > amax ? vabs : amax;
  }
  const float scale = (amax > 1.0e-7f ? amax : 1.0e-7f) / 127.0f;
  for (std::int64_t d = 0; d < head_dim; ++d) {
    const float raw_k = k[row_base + d] / scale;
    const float raw_v = v[row_base + d] / scale;
    float qk = static_cast<float>(raw_k >= 0.0f ? static_cast<int>(raw_k + 0.5f) : static_cast<int>(raw_k - 0.5f));
    float qv = static_cast<float>(raw_v >= 0.0f ? static_cast<int>(raw_v + 0.5f) : static_cast<int>(raw_v - 0.5f));
    qk = qk > 127.0f ? 127.0f : qk;
    qk = qk < -128.0f ? -128.0f : qk;
    qv = qv > 127.0f ? 127.0f : qv;
    qv = qv < -128.0f ? -128.0f : qv;
    out[out_base + d] = qk;
    out[out_base + head_dim + d] = qv;
  }
  out[out_base + kv_dim] = scale;
}

__tile_global__ void kv_quant_unpack_float32_kernel(
    const float* __restrict__ packed,
    float* __restrict__ out_k,
    float* __restrict__ out_v,
    std::int64_t n,
    std::int64_t head_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  packed = ct::assume_aligned(packed, 16_ic);
  out_k = ct::assume_aligned(out_k, 16_ic);
  out_v = ct::assume_aligned(out_v, 16_ic);

  const int bx = ct::bid().x;
  const std::int64_t kv_dim = head_dim * 2;
  const std::int64_t packed_dim = kv_dim + 1;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto head_dim_tile = ct::full<IndexTile>(head_dim);
  auto kv_dim_tile = ct::full<IndexTile>(kv_dim);
  auto d = idx % kv_dim_tile;
  auto row = idx / kv_dim_tile;
  auto scale = ct::load_masked(packed + row * ct::full<IndexTile>(packed_dim) + ct::full<IndexTile>(kv_dim), mask);
  auto quantized = ct::load_masked(packed + row * ct::full<IndexTile>(packed_dim) + d, mask);
  auto value = quantized * scale;
  auto k_mask = mask & (d < head_dim_tile);
  auto v_mask = mask & (!k_mask);
  auto out_row_base = row * head_dim_tile;
  ct::store_masked(out_k + out_row_base + d, value, k_mask);
  ct::store_masked(out_v + out_row_base + (d - head_dim_tile), value, v_mask);
}

__tile_global__ void absolute_position_embedding_float32_kernel(
    const float* __restrict__ weight,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t seq_len,
    std::int64_t model_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  weight = ct::assume_aligned(weight, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto dim_tile = ct::full<IndexTile>(model_dim);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto d = idx % dim_tile;
  auto s = (idx / dim_tile) % seq_tile;
  auto src = s * dim_tile + d;
  auto value = ct::load_masked(weight + src, mask);
  ct::store_masked(out + idx, value, mask);
}

__tile_global__ void token_embedding_float32_kernel(
    const float* __restrict__ weight,
    const std::int64_t* __restrict__ token_ids,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t model_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  weight = ct::assume_aligned(weight, 16_ic);
  token_ids = ct::assume_aligned(token_ids, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto dim_tile = ct::full<IndexTile>(model_dim);
  auto d = idx % dim_tile;
  auto token_offset = idx / dim_tile;
  auto token = ct::load_masked(token_ids + token_offset, mask);
  auto src = token * dim_tile + d;
  auto value = ct::load_masked(weight + src, mask);
  ct::store_masked(out + idx, value, mask);
}

__tile_global__ void rotary_embedding_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ inv_freq,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  inv_freq = ct::assume_aligned(inv_freq, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto head_dim_tile = ct::full<IndexTile>(head_dim);
  auto half_tile = ct::full<IndexTile>(head_dim / 2);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto d = idx % head_dim_tile;
  auto s = (idx / head_dim_tile) % seq_tile;
  auto pair_d = d % half_tile;
  auto first_half = d < half_tile;
  auto base = idx - d;
  auto x1 = ct::load_masked(x + base + pair_d, mask);
  auto x2 = ct::load_masked(x + base + pair_d + half_tile, mask);
  auto inv = ct::load_masked(inv_freq + pair_d, mask);
  auto angle = ct::tile<float, decltype(ct::shape{1024_ic})>(s) * inv;
  auto c = ct::cos(angle);
  auto sn = ct::sin(angle);
  auto first = x1 * c + x2 * sn;
  auto second = -x1 * sn + x2 * c;
  auto value = ct::select(first_half, first, second);
  ct::store_masked(out + idx, value, mask);
}

__tile_global__ void rms_norm_float32_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    std::int64_t rows,
    std::int64_t dim,
    float eps) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int row = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>();
  auto mask = d < ct::full<IndexTile>(dim);
  auto base = ct::full<IndexTile>(static_cast<std::int64_t>(row) * dim);
  auto x_tile = ct::load_masked(x + base + d, mask);
  auto zero = ct::full<decltype(x_tile)>(0.0f);
  auto valid_x = ct::select(mask, x_tile, zero);
  auto sum_sq = ct::sum(valid_x * valid_x, 0_ic);
  auto mean_sq = sum_sq / ct::full<decltype(sum_sq)>(static_cast<float>(dim));
  auto scale = ct::rsqrt(mean_sq + ct::full<decltype(sum_sq)>(eps));
  ct::store_masked(out + base + d, valid_x * scale, mask);
}

__tile_global__ void layer_norm_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    std::int64_t rows,
    std::int64_t dim,
    float eps) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  weight = ct::assume_aligned(weight, 16_ic);
  bias = ct::assume_aligned(bias, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int row = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>();
  auto mask = d < ct::full<IndexTile>(dim);
  auto base = ct::full<IndexTile>(static_cast<std::int64_t>(row) * dim);
  auto x_tile = ct::load_masked(x + base + d, mask);
  auto zero = ct::full<decltype(x_tile)>(0.0f);
  auto valid_x = ct::select(mask, x_tile, zero);
  auto sum_x = ct::sum(valid_x, 0_ic);
  auto mean = sum_x / ct::full<decltype(sum_x)>(static_cast<float>(dim));
  auto centered = ct::select(mask, x_tile - mean, zero);
  auto sum_sq = ct::sum(centered * centered, 0_ic);
  auto var = sum_sq / ct::full<decltype(sum_sq)>(static_cast<float>(dim));
  auto scale = ct::rsqrt(var + ct::full<decltype(var)>(eps));
  auto weight_tile = ct::load_masked(weight + d, mask);
  auto bias_tile = ct::load_masked(bias + d, mask);
  ct::store_masked(out + base + d, centered * scale * weight_tile + bias_tile, mask);
}

__tile_global__ void softmax_lastdim_float32_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    std::int64_t rows,
    std::int64_t dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int row = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto d = ct::iota<IndexTile>();
  auto mask = d < ct::full<IndexTile>(dim);
  auto base = ct::full<IndexTile>(static_cast<std::int64_t>(row) * dim);
  auto x_tile = ct::load_masked(x + base + d, mask);
  auto neg_inf = ct::full<decltype(x_tile)>(-3.4028234663852886e38f);
  auto safe_x = ct::select(mask, x_tile, neg_inf);
  auto max_val = ct::reduce_max(safe_x, 0_ic);
  auto exp_x = ct::select(mask, ct::exp(x_tile - max_val), ct::full<decltype(x_tile)>(0.0f));
  auto denom = ct::sum(exp_x, 0_ic);
  ct::store_masked(out + base + d, exp_x / denom, mask);
}

__tile_global__ void semantic_hash_int64_kernel(
    const float* __restrict__ sem_vec,
    const float* __restrict__ proj,
    std::int64_t* __restrict__ out,
    std::int64_t batch,
    std::int64_t dim,
    std::int64_t tables,
    std::int64_t planes) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  sem_vec = ct::assume_aligned(sem_vec, 16_ic);
  proj = ct::assume_aligned(proj, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const std::int64_t job = static_cast<std::int64_t>(ct::bid().x);
  const std::int64_t row = job / tables;
  const std::int64_t table = job % tables;
  std::int64_t hash = 0;
  for (std::int64_t plane = 0; plane < planes; ++plane) {
    float dot = 0.0f;
    const std::int64_t proj_base = (table * planes + plane) * dim;
    const std::int64_t row_base = row * dim;
    for (std::int64_t d = 0; d < dim; ++d) {
      dot += sem_vec[row_base + d] * proj[proj_base + d];
    }
    if (dot > 0.0f) {
      hash |= (static_cast<std::int64_t>(1) << plane);
    }
  }
  out[row * tables + table] = hash;
}

__global__ void topk_route_float32_kernel(
    const float* __restrict__ logits,
    float* __restrict__ weights,
    std::int64_t* __restrict__ indices,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t top_k) {
  const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  float top_vals[64];
  std::int64_t top_idx[64];
  for (std::int64_t k = 0; k < top_k; ++k) {
    top_vals[k] = -3.4028234663852886e38f;
    top_idx[k] = 0;
  }
  const std::int64_t base = row * experts;
  for (std::int64_t expert = 0; expert < experts; ++expert) {
    const float value = logits[base + expert];
    for (std::int64_t slot = 0; slot < top_k; ++slot) {
      if (value > top_vals[slot]) {
        for (std::int64_t move = top_k - 1; move > slot; --move) {
          top_vals[move] = top_vals[move - 1];
          top_idx[move] = top_idx[move - 1];
        }
        top_vals[slot] = value;
        top_idx[slot] = expert;
        break;
      }
    }
  }
  const float max_val = top_vals[0];
  float denom = 0.0f;
  for (std::int64_t k = 0; k < top_k; ++k) {
    denom += expf(top_vals[k] - max_val);
  }
  const std::int64_t out_base = row * top_k;
  for (std::int64_t k = 0; k < top_k; ++k) {
    weights[out_base + k] = expf(top_vals[k] - max_val) / denom;
    indices[out_base + k] = top_idx[k];
  }
}

__tile_global__ void attentionless_decoder_float32_kernel(
    const std::int64_t* __restrict__ bucket_indices,
    const float* __restrict__ expert_output,
    const float* __restrict__ bucket_embed,
    const float* __restrict__ out_weight,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t residual_dim,
    std::int64_t vocab_size,
    std::int64_t n_buckets) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  bucket_indices = ct::assume_aligned(bucket_indices, 16_ic);
  expert_output = ct::assume_aligned(expert_output, 16_ic);
  bucket_embed = ct::assume_aligned(bucket_embed, 16_ic);
  out_weight = ct::assume_aligned(out_weight, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto vocab_tile = ct::full<IndexTile>(vocab_size);
  auto residual_tile = ct::full<IndexTile>(residual_dim);
  auto out_col = idx % vocab_tile;
  auto row = idx / vocab_tile;
  auto raw_bucket = ct::load_masked(bucket_indices + row, mask);
  auto bucket = raw_bucket % ct::full<IndexTile>(n_buckets);
  bucket = ct::select(bucket < ct::full<IndexTile>(0), bucket + ct::full<IndexTile>(n_buckets), bucket);
  auto acc = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t r = 0; r < residual_dim; ++r) {
    auto r_tile = ct::full<IndexTile>(r);
    auto expert_val = ct::load_masked(expert_output + row * residual_tile + r_tile, mask);
    auto bucket_val = ct::load_masked(bucket_embed + bucket * residual_tile + r_tile, mask);
    auto weight_val = ct::load_masked(out_weight + out_col * residual_tile + r_tile, mask);
    acc = acc + (expert_val + bucket_val) * weight_val;
  }
  ct::store_masked(out + idx, acc, mask);
}

__tile_global__ void expert_bias_add_float32_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t experts) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  logits = ct::assume_aligned(logits, 16_ic);
  bias = ct::assume_aligned(bias, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto expert = idx % ct::full<IndexTile>(experts);
  auto value = ct::load_masked(logits + idx, mask);
  auto bias_value = ct::load_masked(bias + expert, mask);
  ct::store_masked(out + idx, value + bias_value, mask);
}

__tile_global__ void group_norm_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    std::int64_t seq_len,
    std::int64_t dim,
    std::int64_t num_groups,
    float eps) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  weight = ct::assume_aligned(weight, 16_ic);
  bias = ct::assume_aligned(bias, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bg = ct::bid().x;
  const std::int64_t b_scalar = bg / num_groups;
  const std::int64_t g_scalar = bg % num_groups;
  const std::int64_t group_dim = dim / num_groups;
  const std::int64_t group_elems = seq_len * group_dim;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto e = ct::iota<IndexTile>();
  auto mask = e < ct::full<IndexTile>(group_elems);
  auto group_dim_tile = ct::full<IndexTile>(group_dim);
  auto seq = e / group_dim_tile;
  auto c = e % group_dim_tile;
  auto d = ct::full<IndexTile>(g_scalar * group_dim) + c;
  auto src = (ct::full<IndexTile>(b_scalar) * ct::full<IndexTile>(seq_len) + seq) * ct::full<IndexTile>(dim) + d;
  auto x_tile = ct::load_masked(x + src, mask);
  auto zero = ct::full<decltype(x_tile)>(0.0f);
  auto valid_x = ct::select(mask, x_tile, zero);
  auto sum_x = ct::sum(valid_x, 0_ic);
  auto mean = sum_x / ct::full<decltype(sum_x)>(static_cast<float>(group_elems));
  auto centered = ct::select(mask, x_tile - mean, zero);
  auto sum_sq = ct::sum(centered * centered, 0_ic);
  auto var = sum_sq / ct::full<decltype(sum_sq)>(static_cast<float>(group_elems));
  auto scale = ct::rsqrt(var + ct::full<decltype(var)>(eps));
  auto weight_tile = ct::load_masked(weight + d, mask);
  auto bias_tile = ct::load_masked(bias + d, mask);
  ct::store_masked(out + src, centered * scale * weight_tile + bias_tile, mask);
}

__tile_global__ void scaled_residual_add_float32_kernel(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    const float* __restrict__ scale,
    float* __restrict__ out,
    std::int64_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  lhs = ct::assume_aligned(lhs, 16_ic);
  rhs = ct::assume_aligned(rhs, 16_ic);
  scale = ct::assume_aligned(scale, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  auto lhs_tile = ct::partition_view{ct::tensor_span{lhs, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto rhs_tile = ct::partition_view{ct::tensor_span{rhs, ct::extents{n}}, ct::shape{1024_ic}}.load_masked(bx);
  auto scale_tile = ct::full<decltype(lhs_tile)>(*scale);
  auto result = lhs_tile + scale_tile * rhs_tile;
  ct::partition_view{ct::tensor_span{out, ct::extents{n}}, ct::shape{1024_ic}}.store_masked(result, bx);
}

__tile_global__ void linear_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  weight = ct::assume_aligned(weight, 16_ic);
  out = ct::assume_aligned(out, 16_ic);
  if (has_bias) {
    bias = ct::assume_aligned(bias, 16_ic);
  }

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto output_dim_tile = ct::full<IndexTile>(output_dim);
  auto out_col = idx % output_dim_tile;
  auto row = idx / output_dim_tile;
  auto acc = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t in_col = 0; in_col < input_dim; ++in_col) {
    auto x_value = ct::load_masked(x + row * ct::full<IndexTile>(input_dim) + ct::full<IndexTile>(in_col), mask);
    auto w_value = ct::load_masked(weight + out_col * ct::full<IndexTile>(input_dim) + ct::full<IndexTile>(in_col), mask);
    acc = acc + x_value * w_value;
  }
  if (has_bias) {
    acc = acc + ct::load_masked(bias + out_col, mask);
  }
  ct::store_masked(out + idx, acc, mask);
}

__tile_global__ void act_weighted_sum_float32_kernel(
    const float* __restrict__ states,
    const float* __restrict__ weights,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t steps,
    std::int64_t inner) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  states = ct::assume_aligned(states, 16_ic);
  weights = ct::assume_aligned(weights, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto inner_tile = ct::full<IndexTile>(inner);
  auto offset = idx % inner_tile;
  auto batch = idx / inner_tile;
  auto acc = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t step = 0; step < steps; ++step) {
    auto weight = ct::load_masked(weights + batch * ct::full<IndexTile>(steps) + ct::full<IndexTile>(step), mask);
    auto state = ct::load_masked(
        states + (batch * ct::full<IndexTile>(steps) + ct::full<IndexTile>(step)) * inner_tile + offset,
        mask);
    acc = acc + state * weight;
  }
  ct::store_masked(out + idx, acc, mask);
}

__tile_global__ void latent_pool_float32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ mask_values,
    float* __restrict__ out,
    std::int64_t n,
    std::int64_t seq_len,
    std::int64_t dim) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  x = ct::assume_aligned(x, 16_ic);
  mask_values = ct::assume_aligned(mask_values, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto active = idx < ct::full<IndexTile>(n);
  auto dim_tile = ct::full<IndexTile>(dim);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto d = idx % dim_tile;
  auto batch = idx / dim_tile;
  auto acc = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  auto fallback = ct::full<decltype(acc)>(0.0f);
  auto denom = ct::full<decltype(acc)>(0.0f);
  for (std::int64_t step = 0; step < seq_len; ++step) {
    auto mask_value = ct::load_masked(mask_values + batch * seq_tile + ct::full<IndexTile>(step), active);
    auto value = ct::load_masked(x + (batch * seq_tile + ct::full<IndexTile>(step)) * dim_tile + d, active);
    acc = acc + value * mask_value;
    fallback = fallback + value;
    denom = denom + mask_value;
  }
  auto zero = ct::full<decltype(acc)>(0.0f);
  auto pooled = acc / denom;
  auto mean = fallback / ct::full<decltype(acc)>(static_cast<float>(seq_len));
  auto result = ct::select(denom > zero, pooled, mean);
  ct::store_masked(out + idx, result, active);
}

__tile_global__ void token_cross_entropy_partials_float32_kernel(
    const float* __restrict__ logits,
    const std::int64_t* __restrict__ targets,
    float* __restrict__ partials,
    std::int64_t rows,
    std::int64_t vocab) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  logits = ct::assume_aligned(logits, 16_ic);
  targets = ct::assume_aligned(targets, 16_ic);
  partials = ct::assume_aligned(partials, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto row = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto active = row < ct::full<IndexTile>(rows);
  auto vocab_tile = ct::full<IndexTile>(vocab);
  auto maxv = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(-3.4028234663852886e38f);
  for (std::int64_t col = 0; col < vocab; ++col) {
    auto value = ct::load_masked(logits + row * vocab_tile + ct::full<IndexTile>(col), active);
    maxv = ct::select(value > maxv, value, maxv);
  }
  auto denom = ct::full<decltype(maxv)>(0.0f);
  for (std::int64_t col = 0; col < vocab; ++col) {
    auto value = ct::load_masked(logits + row * vocab_tile + ct::full<IndexTile>(col), active);
    denom = denom + ct::exp(value - maxv);
  }
  auto target = ct::load_masked(targets + row, active);
  auto target_value = ct::load_masked(logits + row * vocab_tile + target, active);
  auto loss = ct::log(denom) + maxv - target_value;
  loss = ct::select(active, loss, ct::full<decltype(loss)>(0.0f));
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(bx));
  ct::store(partials + out_idx, ct::sum(loss, 0_ic));
}

__tile_global__ void masked_token_cross_entropy_partials_float32_kernel(
    const float* __restrict__ logits,
    const std::int64_t* __restrict__ targets,
    const float* __restrict__ loss_mask,
    float* __restrict__ loss_partials,
    float* __restrict__ mask_partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t ignore_index) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  logits = ct::assume_aligned(logits, 16_ic);
  targets = ct::assume_aligned(targets, 16_ic);
  loss_mask = ct::assume_aligned(loss_mask, 16_ic);
  loss_partials = ct::assume_aligned(loss_partials, 16_ic);
  mask_partials = ct::assume_aligned(mask_partials, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto row = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto active = row < ct::full<IndexTile>(rows);
  auto vocab_tile = ct::full<IndexTile>(vocab);
  auto target = ct::load_masked(targets + row, active);
  auto valid = active && (target != ct::full<IndexTile>(ignore_index));
  auto safe_target = ct::select(valid, target, ct::full<IndexTile>(0));
  auto maxv = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(-3.4028234663852886e38f);
  for (std::int64_t col = 0; col < vocab; ++col) {
    auto value = ct::load_masked(logits + row * vocab_tile + ct::full<IndexTile>(col), active);
    maxv = ct::select(value > maxv, value, maxv);
  }
  auto denom = ct::full<decltype(maxv)>(0.0f);
  for (std::int64_t col = 0; col < vocab; ++col) {
    auto value = ct::load_masked(logits + row * vocab_tile + ct::full<IndexTile>(col), active);
    denom = denom + ct::exp(value - maxv);
  }
  auto target_value = ct::load_masked(logits + row * vocab_tile + safe_target, active);
  auto per_token = ct::select(valid, ct::log(denom) + maxv - target_value, ct::full<decltype(maxv)>(0.0f));
  auto mask_value = ct::load_masked(loss_mask + row, active);
  auto weighted = ct::select(active, per_token * mask_value, ct::full<decltype(per_token)>(0.0f));
  auto mask_sum = ct::select(active, mask_value, ct::full<decltype(mask_value)>(0.0f));
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(bx));
  ct::store(loss_partials + out_idx, ct::sum(weighted, 0_ic));
  ct::store(mask_partials + out_idx, ct::sum(mask_sum, 0_ic));
}

__tile_global__ void sequence_logp_float32_kernel(
    const float* __restrict__ logits,
    const std::int64_t* __restrict__ targets,
    const float* __restrict__ loss_mask,
    float* __restrict__ out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t vocab,
    std::int64_t ignore_index) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  logits = ct::assume_aligned(logits, 16_ic);
  targets = ct::assume_aligned(targets, 16_ic);
  loss_mask = ct::assume_aligned(loss_mask, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto batch_idx = ct::iota<IndexTile>();
  auto active = batch_idx < ct::full<IndexTile>(batch);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto vocab_tile = ct::full<IndexTile>(vocab);
  auto acc = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t step = 0; step < seq_len; ++step) {
    auto row = batch_idx * seq_tile + ct::full<IndexTile>(step);
    auto target = ct::load_masked(targets + row, active);
    auto valid = active && (target != ct::full<IndexTile>(ignore_index));
    auto safe_target = ct::select(valid, target, ct::full<IndexTile>(0));
    auto maxv = ct::full<decltype(acc)>(-3.4028234663852886e38f);
    for (std::int64_t col = 0; col < vocab; ++col) {
      auto value = ct::load_masked(logits + row * vocab_tile + ct::full<IndexTile>(col), active);
      maxv = ct::select(value > maxv, value, maxv);
    }
    auto denom = ct::full<decltype(acc)>(0.0f);
    for (std::int64_t col = 0; col < vocab; ++col) {
      auto value = ct::load_masked(logits + row * vocab_tile + ct::full<IndexTile>(col), active);
      denom = denom + ct::exp(value - maxv);
    }
    auto target_value = ct::load_masked(logits + row * vocab_tile + safe_target, active);
    auto logp = target_value - maxv - ct::log(denom);
    auto mask_value = ct::load_masked(loss_mask + row, active);
    acc = acc + ct::select(valid, logp * mask_value, ct::full<decltype(acc)>(0.0f));
  }
  ct::store_masked(out + batch_idx, acc, active);
}

__tile_global__ void preference_bce_partials_float32_kernel(
    const float* __restrict__ reward_chosen,
    const float* __restrict__ reward_rejected,
    float* __restrict__ partials,
    std::int64_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  reward_chosen = ct::assume_aligned(reward_chosen, 16_ic);
  reward_rejected = ct::assume_aligned(reward_rejected, 16_ic);
  partials = ct::assume_aligned(partials, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto chosen = ct::load_masked(reward_chosen + idx, mask);
  auto rejected = ct::load_masked(reward_rejected + idx, mask);
  auto zero = ct::full<decltype(chosen)>(0.0f);
  auto diff = chosen - rejected;
  auto loss = ct::log(ct::full<decltype(diff)>(1.0f) + ct::exp(-diff));
  loss = ct::select(mask, loss, zero);
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(bx));
  ct::store(partials + out_idx, ct::sum(loss, 0_ic));
}

__tile_global__ void ppo_clipped_loss_partials_float32_kernel(
    const float* __restrict__ logp_new,
    const float* __restrict__ logp_old,
    const float* __restrict__ advantages,
    const float* __restrict__ value_new,
    const float* __restrict__ value_old,
    const float* __restrict__ returns,
    float* __restrict__ policy_partials,
    float* __restrict__ value_partials,
    std::int64_t n,
    float clip_range) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  logp_new = ct::assume_aligned(logp_new, 16_ic);
  logp_old = ct::assume_aligned(logp_old, 16_ic);
  advantages = ct::assume_aligned(advantages, 16_ic);
  value_new = ct::assume_aligned(value_new, 16_ic);
  value_old = ct::assume_aligned(value_old, 16_ic);
  returns = ct::assume_aligned(returns, 16_ic);
  policy_partials = ct::assume_aligned(policy_partials, 16_ic);
  value_partials = ct::assume_aligned(value_partials, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto new_logp = ct::load_masked(logp_new + idx, mask);
  auto old_logp = ct::load_masked(logp_old + idx, mask);
  auto adv = ct::load_masked(advantages + idx, mask);
  auto v_new = ct::load_masked(value_new + idx, mask);
  auto v_old = ct::load_masked(value_old + idx, mask);
  auto ret = ct::load_masked(returns + idx, mask);
  auto zero = ct::full<decltype(new_logp)>(0.0f);
  auto one = ct::full<decltype(new_logp)>(1.0f);
  auto clip = ct::full<decltype(new_logp)>(clip_range);

  auto ratio = ct::exp(new_logp - old_logp);
  auto ratio_clipped = ct::select(ratio > one + clip, one + clip, ct::select(ratio < one - clip, one - clip, ratio));
  auto unclipped = ratio * adv;
  auto clipped = ratio_clipped * adv;
  auto policy = -ct::select(unclipped < clipped, unclipped, clipped);

  auto delta = v_new - v_old;
  auto delta_clipped = ct::select(delta > clip, clip, ct::select(delta < -clip, -clip, delta));
  auto v_clipped = v_old + delta_clipped;
  auto vf_sq1 = (v_new - ret) * (v_new - ret);
  auto vf_sq2 = (v_clipped - ret) * (v_clipped - ret);
  auto value = ct::full<decltype(vf_sq1)>(0.5f) * ct::select(vf_sq1 > vf_sq2, vf_sq1, vf_sq2);

  policy = ct::select(mask, policy, zero);
  value = ct::select(mask, value, zero);
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(bx));
  ct::store(policy_partials + out_idx, ct::sum(policy, 0_ic));
  ct::store(value_partials + out_idx, ct::sum(value, 0_ic));
}

__tile_global__ void gae_compute_float32_kernel(
    const float* __restrict__ rewards,
    const float* __restrict__ values,
    float* __restrict__ advantages,
    float* __restrict__ returns,
    std::int64_t seq_len,
    float gamma,
    float lambda_value) {
  namespace ct = cuda::tiles;

  const int batch = ct::bid().x;
  const std::int64_t base = static_cast<std::int64_t>(batch) * seq_len;
  float next_adv = 0.0f;
  float next_value = 0.0f;
  for (std::int64_t t = seq_len - 1; t >= 0; --t) {
    const std::int64_t idx = base + t;
    const float delta = rewards[idx] + gamma * next_value - values[idx];
    next_adv = delta + gamma * lambda_value * next_adv;
    advantages[idx] = next_adv;
    returns[idx] = next_adv + values[idx];
    next_value = values[idx];
  }
}

__tile_global__ void dpo_pairwise_partials_float32_kernel(
    const float* __restrict__ policy_logp_chosen,
    const float* __restrict__ policy_logp_rejected,
    const float* __restrict__ ref_logp_chosen,
    const float* __restrict__ ref_logp_rejected,
    float* __restrict__ partials,
    float* __restrict__ chosen_reward_out,
    float* __restrict__ rejected_reward_out,
    std::int64_t n,
    float beta,
    float label_smoothing,
    int loss_type) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  policy_logp_chosen = ct::assume_aligned(policy_logp_chosen, 16_ic);
  policy_logp_rejected = ct::assume_aligned(policy_logp_rejected, 16_ic);
  ref_logp_chosen = ct::assume_aligned(ref_logp_chosen, 16_ic);
  ref_logp_rejected = ct::assume_aligned(ref_logp_rejected, 16_ic);
  partials = ct::assume_aligned(partials, 16_ic);
  chosen_reward_out = ct::assume_aligned(chosen_reward_out, 16_ic);
  rejected_reward_out = ct::assume_aligned(rejected_reward_out, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto policy_chosen = ct::load_masked(policy_logp_chosen + idx, mask);
  auto policy_rejected = ct::load_masked(policy_logp_rejected + idx, mask);
  auto ref_chosen = ct::load_masked(ref_logp_chosen + idx, mask);
  auto ref_rejected = ct::load_masked(ref_logp_rejected + idx, mask);
  auto beta_tile = ct::full<decltype(policy_chosen)>(beta);
  auto one = ct::full<decltype(policy_chosen)>(1.0f);
  auto zero = ct::full<decltype(policy_chosen)>(0.0f);
  auto chosen_logratio = policy_chosen - ref_chosen;
  auto rejected_logratio = policy_rejected - ref_rejected;
  auto logits = beta_tile * (chosen_logratio - rejected_logratio);
  auto loss = ct::log(one + ct::exp(-logits));
  if (loss_type == 1) {
    loss = ct::select(one - logits > zero, one - logits, zero);
  } else if (loss_type == 2) {
    auto target = ct::full<decltype(policy_chosen)>(1.0f / (2.0f * beta));
    auto delta = logits - target;
    loss = delta * delta;
  } else if (label_smoothing > 0.0f) {
    auto smoothing = ct::full<decltype(policy_chosen)>(label_smoothing);
    loss = loss * (one - smoothing) + ct::log(one + ct::exp(logits)) * smoothing;
  }
  loss = ct::select(mask, loss, zero);
  ct::store_masked(chosen_reward_out + idx, beta_tile * chosen_logratio, mask);
  ct::store_masked(rejected_reward_out + idx, beta_tile * rejected_logratio, mask);
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(bx));
  ct::store(partials + out_idx, ct::sum(loss, 0_ic));
}

__tile_global__ void route_selection_loss_partials_float32_kernel(
    const float* __restrict__ route_logits,
    const std::int64_t* __restrict__ sem_targets,
    float* __restrict__ loss_partials,
    float* __restrict__ count_partials,
    std::int64_t n,
    std::int64_t seq_len,
    std::int64_t experts,
    std::int64_t num_vocab_dims,
    std::int64_t shared_experts,
    std::int64_t ignore_index) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  route_logits = ct::assume_aligned(route_logits, 16_ic);
  sem_targets = ct::assume_aligned(sem_targets, 16_ic);
  loss_partials = ct::assume_aligned(loss_partials, 16_ic);
  count_partials = ct::assume_aligned(count_partials, 16_ic);

  const int bx = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>() + ct::full<IndexTile>(static_cast<std::int64_t>(bx) * kTileSize);
  auto mask = idx < ct::full<IndexTile>(n);
  auto dims_tile = ct::full<IndexTile>(num_vocab_dims);
  auto seq_tile = ct::full<IndexTile>(seq_len);
  auto experts_tile = ct::full<IndexTile>(experts);
  auto shared_tile = ct::full<IndexTile>(shared_experts);
  auto d = idx % dims_tile;
  auto s = (idx / dims_tile) % seq_tile;
  auto b = idx / (dims_tile * seq_tile);
  auto target = ct::load_masked(sem_targets + b * dims_tile + d, mask);
  auto valid = mask & (target != ct::full<IndexTile>(ignore_index));
  auto logit = ct::load_masked(route_logits + (b * seq_tile + s) * experts_tile + shared_tile + d, valid);
  auto zero = ct::full<decltype(logit)>(0.0f);
  auto one = ct::full<decltype(logit)>(1.0f);
  auto loss = ct::select(valid, ct::log(one + ct::exp(-logit)), zero);
  auto count = ct::select(valid, one, zero);
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(bx));
  ct::store(loss_partials + out_idx, ct::sum(loss, 0_ic));
  ct::store(count_partials + out_idx, ct::sum(count, 0_ic));
}

__tile_global__ void route_balance_density_float32_kernel(
    const float* __restrict__ route_logits,
    float* __restrict__ density,
    std::int64_t rows,
    std::int64_t experts) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  route_logits = ct::assume_aligned(route_logits, 16_ic);
  density = ct::assume_aligned(density, 16_ic);

  const int expert = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto row = ct::iota<IndexTile>();
  auto mask = row < ct::full<IndexTile>(rows);
  auto maxv = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(-3.4028234663852886e38f);
  for (std::int64_t col = 0; col < experts; ++col) {
    auto value = ct::load_masked(route_logits + row * ct::full<IndexTile>(experts) + ct::full<IndexTile>(col), mask);
    maxv = ct::select(value > maxv, value, maxv);
  }
  auto denom = ct::full<decltype(maxv)>(0.0f);
  for (std::int64_t col = 0; col < experts; ++col) {
    auto value = ct::load_masked(route_logits + row * ct::full<IndexTile>(experts) + ct::full<IndexTile>(col), mask);
    denom = denom + ct::exp(value - maxv);
  }
  auto selected = ct::load_masked(route_logits + row * ct::full<IndexTile>(experts) + ct::full<IndexTile>(expert), mask);
  auto prob = ct::exp(selected - maxv) / denom;
  prob = ct::select(mask, prob, ct::full<decltype(prob)>(0.0f));
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(expert));
  auto scale = ct::full<ct::tile<float, decltype(ct::shape{1_ic})>>(1.0f / static_cast<float>(rows));
  ct::store(density + out_idx, ct::sum(prob, 0_ic) * scale);
}

__tile_global__ void route_balance_loss_float32_kernel(
    const float* __restrict__ density,
    float* __restrict__ out,
    std::int64_t experts) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  density = ct::assume_aligned(density, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto idx = ct::iota<IndexTile>();
  auto mask = idx < ct::full<IndexTile>(experts);
  auto value = ct::load_masked(density + idx, mask);
  auto sq = ct::select(mask, value * value, ct::full<decltype(value)>(0.0f));
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(0);
  auto scale = ct::full<ct::tile<float, decltype(ct::shape{1_ic})>>(static_cast<float>(experts));
  ct::store(out + out_idx, ct::sum(sq, 0_ic) * scale);
}

__tile_global__ void softmax_distillation_partials_float32_kernel(
    const float* __restrict__ teacher_logits,
    const float* __restrict__ student_logits,
    float* __restrict__ partials,
    std::int64_t rows,
    std::int64_t vocab) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  teacher_logits = ct::assume_aligned(teacher_logits, 16_ic);
  student_logits = ct::assume_aligned(student_logits, 16_ic);
  partials = ct::assume_aligned(partials, 16_ic);

  const int row_id = ct::bid().x;
  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto col = ct::iota<IndexTile>();
  auto mask = col < ct::full<IndexTile>(vocab);
  auto base = ct::full<IndexTile>(static_cast<std::int64_t>(row_id)) * ct::full<IndexTile>(vocab);
  auto teacher = ct::load_masked(teacher_logits + base + col, mask);
  auto student = ct::load_masked(student_logits + base + col, mask);
  auto teacher_exp = ct::select(mask, ct::exp(teacher), ct::full<decltype(teacher)>(0.0f));
  auto student_exp = ct::select(mask, ct::exp(student), ct::full<decltype(student)>(0.0f));
  auto teacher_logsum = ct::log(ct::sum(teacher_exp, 0_ic));
  auto student_logsum = ct::log(ct::sum(student_exp, 0_ic));
  auto teacher_prob = ct::exp(teacher - teacher_logsum);
  auto kl = teacher_prob * ((teacher - teacher_logsum) - (student - student_logsum));
  kl = ct::select(mask, kl, ct::full<decltype(kl)>(0.0f));
  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(static_cast<std::int64_t>(row_id));
  ct::store(partials + out_idx, ct::sum(kl, 0_ic));
}

__tile_global__ void scaled_dot_product_attention_float32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
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
    std::int64_t compress_stride) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  q = ct::assume_aligned(q, 16_ic);
  k = ct::assume_aligned(k, 16_ic);
  v = ct::assume_aligned(v, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  const std::int64_t out_idx_scalar = static_cast<std::int64_t>(ct::bid().x);
  if (out_idx_scalar >= n) {
    return;
  }

  const std::int64_t d_out = out_idx_scalar % value_dim;
  const std::int64_t q_pos = (out_idx_scalar / value_dim) % seq_q;
  const std::int64_t q_head = (out_idx_scalar / (value_dim * seq_q)) % query_heads;
  const std::int64_t batch = out_idx_scalar / (value_dim * seq_q * query_heads);
  const std::int64_t k_head = (q_head * key_heads) / query_heads;
  const std::int64_t q_base_scalar = ((batch * query_heads + q_head) * seq_q + q_pos) * qk_dim;
  const std::int64_t k_base_scalar = (batch * key_heads + k_head) * seq_k * qk_dim;
  const std::int64_t v_base_scalar = (batch * key_heads + k_head) * seq_k * value_dim;

  using IndexTile = ct::tile<std::int64_t, decltype(ct::shape{1024_ic})>;
  auto key_pos = ct::iota<IndexTile>();
  auto valid = key_pos < ct::full<IndexTile>(seq_k);
  if (is_causal) {
    const std::int64_t offset = right_align_causal ? (seq_k - seq_q) : 0;
    valid = valid & (key_pos <= ct::full<IndexTile>(q_pos + offset));
  }
  if (use_sparse_rules) {
    const std::int64_t offset = right_align_causal ? (seq_k - seq_q) : 0;
    auto keep = ct::full<decltype(valid)>(false);
    bool any_rule = false;
    if (window > 0) {
      keep = keep | (key_pos > ct::full<IndexTile>(q_pos + offset - window));
      any_rule = true;
    }
    if (num_sinks > 0) {
      keep = keep | (key_pos < ct::full<IndexTile>(num_sinks));
      any_rule = true;
    }
    if (block_size > 0) {
      keep = keep | (ct::full<IndexTile>((q_pos + offset) / block_size) == (key_pos / ct::full<IndexTile>(block_size)));
      any_rule = true;
    }
    if (compress_stride > 1) {
      keep = keep | ((key_pos % ct::full<IndexTile>(compress_stride)) == ct::full<IndexTile>(0));
      any_rule = true;
    }
    if (any_rule) {
      valid = valid & keep;
    }
  }

  auto score = ct::full<ct::tile<float, decltype(ct::shape{1024_ic})>>(0.0f);
  for (std::int64_t d = 0; d < qk_dim; ++d) {
    auto q_val = ct::full<decltype(score)>(q[q_base_scalar + d]);
    auto k_val = ct::load_masked(k + ct::full<IndexTile>(k_base_scalar + d) + key_pos * ct::full<IndexTile>(qk_dim), valid);
    score = score + q_val * k_val;
  }
  score = score * ct::full<decltype(score)>(scale);
  auto neg_inf = ct::full<decltype(score)>(-3.4028234663852886e38f);
  auto safe_score = ct::select(valid, score, neg_inf);
  auto max_score = ct::reduce_max(safe_score, 0_ic);
  auto exp_score = ct::select(valid, ct::exp(score - max_score), ct::full<decltype(score)>(0.0f));
  auto denom = ct::sum(exp_score, 0_ic);
  auto v_value = ct::load_masked(
      v + ct::full<IndexTile>(v_base_scalar + d_out) + key_pos * ct::full<IndexTile>(value_dim),
      valid);
  auto weighted = exp_score * v_value;
  auto result = ct::sum(weighted, 0_ic) / denom;

  using OneIndexTile = ct::tile<std::int64_t, decltype(ct::shape{1_ic})>;
  auto out_idx = ct::full<OneIndexTile>(out_idx_scalar);
  ct::store(out + out_idx, result);
}

__device__ float deterministic_uniform_value(std::int64_t linear_idx, std::int64_t counter, std::int64_t salt) {
  const std::uint64_t modulus = 16777216ULL;
  std::uint64_t value =
      (static_cast<std::uint64_t>(linear_idx) * 1103515245ULL +
       static_cast<std::uint64_t>(counter) * 12345ULL +
       static_cast<std::uint64_t>(salt)) %
      modulus;
  return static_cast<float>(value) / 16777216.0f;
}

__global__ void random_timesteps_float32_kernel(float* out, std::int64_t batch, std::int64_t counter) {
  const std::int64_t base = static_cast<std::int64_t>(blockIdx.x) * kTileSize;
  for (std::int64_t offset = 0; offset < kTileSize; ++offset) {
    const std::int64_t idx = base + offset;
    if (idx < batch) {
      out[idx] = deterministic_uniform_value(idx, counter, 17);
    }
  }
}

__global__ void mask_scheduler_int64_kernel(
    const std::int64_t* tokens,
    const float* timesteps,
    std::int64_t* out,
    std::int64_t n,
    std::int64_t seq_len,
    std::int64_t mask_token_id,
    std::int64_t counter) {
  const std::int64_t base = static_cast<std::int64_t>(blockIdx.x) * kTileSize;
  for (std::int64_t offset = 0; offset < kTileSize; ++offset) {
    const std::int64_t idx = base + offset;
    if (idx < n) {
      const std::int64_t batch = idx / seq_len;
      const float noise = deterministic_uniform_value(idx, counter, 53);
      out[idx] = noise < timesteps[batch] ? mask_token_id : tokens[idx];
    }
  }
}

__global__ void jepa_mask_int64_kernel(
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
    std::int64_t counter) {
  const std::int64_t base = static_cast<std::int64_t>(blockIdx.x) * kTileSize;
  for (std::int64_t offset = 0; offset < kTileSize; ++offset) {
    const std::int64_t idx = base + offset;
    if (idx < n) {
      const std::int64_t batch = idx / seq_len;
      const std::int64_t pos = idx - batch * seq_len;
      bool is_masked = false;
      if (strategy == 1) {
        const std::int64_t min_len = max(static_cast<std::int64_t>(1), static_cast<std::int64_t>(min_block_ratio * seq_len));
        const std::int64_t max_len = max(min_len, static_cast<std::int64_t>(max_block_ratio * seq_len));
        const std::int64_t len_span = max(static_cast<std::int64_t>(1), max_len - min_len + 1);
        for (std::int64_t block = 0; block < num_blocks; ++block) {
          const float block_noise = deterministic_uniform_value(batch, counter, 101 + block * 2);
          const float start_noise = deterministic_uniform_value(batch, counter, 102 + block * 2);
          const std::int64_t block_len = min_len + min(len_span - 1, static_cast<std::int64_t>(block_noise * len_span));
          const std::int64_t max_start = max(static_cast<std::int64_t>(0), seq_len - block_len);
          const std::int64_t start = min(max_start, static_cast<std::int64_t>(start_noise * (static_cast<float>(max_start) + 1.0f)));
          if (pos >= start && pos < start + block_len) {
            is_masked = true;
          }
        }
      } else {
        is_masked = deterministic_uniform_value(idx, counter, 29) < mask_ratio;
      }
      masked_tokens[idx] = is_masked ? mask_token_id : tokens[idx];
      mask_values[idx] = is_masked ? 1.0f : 0.0f;
    }
  }
}

}  // namespace

void launch_unary_float32(const float* x, float* out, std::int64_t n, int op, cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  unary_float32_kernel<<<blocks, 1, 0, stream>>>(x, out, n, op);
}

void launch_binary_float32(
    const float* lhs,
    const float* rhs,
    float* out,
    std::int64_t n,
    int op,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  binary_float32_kernel<<<blocks, 1, 0, stream>>>(lhs, rhs, out, n, op);
}

void launch_binary_pair_float32(
    const float* lhs,
    const float* rhs,
    float* out0,
    float* out1,
    std::int64_t n,
    int op,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  binary_pair_float32_kernel<<<blocks, 1, 0, stream>>>(lhs, rhs, out0, out1, n, op);
}

void launch_scalar_unary_float32(
    const float* x,
    float* out,
    std::int64_t n,
    float value,
    int op,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  scalar_unary_float32_kernel<<<blocks, 1, 0, stream>>>(x, out, n, value, op);
}

void launch_scalar_binary_float32(
    const float* lhs,
    const float* rhs,
    float* out,
    std::int64_t n,
    float value,
    int op,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  scalar_binary_float32_kernel<<<blocks, 1, 0, stream>>>(lhs, rhs, out, n, value, op);
}

void launch_ema_update_float32(
    float* target,
    const float* source,
    std::int64_t n,
    float decay,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  ema_update_float32_kernel<<<blocks, 1, 0, stream>>>(target, source, n, decay);
}

void launch_gradient_accumulate_float32(
    float* buffer,
    const float* grad,
    std::int64_t n,
    float scale,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  gradient_accumulate_float32_kernel<<<blocks, 1, 0, stream>>>(buffer, grad, n, scale);
}

void launch_sumsq_partials_float32(
    const float* values,
    float* partials,
    std::int64_t n,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  sumsq_partials_float32_kernel<<<blocks, 1, 0, stream>>>(values, partials, n);
}

void launch_scale_inplace_float32(
    float* values,
    std::int64_t n,
    float scale,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  scale_inplace_float32_kernel<<<blocks, 1, 0, stream>>>(values, n, scale);
}

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
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  adamw_step_float32_kernel<<<blocks, 1, 0, stream>>>(
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
      sqrt_bias_correction2);
}

void launch_scalar_ternary_float32(
    const float* a,
    const float* b,
    const float* c,
    float* out,
    std::int64_t n,
    float value,
    int op,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  scalar_ternary_float32_kernel<<<blocks, 1, 0, stream>>>(a, b, c, out, n, value, op);
}

void launch_vector_binary_float32(
    const float* lhs,
    const float* rhs,
    const float* scale0,
    const float* scale1,
    float* out,
    std::int64_t n,
    std::int64_t dim,
    int op,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  vector_binary_float32_kernel<<<blocks, 1, 0, stream>>>(lhs, rhs, scale0, scale1, out, n, dim, op);
}

void launch_qk_gain_float32(
    const float* q,
    const float* gain,
    float* out,
    std::int64_t n,
    std::int64_t heads,
    std::int64_t inner,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  qk_gain_float32_kernel<<<blocks, 1, 0, stream>>>(q, gain, out, n, heads, inner);
}

void launch_dyt_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t n,
    std::int64_t dim,
    float alpha,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  dyt_float32_kernel<<<blocks, 1, 0, stream>>>(x, weight, bias, out, n, dim, alpha);
}

void launch_reshape_heads_float32(
    const float* x,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t heads,
    std::int64_t head_dim,
    cudaStream_t stream) {
  const std::int64_t n = batch * seq_len * heads * head_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  reshape_heads_float32_kernel<<<blocks, 1, 0, stream>>>(x, out, n, seq_len, heads, head_dim);
}

void launch_merge_heads_float32(
    const float* x,
    float* out,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    cudaStream_t stream) {
  const std::int64_t n = batch * heads * seq_len * head_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  merge_heads_float32_kernel<<<blocks, 1, 0, stream>>>(x, out, n, heads, seq_len, head_dim);
}

void launch_repeat_kv_float32(
    const float* x,
    float* out,
    std::int64_t batch,
    std::int64_t kv_heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    std::int64_t repeats,
    cudaStream_t stream) {
  const std::int64_t n = batch * kv_heads * repeats * seq_len * head_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  repeat_kv_float32_kernel<<<blocks, 1, 0, stream>>>(x, out, n, kv_heads, seq_len, head_dim, repeats);
}

void launch_broadcast_expert_routes_float32(
    const float* weights,
    const std::int64_t* indices,
    float* out_weights,
    std::int64_t* out_indices,
    std::int64_t batch,
    std::int64_t route_seq,
    std::int64_t seq_len,
    std::int64_t route_width,
    cudaStream_t stream) {
  const std::int64_t n = batch * seq_len * route_width;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  broadcast_expert_routes_float32_kernel<<<blocks, 1, 0, stream>>>(
      weights, indices, out_weights, out_indices, n, route_seq, seq_len, route_width);
}

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
    cudaStream_t stream) {
  const std::int64_t n = batch * seq_len * route_width;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  broadcast_chunk_routes_float32_kernel<<<blocks, 1, 0, stream>>>(
      weights, indices, out_weights, out_indices, n, chunks, seq_len, route_width, chunk_size);
}

void launch_byte_patch_merge_float32(
    const float* x,
    float* out,
    std::int64_t batch,
    std::int64_t source_len,
    std::int64_t target_len,
    std::int64_t dim,
    cudaStream_t stream) {
  const std::int64_t n = batch * target_len * dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  byte_patch_merge_float32_kernel<<<blocks, 1, 0, stream>>>(x, out, n, source_len, target_len, dim);
}

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
    cudaStream_t stream) {
  const std::int64_t n = batch * out_len * model_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  byte_patch_embed_float32_kernel<<<blocks, 1, 0, stream>>>(
      tokens, embedding, proj, out, n, seq_len, model_dim, patch_size, stride, out_len, vocab_size);
}

void launch_causal_chunk_state_float32(
    const float* hidden,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t dim,
    std::int64_t chunk_size,
    std::int64_t chunks,
    int mode,
    cudaStream_t stream) {
  const std::int64_t n = batch * chunks * dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  causal_chunk_state_float32_kernel<<<blocks, 1, 0, stream>>>(hidden, out, n, seq_len, dim, chunk_size, chunks, mode);
}

void launch_latent_mse_partials_float32(
    const float* pred,
    const float* target,
    float* partials,
    std::int64_t n,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  latent_mse_partials_float32_kernel<<<blocks, 1, 0, stream>>>(pred, target, partials, n);
}

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
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  semantic_alignment_loss_items_float32_kernel<<<blocks, 1, 0, stream>>>(
      logits, targets, term_counts, losses, counts, n, dims, terms, ignore_index);
}

void launch_sum_partials_float32(
    const float* values,
    float* partials,
    std::int64_t n,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  sum_partials_float32_kernel<<<blocks, 1, 0, stream>>>(values, partials, n);
}

void launch_scale_float32(const float* x, float* out, std::int64_t n, float value, cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  scalar_unary_float32_kernel<<<blocks, 1, 0, stream>>>(x, out, n, value, 0);
}

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
    cudaStream_t stream) {
  const std::int64_t n = batch * heads * (cache_seq + current_seq) * head_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  kv_cache_read_float32_kernel<<<blocks, 1, 0, stream>>>(
      k, v, cache_k, cache_v, out_k, out_v, n, heads, cache_seq, current_seq, head_dim);
}

void launch_kv_quant_pack_float32(
    const float* k,
    const float* v,
    float* out,
    std::int64_t rows,
    std::int64_t head_dim,
    cudaStream_t stream) {
  kv_quant_pack_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(k, v, out, head_dim);
}

void launch_kv_quant_unpack_float32(
    const float* packed,
    float* out_k,
    float* out_v,
    std::int64_t rows,
    std::int64_t head_dim,
    cudaStream_t stream) {
  const std::int64_t n = rows * head_dim * 2;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  kv_quant_unpack_float32_kernel<<<blocks, 1, 0, stream>>>(packed, out_k, out_v, n, head_dim);
}

void launch_absolute_position_embedding_float32(
    const float* weight,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t model_dim,
    cudaStream_t stream) {
  const std::int64_t n = batch * seq_len * model_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  absolute_position_embedding_float32_kernel<<<blocks, 1, 0, stream>>>(weight, out, n, seq_len, model_dim);
}

void launch_token_embedding_float32(
    const float* weight,
    const std::int64_t* token_ids,
    float* out,
    std::int64_t tokens,
    std::int64_t model_dim,
    cudaStream_t stream) {
  const std::int64_t n = tokens * model_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  token_embedding_float32_kernel<<<blocks, 1, 0, stream>>>(weight, token_ids, out, n, model_dim);
}

void launch_rotary_embedding_float32(
    const float* x,
    const float* inv_freq,
    float* out,
    std::int64_t batch,
    std::int64_t heads,
    std::int64_t seq_len,
    std::int64_t head_dim,
    cudaStream_t stream) {
  const std::int64_t n = batch * heads * seq_len * head_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  rotary_embedding_float32_kernel<<<blocks, 1, 0, stream>>>(x, inv_freq, out, n, heads, seq_len, head_dim);
}

void launch_rms_norm_float32(
    const float* x,
    float* out,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    cudaStream_t stream) {
  rms_norm_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(x, out, rows, dim, eps);
}

void launch_layer_norm_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t dim,
    float eps,
    cudaStream_t stream) {
  layer_norm_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(x, weight, bias, out, rows, dim, eps);
}

void launch_softmax_lastdim_float32(
    const float* x,
    float* out,
    std::int64_t rows,
    std::int64_t dim,
    cudaStream_t stream) {
  softmax_lastdim_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(x, out, rows, dim);
}

void launch_semantic_hash_int64(
    const float* sem_vec,
    const float* proj,
    std::int64_t* out,
    std::int64_t batch,
    std::int64_t dim,
    std::int64_t tables,
    std::int64_t planes,
    cudaStream_t stream) {
  semantic_hash_int64_kernel<<<static_cast<int>(batch * tables), 1, 0, stream>>>(
      sem_vec, proj, out, batch, dim, tables, planes);
}

void launch_topk_route_float32(
    const float* logits,
    float* weights,
    std::int64_t* indices,
    std::int64_t rows,
    std::int64_t experts,
    std::int64_t top_k,
    cudaStream_t stream) {
  topk_route_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      logits, weights, indices, rows, experts, top_k);
}

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
    cudaStream_t stream) {
  const std::int64_t n = batch * vocab_size;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  attentionless_decoder_float32_kernel<<<blocks, 1, 0, stream>>>(
      bucket_indices, expert_output, bucket_embed, out_weight, out, n, residual_dim, vocab_size, n_buckets);
}

void launch_expert_bias_add_float32(
    const float* logits,
    const float* bias,
    float* out,
    std::int64_t n,
    std::int64_t experts,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  expert_bias_add_float32_kernel<<<blocks, 1, 0, stream>>>(logits, bias, out, n, experts);
}

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
    cudaStream_t stream) {
  group_norm_float32_kernel<<<static_cast<int>(batch * num_groups), 1, 0, stream>>>(
      x, weight, bias, out, seq_len, dim, num_groups, eps);
}

void launch_scaled_residual_add_float32(
    const float* lhs,
    const float* rhs,
    const float* scale,
    float* out,
    std::int64_t n,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  scaled_residual_add_float32_kernel<<<blocks, 1, 0, stream>>>(lhs, rhs, scale, out, n);
}

void launch_linear_float32(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::int64_t rows,
    std::int64_t input_dim,
    std::int64_t output_dim,
    bool has_bias,
    cudaStream_t stream) {
  const std::int64_t n = rows * output_dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  linear_float32_kernel<<<blocks, 1, 0, stream>>>(x, weight, bias, out, n, input_dim, output_dim, has_bias);
}

void launch_act_weighted_sum_float32(
    const float* states,
    const float* weights,
    float* out,
    std::int64_t batch,
    std::int64_t steps,
    std::int64_t inner,
    cudaStream_t stream) {
  const std::int64_t n = batch * inner;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  act_weighted_sum_float32_kernel<<<blocks, 1, 0, stream>>>(states, weights, out, n, steps, inner);
}

void launch_latent_pool_float32(
    const float* x,
    const float* mask_values,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t dim,
    cudaStream_t stream) {
  const std::int64_t n = batch * dim;
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  latent_pool_float32_kernel<<<blocks, 1, 0, stream>>>(x, mask_values, out, n, seq_len, dim);
}

void launch_token_cross_entropy_partials_float32(
    const float* logits,
    const std::int64_t* targets,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((rows + kTileSize - 1) / kTileSize);
  token_cross_entropy_partials_float32_kernel<<<blocks, 1, 0, stream>>>(logits, targets, partials, rows, vocab);
}

void launch_masked_token_cross_entropy_partials_float32(
    const float* logits,
    const std::int64_t* targets,
    const float* loss_mask,
    float* loss_partials,
    float* mask_partials,
    std::int64_t rows,
    std::int64_t vocab,
    std::int64_t ignore_index,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((rows + kTileSize - 1) / kTileSize);
  masked_token_cross_entropy_partials_float32_kernel<<<blocks, 1, 0, stream>>>(
      logits, targets, loss_mask, loss_partials, mask_partials, rows, vocab, ignore_index);
}

void launch_sequence_logp_float32(
    const float* logits,
    const std::int64_t* targets,
    const float* loss_mask,
    float* out,
    std::int64_t batch,
    std::int64_t seq_len,
    std::int64_t vocab,
    std::int64_t ignore_index,
    cudaStream_t stream) {
  sequence_logp_float32_kernel<<<1, 1, 0, stream>>>(logits, targets, loss_mask, out, batch, seq_len, vocab, ignore_index);
}

void launch_preference_bce_partials_float32(
    const float* reward_chosen,
    const float* reward_rejected,
    float* partials,
    std::int64_t n,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  preference_bce_partials_float32_kernel<<<blocks, 1, 0, stream>>>(reward_chosen, reward_rejected, partials, n);
}

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
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  ppo_clipped_loss_partials_float32_kernel<<<blocks, 1, 0, stream>>>(
      logp_new, logp_old, advantages, value_new, value_old, returns, policy_partials, value_partials, n, clip_range);
}

void launch_gae_compute_float32(
    const float* rewards,
    const float* values,
    float* advantages,
    float* returns,
    std::int64_t batch,
    std::int64_t seq_len,
    float gamma,
    float lambda_value,
    cudaStream_t stream) {
  gae_compute_float32_kernel<<<static_cast<int>(batch), 1, 0, stream>>>(
      rewards, values, advantages, returns, seq_len, gamma, lambda_value);
}

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
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  dpo_pairwise_partials_float32_kernel<<<blocks, 1, 0, stream>>>(
      policy_logp_chosen,
      policy_logp_rejected,
      ref_logp_chosen,
      ref_logp_rejected,
      partials,
      chosen_reward_out,
      rejected_reward_out,
      n,
      beta,
      label_smoothing,
      loss_type);
}

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
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  route_selection_loss_partials_float32_kernel<<<blocks, 1, 0, stream>>>(
      route_logits,
      sem_targets,
      loss_partials,
      count_partials,
      n,
      seq_len,
      experts,
      num_vocab_dims,
      shared_experts,
      ignore_index);
}

void launch_route_balance_density_float32(
    const float* route_logits,
    float* density,
    std::int64_t rows,
    std::int64_t experts,
    cudaStream_t stream) {
  route_balance_density_float32_kernel<<<static_cast<int>(experts), 1, 0, stream>>>(route_logits, density, rows, experts);
}

void launch_route_balance_loss_float32(
    const float* density,
    float* out,
    std::int64_t experts,
    cudaStream_t stream) {
  route_balance_loss_float32_kernel<<<1, 1, 0, stream>>>(density, out, experts);
}

void launch_softmax_distillation_partials_float32(
    const float* teacher_logits,
    const float* student_logits,
    float* partials,
    std::int64_t rows,
    std::int64_t vocab,
    cudaStream_t stream) {
  softmax_distillation_partials_float32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      teacher_logits, student_logits, partials, rows, vocab);
}

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
    cudaStream_t stream) {
  scaled_dot_product_attention_float32_kernel<<<static_cast<int>(n), 1, 0, stream>>>(
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
      compress_stride);
}

void launch_random_timesteps_float32(float* out, std::int64_t batch, std::int64_t counter, cudaStream_t stream) {
  const int blocks = static_cast<int>((batch + kTileSize - 1) / kTileSize);
  random_timesteps_float32_kernel<<<blocks, 1, 0, stream>>>(out, batch, counter);
}

void launch_mask_scheduler_int64(
    const std::int64_t* tokens,
    const float* timesteps,
    std::int64_t* out,
    std::int64_t n,
    std::int64_t seq_len,
    std::int64_t mask_token_id,
    std::int64_t counter,
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  mask_scheduler_int64_kernel<<<blocks, 1, 0, stream>>>(tokens, timesteps, out, n, seq_len, mask_token_id, counter);
}

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
    cudaStream_t stream) {
  const int blocks = static_cast<int>((n + kTileSize - 1) / kTileSize);
  jepa_mask_int64_kernel<<<blocks, 1, 0, stream>>>(
      tokens, masked_tokens, mask_values, n, seq_len, mask_ratio, mask_token_id, strategy, num_blocks, min_block_ratio, max_block_ratio, counter);
}

}  // namespace neuralfn::tile_cuda
