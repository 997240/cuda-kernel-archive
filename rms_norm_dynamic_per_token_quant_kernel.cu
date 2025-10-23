namespace vllm {

template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
__device__ void rms_norm_dynamic_per_token_quant_vec(
    scalar_out_t* __restrict__ out,
    float* __restrict__ scales,
    scalar_t const* __restrict__ input,
    scalar_t const* __restrict__ weight,
    float const* scale_ub, float const var_epsilon, int32_t const hidden_size,
    scalar_t* __restrict__ residual = nullptr) {
  float rms = 0.0f;
  float token_scale = 0.0f;

  vllm::vectorized::compute_rms<scalar_t, has_residual>(
      &rms, input, hidden_size, var_epsilon, residual);

  vllm::vectorized::compute_dynamic_per_token_scales<scalar_t, scalar_out_t,
                                                     has_residual>(
      &token_scale, scales, input, weight, rms, scale_ub, hidden_size,
      residual);

  if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
    vllm::vectorized::norm_and_quant<scalar_t, scalar_out_t, true,
                                     has_residual>(
        out, input, weight, rms, 1.0f / token_scale, hidden_size, residual);
  } else {
    vllm::vectorized::norm_and_quant<scalar_t, scalar_out_t, false,
                                     has_residual>(
        out, input, weight, rms, token_scale, hidden_size, residual);
  }
}

template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
__global__ void rms_norm_dynamic_per_token_quant_kernel(
    scalar_out_t* __restrict__ out,
    float* __restrict__ scales,
    scalar_t const* __restrict__ input,
    scalar_t const* __restrict__ weight,
    float const* scale_ub, float const var_epsilon, int32_t const hidden_size,
    scalar_t* __restrict__ residual) {
  bool const can_vectorize = hidden_size % 4 == 0;

  if (can_vectorize) {
    return rms_norm_dynamic_per_token_quant_vec<scalar_t, scalar_out_t,
                                                has_residual>(
        out, scales, input, weight, scale_ub, var_epsilon, hidden_size,
        residual);
  }

  float rms = 0.0f;
  float token_scale = 0.0f;

  vllm::compute_rms<scalar_t, has_residual>(&rms, input, hidden_size,
                                            var_epsilon, residual);
  vllm::compute_dynamic_per_token_scales<scalar_t, scalar_out_t, has_residual>(
      &token_scale, scales, input, weight, rms, scale_ub, hidden_size,
      residual);

  if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
    vllm::norm_and_quant<scalar_t, scalar_out_t, true, has_residual>(
        out, input, weight, rms, 1.0f / token_scale, hidden_size, residual);
  } else {
    vllm::norm_and_quant<scalar_t, scalar_out_t, false, has_residual>(
        out, input, weight, rms, token_scale, hidden_size, residual);
  }
}
}  // namespace vllm
