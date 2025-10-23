#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>

namespace vllm {

/* Generic fused_add_rms_norm_kernel
   The width field is not used here but necessary for other specializations.
 */
template <typename scalar_t, int width, typename fp8_type>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_static_fp8_quant_kernel(
    fp8_type* __restrict__ out,    // [..., hidden_size]
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float* __restrict__ scale,      // [1]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * input_stride + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = (float)z;
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  // invert scale to avoid division
  float const scale_inv = 1.0f / *scale;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * hidden_size + idx];
    float const out_norm = ((scalar_t)(x * s_variance)) * weight[idx];
    out[blockIdx.x * hidden_size + idx] =
        scaled_fp8_conversion<true, fp8_type>(out_norm, scale_inv);
  }
}

}  // namespace vllm
