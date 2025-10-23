#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <c10/util/Float8_e4m3fn.h>
#include <array>
#include <type_traits>
#ifndef USE_ROCM
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
  #include <cuda_fp8.h>
#else
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>
  #include <hip/hip_fp8.h>

typedef __hip_bfloat162 __nv_bfloat162;
typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat16_raw __nv_bfloat16_raw;
  #if defined(HIP_FP8_TYPE_OCP)
typedef __hip_fp8_e4m3 __nv_fp8_e4m3;
typedef __hip_fp8x4_e4m3 __nv_fp8x4_e4m3;
  #else
// ROCm 6.2 fallback: only *_fnuz types exist
typedef __hip_fp8_e4m3_fnuz __nv_fp8_e4m3;
typedef __hip_fp8x4_e4m3_fnuz __nv_fp8x4_e4m3;
  #endif
#endif

namespace vllm {

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          typename fp8_type>
__global__ void act_and_mul_quant_kernel(
    fp8_type* __restrict__ out,
    const scalar_t* __restrict__ input,
    const float* scale,
    const int d) {
  const int32_t blocks_per_token = gridDim.y;

  const int32_t elems_per_128bit_load = (128 / 8) / sizeof(scalar_t);

  const int32_t tgt_elems_per_block = div_ceil(d, blocks_per_token);
  const int32_t elems_per_block =
      round_to_next_multiple_of(tgt_elems_per_block, elems_per_128bit_load);
  const int32_t block_start = blockIdx.y * elems_per_block;
  int32_t block_end = block_start + elems_per_block;
  block_end = block_end > d ? d : block_end;

  const int64_t token_idx = blockIdx.x;
  const scalar_t* __restrict__ x_ptr = input + token_idx * 2 * d;
  const scalar_t* __restrict__ y_ptr = input + token_idx * 2 * d + d;
  fp8_type* __restrict__ out_ptr = out + token_idx * d;

  const int32_t vec_loop_end =
      round_to_previous_multiple_of(elems_per_128bit_load, block_end);
  const int32_t vec_end_idx = vec_loop_end / elems_per_128bit_load;
  const int32_t vec_start_idx = block_start / elems_per_128bit_load;

  const int4* __restrict__ x_128bit_ptr = reinterpret_cast<const int4*>(x_ptr);
  const int4* __restrict__ y_128bit_ptr = reinterpret_cast<const int4*>(y_ptr);
  int2* __restrict__ out_128bit_ptr = reinterpret_cast<int2*>(out_ptr);

  float inverted_scale = 1 / *scale;
#pragma unroll
  for (int32_t vec_idx = vec_start_idx + threadIdx.x; vec_idx < vec_end_idx;
       vec_idx += blockDim.x) {
    const int4 x_128bit = VLLM_LDG(&x_128bit_ptr[vec_idx]);
    const int4 y_128bit = VLLM_LDG(&y_128bit_ptr[vec_idx]);
    using scalar_128bit_vec_t = std::array<scalar_t, elems_per_128bit_load>;
    using scalar_64bit_vec_t = std::array<fp8_type, elems_per_128bit_load>;

    scalar_64bit_vec_t out_vec;
    const auto x_vec = reinterpret_cast<scalar_128bit_vec_t const&>(x_128bit);
    const auto y_vec = reinterpret_cast<scalar_128bit_vec_t const&>(y_128bit);

#pragma unroll
    for (int i = 0; i < elems_per_128bit_load; i++) {
      out_vec[i] = scaled_fp8_conversion<true, fp8_type>(
          ACT_FN(x_vec[i]) * y_vec[i], inverted_scale);
    }

    out_128bit_ptr[vec_idx] = reinterpret_cast<const int2&>(out_vec);
  }

  if (block_end > vec_loop_end) {
    for (int64_t idx = vec_loop_end + threadIdx.x; idx < block_end;
         idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&x_ptr[idx]);
      const scalar_t y = VLLM_LDG(&y_ptr[idx]);
      out_ptr[idx] =
          scaled_fp8_conversion<true, fp8_type>(ACT_FN(x) * y, inverted_scale);
    }
  }
}

}  // namespace vllm
