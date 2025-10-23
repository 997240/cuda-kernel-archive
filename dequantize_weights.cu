#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdint.h>

namespace vllm {
namespace awq {

__device__ __forceinline__ uint4 dequantize_s4_to_fp16x2(uint32_t packed_values) {
  union {
    uint4 u32x4;
    half2 h2[4];
  } result;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int low = (packed_values >> (8 * i)) & 0xF;
    int high = (packed_values >> (8 * i + 4)) & 0xF;
    low = (low << 28) >> 28;
    high = (high << 28) >> 28;
    result.h2[i] = __halves2half2(__int2half_rn(low), __int2half_rn(high));
  }
  return result.u32x4;
}

__global__ void __launch_bounds__(64)
    dequantize_weights(int* __restrict__ B, half* __restrict__ scaling_factors,
                       int* __restrict__ zeros, half* __restrict__ C, int G) {
  static constexpr uint32_t ZERO = 0x0;
  half B_shared[32 * (128 + 8)];

  half* B_shared_ptr2 = B_shared;

  int N = blockDim.x * gridDim.x;
  int col = (blockIdx.x * blockDim.x + threadIdx.x);
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int index1 = 8 * col + 8 * row * N;
  half* C_ptr2 = C + index1;

  int index2 = col + row * N;
  int* B_ptr2 = B + index2;

  int index3 = col + (int)(row / G) * N;
  int* zeros_ptr2 = zeros + index3;
  int index4 = 8 * col + (int)(row / G) * N * 8;
  half* scaling_factors_ptr2 = scaling_factors + index4;

  uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr2);
  uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);
  uint4 B_loaded_scale = *(uint4*)(scaling_factors_ptr2);

  uint32_t B_loaded = *(uint32_t*)B_ptr2;
  uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(B_loaded_fp16.x)
               : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(B_loaded_fp16.x)
               : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x), "r"(ZERO));
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(B_loaded_fp16.y)
               : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.y));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(B_loaded_fp16.y)
               : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.y), "r"(ZERO));
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(B_loaded_fp16.z)
               : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.z));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(B_loaded_fp16.z)
               : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.z), "r"(ZERO));
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(B_loaded_fp16.w)
               : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(B_loaded_fp16.w)
               : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w), "r"(ZERO));

  *(uint4*)B_shared_ptr2 = B_loaded_fp16;

  for (int i = 0; i < 8; ++i) {
    *(C_ptr2 + i) = B_shared[i];
  }
}

}  // namespace awq
}  // namespace vllm
