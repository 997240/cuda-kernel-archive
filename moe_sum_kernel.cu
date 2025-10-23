#include <cub/cub.cuh>

#ifndef VLLM_LDG
#define VLLM_LDG(ptr) __ldg(ptr)
#endif

namespace vllm {
namespace moe {

template <typename scalar_t, int TOPK>
__global__ void moe_sum_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input,
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    scalar_t x = 0.0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += VLLM_LDG(&input[token_idx * TOPK * d + k * d + idx]);
    }
    out[token_idx * d + idx] = x;
  }
}

}  // namespace moe
}  // namespace vllm
