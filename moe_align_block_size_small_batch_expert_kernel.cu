#include <cub/cub.cuh>

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

namespace vllm {
namespace moe {

template <typename scalar_t>
__global__ void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t block_size, size_t numel, int32_t max_num_tokens_padded) {
  for (size_t it = threadIdx.x; it < max_num_tokens_padded; it += blockDim.x) {
    sorted_token_ids[it] = numel;
  }

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;
  int32_t* tokens_cnts = (int32_t*)(shared_mem + num_experts + 1);

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(threadIdx.x + 1) * num_experts + i] = 0;
  }

  for (size_t i = tid; i < numel; i += stride) {
    ++tokens_cnts[(threadIdx.x + 1) * num_experts + topk_ids[i]];
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    tokens_cnts[threadIdx.x] = 0;
    for (int i = 1; i <= blockDim.x; ++i) {
      tokens_cnts[i * num_experts + threadIdx.x] +=
          tokens_cnts[(i - 1) * num_experts + threadIdx.x];
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] =
          cumsum[i - 1] +
          CEILDIV(tokens_cnts[blockDim.x * num_experts + i - 1], block_size) *
              block_size;
    }
    *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
         i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }

  const size_t fill_start_idx = cumsum[num_experts] / block_size + threadIdx.x;
  const size_t expert_ids_size = CEILDIV(max_num_tokens_padded, block_size);
  for (size_t i = fill_start_idx; i < expert_ids_size; i += blockDim.x) {
    expert_ids[i] = 0;
  }

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    int32_t rank_post_pad =
        tokens_cnts[threadIdx.x * num_experts + expert_id] + cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    ++tokens_cnts[threadIdx.x * num_experts + expert_id];
  }
}

}  // namespace moe
}  // namespace vllm
