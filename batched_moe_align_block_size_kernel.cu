#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>

#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

namespace vllm {
namespace moe {
namespace batched_moe_align_block_size {

static constexpr int32_t num_threads = 1024;
static constexpr int32_t num_blocks = 1;
__global__ void batched_moe_align_block_size_kernel(
    int32_t const num_batches, int32_t const max_tokens_per_batch,
    int32_t const block_size, int32_t const* __restrict__ batch_num_tokens,
    int32_t* __restrict__ sorted_ids, int32_t* __restrict__ block_ids,
    int32_t* __restrict__ num_tokens_post_pad) {
  size_t const batch_id = threadIdx.x;
  size_t const stride = blockDim.x * gridDim.x;
  int32_t const num_blocks_per_batch =
      CEILDIV(max_tokens_per_batch, block_size);
  int32_t const sorted_ids_size =
      num_blocks_per_batch * num_batches * block_size;
  int32_t const block_ids_size = sorted_ids_size / block_size;
  int32_t const SENTINEL =
      num_batches * max_tokens_per_batch;
  for (size_t i = threadIdx.x; i < sorted_ids_size; i += stride) {
    sorted_ids[i] = SENTINEL;
  }
  for (size_t i = threadIdx.x; i < block_ids_size; i += stride) {
    block_ids[i] = -1;
  }

  int32_t b_num_tokens = 0;
  if (batch_id < num_batches) {
    b_num_tokens = batch_num_tokens[batch_id];
  }
  int32_t const ceil_b_num_tokens =
      CEILDIV(b_num_tokens, block_size) * block_size;

  using BlockScan = cub::BlockScan<int32_t, 1024>;
  __shared__ typename BlockScan::TempStorage temp_storage;
  int cumsum_val;
  BlockScan(temp_storage).ExclusiveSum(ceil_b_num_tokens, cumsum_val);
  __syncthreads();

  bool const is_last_batch = batch_id == (num_batches - 1);
  if (is_last_batch) {
    *num_tokens_post_pad = cumsum_val + ceil_b_num_tokens;
  }

  if (batch_id < num_batches) {
    int32_t const batch_offset = batch_id * max_tokens_per_batch;
    for (size_t i = 0; i < b_num_tokens; ++i) {
      sorted_ids[cumsum_val + i] = batch_offset + i;
    }

    int32_t const block_start = cumsum_val / block_size;
    int32_t const num_blocks = ceil_b_num_tokens / block_size;
    for (size_t i = 0; i < num_blocks; ++i) {
      block_ids[block_start + i] = batch_id;
    }
  }
}

}  // namespace batched_moe_align_block_size
}  // namespace moe
}  // namespace vllm
