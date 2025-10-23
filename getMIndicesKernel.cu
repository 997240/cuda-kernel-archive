#include <cuda_runtime.h>
#include <stdint.h>

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12000)

template <bool ALIGN_BLOCK_SIZE>
__global__ void getMIndicesKernel(int64_t* expert_first_token_offset,
                                  int64_t* align_expert_first_token_offset,
                                  int* m_indices, const int num_local_expert,
                                  const int align_block_size) {
  int eidx = blockIdx.x;
  int tidx = threadIdx.x;
  extern __shared__ int64_t smem_expert_first_token_offset[];
  for (int i = tidx; i <= num_local_expert; i += blockDim.x) {
    smem_expert_first_token_offset[i] = __ldg(expert_first_token_offset + i);
  }
  __syncthreads();
  auto last_token_offset = smem_expert_first_token_offset[eidx + 1];
  auto first_token_offset = smem_expert_first_token_offset[eidx];
  int n_token_in_expert = last_token_offset - first_token_offset;

  if constexpr (ALIGN_BLOCK_SIZE) {
    n_token_in_expert = (n_token_in_expert + align_block_size - 1) /
                        align_block_size * align_block_size;
    int64_t accumulate_align_offset = 0;
    for (int i = 1; i <= eidx + 1; i++) {
      int n_token = smem_expert_first_token_offset[i] -
                    smem_expert_first_token_offset[i - 1];
      accumulate_align_offset =
          accumulate_align_offset + (n_token + align_block_size - 1) /
                                        align_block_size * align_block_size;
      if (i == eidx) {
        first_token_offset = accumulate_align_offset;
      }
      if (eidx == num_local_expert - 1 && threadIdx.x == 0) {
        align_expert_first_token_offset[i] = accumulate_align_offset;
      }
    }
  }
  for (int idx = tidx; idx < n_token_in_expert; idx += blockDim.x) {
    m_indices[first_token_offset + idx] = eidx;
  }
}

#endif
