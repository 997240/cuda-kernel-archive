#include <cuda_runtime.h>
#include <stdint.h>

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12000)

__global__ void preprocessTopkIdKernel(int* topk_id_ptr, int size,
                                       const int* expert_map_ptr,
                                       int num_experts) {
  auto tidx = threadIdx.x;
  auto bidx = blockIdx.x;
  auto offset = bidx * blockDim.x;
  auto bound = min(offset + blockDim.x, size);
  extern __shared__ int smem_expert_map[];
  for (int i = tidx; i < num_experts; i += blockDim.x) {
    smem_expert_map[i] = expert_map_ptr[i];
  }
  __syncthreads();

  if (offset + tidx < bound) {
    auto topk_id = topk_id_ptr[offset + tidx];
    auto local_expert_idx = smem_expert_map[topk_id];
    if (local_expert_idx == -1) {
      topk_id += num_experts;
    } else {
      topk_id = local_expert_idx;
    }
    __syncwarp();
    topk_id_ptr[offset + tidx] = topk_id;
  }
}

#endif
