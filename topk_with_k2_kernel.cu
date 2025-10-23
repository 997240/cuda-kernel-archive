#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/std/limits>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace vllm {
namespace moe {

constexpr unsigned FULL_WARP_MASK = 0xffffffff;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t BLOCK_SIZE = 512;
constexpr int32_t NUM_WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

template <typename T_OUT, typename T_IN>
__device__ inline T_OUT cuda_cast(T_IN val) {
  return val;
}

template <>
__device__ inline float cuda_cast<float, __nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}

template <typename T>
__device__ inline T neg_inf() {
  return cuda_cast<T, float>(-cuda::std::numeric_limits<float>::infinity());
}

template <typename T>
__device__ void topk_with_k2(T* output, T const* input,
                             cg::thread_block_tile<32> const& tile,
                             int32_t const lane_id,
                             int const num_experts_per_group) {
  T largest = neg_inf<T>();
  T second_largest = neg_inf<T>();

  if (num_experts_per_group > WARP_SIZE) {
    for (int i = lane_id; i < num_experts_per_group; i += WARP_SIZE) {
      T value = input[i];
      if (value > largest) {
        second_largest = largest;
        largest = value;
      } else if (value > second_largest) {
        second_largest = value;
      }
    }
  } else {
    for (int i = lane_id; i < num_experts_per_group; i += WARP_SIZE) {
      largest = input[i];
    }
  }

  __syncwarp();
  T max1 = cg::reduce(tile, largest, cg::greater<T>());

  T max2 = max1;
  bool equal_to_max1 = (max1 == largest);

  int count_max1 = __popc(__ballot_sync(FULL_WARP_MASK, equal_to_max1));

  if (count_max1 == 1) {
    largest = (largest == max1) ? second_largest : largest;
    max2 = cg::reduce(tile, largest, cg::greater<T>());
  }

  if (lane_id == 0) {
    *output = max1 + max2;
  }
}

template <typename T>
__global__ void topk_with_k2_kernel(T* output, T* input,
                                    int64_t const num_tokens,
                                    int64_t const num_cases,
                                    int64_t const n_group,
                                    int64_t const num_experts_per_group) {
  int32_t warp_id = threadIdx.x / WARP_SIZE;
  int32_t lane_id = threadIdx.x % WARP_SIZE;

  int32_t case_id = blockIdx.x * NUM_WARPS_PER_BLOCK + warp_id;
  if (case_id < num_cases) {
    input += case_id * num_experts_per_group;
    output += case_id;

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    topk_with_k2(output, input, tile, lane_id, num_experts_per_group);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

}  // namespace moe
}  // namespace vllm
