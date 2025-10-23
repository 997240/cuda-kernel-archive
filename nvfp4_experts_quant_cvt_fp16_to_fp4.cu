#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

namespace vllm {

template <class Type, bool UE8M0_SF = false, bool SMALL_NUM_EXPERTS = false>
__global__ void __launch_bounds__(1024, VLLM_BLOCKS_PER_SM(1024))
    cvt_fp16_to_fp4(int32_t numRows, int32_t numCols, Type const* in,
                    float const* SFScale, uint32_t* out, uint32_t* SFout,
                    uint32_t* input_offset_by_experts,
                    uint32_t* output_scale_offset_by_experts, int n_experts) {
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");
  extern __shared__ uint32_t shared_input_offsets[];

  if constexpr (SMALL_NUM_EXPERTS) {
    for (int i = threadIdx.x; i < n_experts + 1; i += blockDim.x) {
      shared_input_offsets[i] = input_offset_by_experts[i];
    }
  } else {
    for (int i = threadIdx.x * 4; i < n_experts; i += blockDim.x * 4) {
      *reinterpret_cast<int4*>(&shared_input_offsets[i]) =
          *reinterpret_cast<const int4*>(&input_offset_by_experts[i]);
    }
    if (threadIdx.x == 0) {
      shared_input_offsets[n_experts] = input_offset_by_experts[n_experts];
    }
  }

  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;

  for (int globalIdx = tid; globalIdx < numRows * colsPerRow;
       globalIdx += gridDim.x * blockDim.x) {
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;

    int64_t inOffset = rowIdx * colsPerRow + colIdx;
    PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
    int64_t outOffset = inOffset;
    auto& out_pos = out[outOffset];

    int rowIdx_in_expert = 0;
    int expert_idx = 0;

    int left = 0, right = n_experts - 1;
    while (left <= right) {
      int mid = (left + right) / 2;
      uint32_t mid_offset = shared_input_offsets[mid];
      uint32_t next_offset = shared_input_offsets[mid + 1];

      if (rowIdx >= mid_offset && rowIdx < next_offset) {
        rowIdx_in_expert = rowIdx - mid_offset;
        expert_idx = mid;
        break;
      } else if (rowIdx < mid_offset) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }

    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[expert_idx];

    int factor = CVT_FP4_SF_VEC_SIZE * 4;
    int32_t numCols_padded = (numCols + factor - 1) / factor * factor;
    int numCols_SFout = numCols_padded / CVT_FP4_SF_VEC_SIZE / 4;
    uint32_t* SFout_in_expert =
        SFout + output_scale_offset_by_experts[expert_idx] * numCols_SFout;

    auto sf_out =
        cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                           CVT_FP4_NUM_THREADS_PER_SF>(
            rowIdx_in_expert, colIdx, numCols, SFout_in_expert);

    out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
  }
}

}  // namespace vllm
