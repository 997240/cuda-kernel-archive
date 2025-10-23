#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

namespace vllm {

__device__ __forceinline__ float silu(float x) {
  return __fdividef(x, (1.f + __expf(-x)));
}

__device__ __forceinline__ float2 silu2(float2 x) {
  return make_float2(silu(x.x), silu(x.y));
}

template <class Type>
__inline__ __device__ PackedVec<Type> compute_silu_mul(PackedVec<Type>& vec,
                                                       PackedVec<Type>& vec2) {
  PackedVec<Type> result;
  using packed_type = typename TypeConverter<Type>::Type;

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    if constexpr (std::is_same_v<Type, half>) {
      float2 silu_vec = silu2(__half22float2(vec.elts[i]));
      result.elts[i] =
          __float22half2_rn(__fmul2_rn(silu_vec, __half22float2(vec2.elts[i])));
    } else {
      float2 silu_vec = silu2(__bfloat1622float2(vec.elts[i]));
      result.elts[i] = __float22bfloat162_rn(
          __fmul2_rn(silu_vec, __bfloat1622float2(vec2.elts[i])));
    }
  }
  return result;
}

template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(1024, VLLM_BLOCKS_PER_SM(1024))
    silu_mul_cvt_fp16_to_fp4(int32_t numRows, int32_t numCols, Type const* in,
                             float const* SFScale, uint32_t* out,
                             uint32_t* SFout) {
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int colIdx = threadIdx.x; colIdx < numCols / CVT_FP4_ELTS_PER_THREAD;
         colIdx += blockDim.x) {
      int64_t inOffset =
          rowIdx * (numCols * 2 / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      int64_t inOffset2 = rowIdx * (numCols * 2 / CVT_FP4_ELTS_PER_THREAD) +
                          numCols / CVT_FP4_ELTS_PER_THREAD + colIdx;
      PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
      PackedVec in_vec2 = reinterpret_cast<PackedVec const*>(in)[inOffset2];

      int64_t outOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      auto& out_pos = out[outOffset];

      PackedVec out_silu_mul = compute_silu_mul(in_vec, in_vec2);

      auto sf_out =
          cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
              rowIdx, colIdx, numCols, SFout);

      out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(out_silu_mul, SFScaleVal,
                                                     sf_out);
    }
  }
}

}  // namespace vllm
