#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace allspark {

template <typename T>
__device__ __forceinline__ T zero_value() {
  return T();
}

template <typename T>
__device__ __forceinline__ void ldg16_cg_0(T& out, const T* ptr, bool guard) {
  out = guard ? *ptr : zero_value<T>();
}

__device__ __forceinline__ void ldg128_cg_0(uint32_t& out0, uint32_t& out1,
                                            uint32_t& out2, uint32_t& out3,
                                            const uint8_t* ptr, bool guard) {
  if (guard) {
    int4 data = *reinterpret_cast<const int4*>(ptr);
    const uint32_t* u32 = reinterpret_cast<const uint32_t*>(&data);
    out0 = u32[0];
    out1 = u32[1];
    out2 = u32[2];
    out3 = u32[3];
  } else {
    out0 = 0;
    out1 = 0;
    out2 = 0;
    out3 = 0;
  }
}

template <typename FType>
__global__ void __launch_bounds__(128)
    rearrange_kn_weight_as_n32k16_order_ldg16_kernel(
        const uint8_t* B, const FType* B_scale, const FType* B_zero,
        uint8_t* B_result, FType* B_scale_result, FType* B_zero_result,
        const int K, const int N, const int N_32align) {
  const auto lane_id = threadIdx.x % 32;
  const auto warp_id = threadIdx.x / 32;

  if (blockIdx.x != gridDim.x - 1) {
    const int src_row_base_idx =
        blockIdx.x * 64 + warp_id * 16 + ((lane_id % 8) / 2) * 2;
    const int src_col_idx =
        blockIdx.y * 128 + (lane_id / 8) * 32 + (lane_id % 2) * 16;
    uint8_t B_frag[4][16];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int src_row_idx = src_row_base_idx + (i / 2) * 8 + (i % 2);
      int src_offset = src_row_idx * N + src_col_idx;
      bool guard = src_row_idx < K && src_col_idx < N;
      ldg128_cg_0(*reinterpret_cast<uint32_t*>(B_frag[i]),
                  *(reinterpret_cast<uint32_t*>(B_frag[i]) + 1),
                  *(reinterpret_cast<uint32_t*>(B_frag[i]) + 2),
                  *(reinterpret_cast<uint32_t*>(B_frag[i]) + 3), B + src_offset,
                  guard);
    }

    uint8_t B_reorder_frag[8][8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        int dst_i = j % 8;
        int dst_j = i + (j / 8) * 4;
        B_reorder_frag[dst_i][dst_j] = B_frag[i][j];
      }
    }

    const auto dst_row_base_idx = blockIdx.y * (128 / 4) + (lane_id / 8) * 8;
    const int dst_col_idx =
        blockIdx.x * (64 * 4) + warp_id * 64 + (lane_id % 8) * 8;
    for (int i = 0; i < 8; ++i) {
      int dst_row_idx = dst_row_base_idx + i;
      int dst_offset = dst_row_idx * K * 4 + dst_col_idx;
      bool guard = (dst_row_base_idx < N_32align / 4) && (dst_col_idx < K * 4);
      if (guard) {
        *reinterpret_cast<int2*>(B_result + dst_offset) =
            *reinterpret_cast<int2*>(B_reorder_frag[i]);
      }
    }
  } else {
    FType b_scale_reg, b_zero_reg;
    auto src_offset = blockIdx.y * 128 + threadIdx.x;
    ldg16_cg_0(b_scale_reg, B_scale + src_offset, src_offset < N);
    if (B_zero != nullptr)
      ldg16_cg_0(b_zero_reg, B_zero + src_offset, src_offset < N);
    int dst_offset =
        blockIdx.y * 128 + warp_id * 32 + (lane_id % 8) * 4 + lane_id / 8;
    if (dst_offset < N_32align) {
      B_scale_result[dst_offset] = b_scale_reg;
      if (B_zero != nullptr) B_zero_result[dst_offset] = b_zero_reg;
    }
  }
}

}  // namespace allspark
