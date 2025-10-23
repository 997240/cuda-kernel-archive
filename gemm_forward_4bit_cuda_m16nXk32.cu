#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
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

template <int N>
__global__ void __launch_bounds__(64)
    gemm_forward_4bit_cuda_m16nXk32(int G, int split_k_iters,
                                    half* __restrict__ A, int* __restrict__ B,
                                    half* __restrict__ scaling_factors,
                                    int* __restrict__ zeros, int M, int IC,
                                    int OC, half* __restrict__ C) {
  assert(N == 64 || N == 128);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
  assert(false);
#else
  static constexpr uint32_t ZERO = 0x0;
  float C_warp[32];
  __shared__ half A_shared[16 * (32 + 8)];
  __shared__ half B_shared[32 * (N + 8)];

  int j_factors1 = ((OC + N - 1) / N);
  int blockIdx_y = blockIdx.x % ((M + 16 - 1) / 16 * j_factors1);
  int blockIdx_z = blockIdx.x / ((M + 16 - 1) / 16 * j_factors1);

  half A_shared_warp[8];
  half B_shared_warp[N / 4];
  for (int j_0_4_init = 0; j_0_4_init < N / 32; ++j_0_4_init) {
    for (int i = 0; i < 8; ++i) {
      C_warp[(j_0_4_init * 8) + i] = 0.0;
    }
  }

  static constexpr int row_stride_warp = 32 * 8 / 32;
  static constexpr int row_stride = 2 * 32 * 8 / N;
  bool ld_A_flag =
      (blockIdx_y / j_factors1 * 16 + threadIdx.y * row_stride_warp +
       threadIdx.x * 8 / 32) < M;

  half* A_ptr =
      A +
      (((int)blockIdx_y) / j_factors1 * 16 +
       (((int)threadIdx.y) * row_stride_warp) + ((int)threadIdx.x) / (32 / 8)) *
          IC +
      (((int)threadIdx.x) % (32 / 8)) * 8;

  int* B_ptr = B + ((int)threadIdx.y) * (OC / 8) * (256 / N) +
               (((int)threadIdx.x) / (N / 8)) * (OC / 8) +
               (((int)blockIdx_y) % j_factors1) * (N / 8) +
               (((int)threadIdx.x) % (N / 8)) * 1;

  half* A_shared_ptr = A_shared +
                       ((int)threadIdx.y) * row_stride_warp * (32 + 8) +
                       (((int)threadIdx.x) / (32 / 8)) * (32 + 8) +
                       (((int)threadIdx.x) % (32 / 8)) * 8;

  half* B_shared_ptr = B_shared +
                       ((int)threadIdx.y) * (row_stride / 2) * (N + 8) +
                       (((int)threadIdx.x) / (N / 8)) * (N + 8) +
                       (((int)threadIdx.x) % (N / 8)) * 8;

  int* zeros_ptr = zeros + (((int)blockIdx_y) % j_factors1) * (N / 8) +
                   ((int)threadIdx.x) % (N / 8);

  half* scaling_factors_ptr = scaling_factors +
                              (((int)blockIdx_y) % j_factors1) * N +
                              (((int)threadIdx.x) % (N / 8)) * 8;

  half* C_ptr =
      C +
      static_cast<long long>(blockIdx_z) * M * OC +
      (((int)blockIdx_y) % j_factors1) * N + ((int)threadIdx.y) * (N / 2) +
      (((int)threadIdx.x) % 4) * 2;

  int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;
  if ((k_bound - 1) * split_k_iters * 32 + blockIdx_z * 32 >= IC) k_bound -= 1;
  for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
    int k_0_0 = _k_0_0 * split_k_iters + blockIdx_z;
    __syncthreads();
    if (ld_A_flag) {
      *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + (k_0_0 * 32));
    } else {
      *(uint4*)(A_shared_ptr) = make_uint4(0, 0, 0, 0);
    }

    uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr + k_0_0 * 32 / G * (OC / 8));
    uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);
    uint4 B_loaded_scale =
        *(uint4*)(scaling_factors_ptr + k_0_0 * 32 / G * (OC));
    int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);

    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < N / 16; ++ax0_ax1_fused_0) {
      uint32_t B_loaded =
          *(uint32_t*)(B_ptr_local + ax0_ax1_fused_0 * row_stride * (OC / 8));
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

      *(uint4*)(B_shared_ptr + ax0_ax1_fused_0 * row_stride * (N + 8)) =
          B_loaded_fp16;
    }
    __syncthreads();

    for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
      {
        unsigned int addr;
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
            : "=r"(addr)
            : "l"((void*)((&(A_shared[(k_0_1 * 16)])) +
                          (((((int)threadIdx.x) & 15) * 40) +
                           ((((int)threadIdx.x) >> 4) * 8)))));

        __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(((unsigned*)(A_shared_warp + 0))[0]),
              "=r"(((unsigned*)(A_shared_warp + 0))[1]),
              "=r"(((unsigned*)(A_shared_warp + 0))[2]),
              "=r"(((unsigned*)(A_shared_warp + 0))[3])
            : "r"(addr));
      }

      for (int ax1_0 = 0; ax1_0 < N / 32; ++ax1_0) {
        {
          unsigned int addr;
          __asm__ __volatile__(
              "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
              : "=r"(addr)
              : "l"((void*)((&(B_shared[(((k_0_1 * (N * 16 + 128)) +
                                          (((int)threadIdx.y) * (N / 2))) +
                                         (ax1_0 * 16))])) +
                            (((((int)threadIdx.x) & 15) * (N + 8)) +
                             ((((int)threadIdx.x) >> 4) * 8)))));
          __asm__ __volatile__(
              "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
              : "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[0]),
                "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[1]),
                "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[2]),
                "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[3])
              : "r"(addr));
        }
      }
      for (int j_0_4 = 0; j_0_4 < N / 32; ++j_0_4) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              : "=f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[0]),
                "r"(((unsigned*)(A_shared_warp + 0))[1]),
                "r"(((unsigned*)(B_shared_warp + (j_0_4 * 8)))[0]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[3]));
        }

        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              : "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[0]),
                "r"(((unsigned*)(A_shared_warp + 0))[1]),
                "r"(((unsigned*)(B_shared_warp + ((j_0_4 * 8) + 4)))[0]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3]));
        }

        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              : "=f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[2]),
                "r"(((unsigned*)(A_shared_warp + 0))[3]),
                "r"(((unsigned*)(B_shared_warp + (j_0_4 * 8)))[1]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[3]));
        }

        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              : "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[2]),
                "r"(((unsigned*)(A_shared_warp + 0))[3]),
                "r"(((unsigned*)(B_shared_warp + ((j_0_4 * 8) + 4)))[1]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3]));
        }
#else
        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
              : "=f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[0]),
                "r"(((unsigned*)(A_shared_warp + 0))[1]),
                "r"(((unsigned*)(A_shared_warp + 0))[2]),
                "r"(((unsigned*)(A_shared_warp + 0))[3]),
                "r"(((unsigned*)(B_shared_warp + (j_0_4 * 8)))[0]),
                "r"(((unsigned*)(B_shared_warp + (j_0_4 * 8)))[1]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[3]));
        }

        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
              : "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[0]),
                "r"(((unsigned*)(A_shared_warp + 0))[1]),
                "r"(((unsigned*)(A_shared_warp + 0))[2]),
                "r"(((unsigned*)(A_shared_warp + 0))[3]),
                "r"(((unsigned*)(B_shared_warp + ((j_0_4 * 8) + 4)))[0]),
                "r"(((unsigned*)(B_shared_warp + ((j_0_4 * 8) + 4)))[1]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3]));
        }

#endif
      }
    }
  }

  for (int ax1_0_1 = 0; ax1_0_1 < (N / 32); ++ax1_0_1) {
    for (int local_id = 0; local_id < 8; ++local_id) {
      int row_offset = (((int)blockIdx_y) / j_factors1) * 16 +
                       ((int)threadIdx.x) / 4 + (local_id % 4) / 2 * 8;
      if (row_offset < M) {
        *(C_ptr + ax1_0_1 * 16 + row_offset * OC + (local_id / 4) * 8 +
          local_id % 2) = __float2half(C_warp[(ax1_0_1 * 8) + local_id]);
      }
    }
  }
#endif
}

}  // namespace awq
}  // namespace vllm
