#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>

namespace allspark {

using marlin::ScalarType;

template <typename FType, int BLOCK, int N_MATRIX>
__global__ void f16_gemm_splitk_reduce_kernel(const FType* C_split, FType* C,
                                              uint32_t n, uint32_t n_matrix,
                                              uint32_t matrix_size) {
  auto idx = blockIdx.x * BLOCK + threadIdx.x;

  if (idx >= matrix_size) {
    return;
  }

  float sum = 0.f;

  int n_mat = N_MATRIX > 0 ? N_MATRIX : (int)n_matrix;
  for (int i = 0; i < n_mat; ++i) {
    sum += ScalarType<FType>::num2float(C_split[idx + i * matrix_size]);
  }

  C[idx] = ScalarType<FType>::float2num(sum);
}

}  // namespace allspark
