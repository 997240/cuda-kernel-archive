#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <limits>
#include <cuda_runtime.h>

namespace vllm::cutlass_w4a8 {

__constant__ uint8_t kNibbleLUT[256];

__global__ void unified_encode_int4b_device(const uint8_t* in, uint8_t* out, size_t nbytes) {
  constexpr size_t V = sizeof(uint4);
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t nthreads = size_t(gridDim.x) * blockDim.x;
  const size_t nvec = nbytes / V;

  for (size_t vec = tid; vec < nvec; vec += nthreads) {
    uint4 v = reinterpret_cast<const uint4*>(in)[vec];
    uint8_t* b = reinterpret_cast<uint8_t*>(&v);
#pragma unroll
    for (int i = 0; i < int(V); ++i) b[i] = kNibbleLUT[b[i]];
    reinterpret_cast<uint4*>(out)[vec] = v;
  }
}

}  // namespace vllm::cutlass_w4a8
