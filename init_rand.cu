#include <cuda.h>
#include <curand_kernel.h>

__global__ void init_rand(curandState_t* state, int size, int nRanks) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    for (int i = 0; i < nRanks; i++) {
      curand_init(i + 1, idx, 0, &state[idx * nRanks + i]);
    }
  }
}