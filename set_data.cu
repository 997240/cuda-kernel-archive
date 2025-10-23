#include <cuda.h>

template <typename T>
__global__ void set_data(T* data, int size, int myRank) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    data[idx] = myRank * 0.11f;
  }
}