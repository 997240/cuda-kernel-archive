#include <cuda.h>

template <typename T>
__global__ void convert_data(const T* data1, const T* data2, double* fdata1,
                             double* fdata2, int size) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    fdata1[idx] = data1[idx];
    fdata2[idx] = data2[idx];
  }
}