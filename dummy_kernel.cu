#include <cuda.h>

#ifdef USE_ROCM
__global__ void dummy_kernel() {
  for (int i = 0; i < 100; i++) {
    uint64_t start = wall_clock64();
    uint64_t cycles_elapsed;
    do {
      cycles_elapsed = wall_clock64() - start;
    } while (cycles_elapsed < 100);
  }
  for (int i = 0; i < 100; i++) __nanosleep(1000000);
}
#else
__global__ void dummy_kernel() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  for (int i = 0; i < 100; i++) __nanosleep(1000000);
#else
  for (int i = 0; i < 100; i++) {
    long long int start = clock64();
    while (clock64() - start < 150000000)
      ;
  }
#endif
}
#endif