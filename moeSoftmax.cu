#include <type_traits>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <cfloat>
#include <math.h>

#ifndef USE_ROCM
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#else
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;
#endif

struct CubMaxOp {
    template <typename T>
    __device__ __forceinline__ T operator()(const T& a, const T& b) const {
        return a > b ? a : b;
    }
};

struct CubAddOp {
    template <typename T>
    __device__ __forceinline__ T operator()(const T& a, const T& b) const {
        return a + b;
    }
};

template <typename T>
__device__ __forceinline__ float toFloat(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(value);
    } else if constexpr (std::is_same_v<T, __half>) {
        return __half2float(value);
    }
}

template <int TPB, typename InputType>
__launch_bounds__(TPB) __global__
void moeSoftmax(const InputType* input, const bool* finished, float* output, const int num_cols)
{
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    __shared__ float normalizing_factor;
    __shared__ float float_max;

    const int thread_row_offset = blockIdx.x * num_cols;

    float threadData(-FLT_MAX);

    if ((finished != nullptr) && finished[blockIdx.x])
    {
        return;
    }

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        const float val = toFloat(input[idx]);
        threadData = max(val, threadData);
    }

    const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, CubMaxOp());
    if (threadIdx.x == 0)
    {
        float_max = maxElem;
    }
    __syncthreads();

    threadData = 0;

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        const float val = toFloat(input[idx]);
        threadData += expf(val - float_max);
    }

    const auto Z = BlockReduce(tmpStorage).Reduce(threadData, CubAddOp());

    if (threadIdx.x == 0)
    {
        normalizing_factor = 1.f / Z;
    }
    __syncthreads();

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        const float val = toFloat(input[idx]);
        const float softmax_val = expf(val - float_max) * normalizing_factor;
        output[idx] = softmax_val;
    }
}
