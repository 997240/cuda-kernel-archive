#include <type_traits>
#include <cub/cub.cuh>
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

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#ifdef USE_ROCM
#define VLLM_SHFL_XOR_SYNC_WIDTH(val, mask, width) __shfl_xor(val, mask, width)
#else
#define VLLM_SHFL_XOR_SYNC_WIDTH(val, mask, width) __shfl_xor_sync(0xffffffff, val, mask, width)
#endif

template <typename T, int N, int Alignment = sizeof(T) * N>
struct alignas(Alignment) AlignedArray {
    T data[N];
};

template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG, int WARP_SIZE_PARAM, typename IndType, typename InputType>
__launch_bounds__(WARPS_PER_CTA * WARP_SIZE_PARAM) __global__
void topkGatingSoftmax(const InputType* input, const bool* finished, float* output, const int num_rows, IndType* indices,
        int* source_rows, const int k, const int start_expert, const int end_expert, const bool renormalize)
{
    static_assert(std::is_same_v<InputType, float> || std::is_same_v<InputType, __nv_bfloat16> ||
                      std::is_same_v<InputType, __half>,
                  "InputType must be float, __nv_bfloat16, or __half");

    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    if constexpr (std::is_same_v<InputType, __nv_bfloat16> || std::is_same_v<InputType, __half>) {
        static_assert(ELTS_PER_LDG == 1 || ELTS_PER_LDG % 2 == 0,
            "ELTS_PER_LDG must be 1 or even for 16-bit conversion");
    }

    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE_PARAM % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE_PARAM, "THREADS_PER_ROW can be at most warp size");

    static constexpr int ELTS_PER_WARP = WARP_SIZE_PARAM * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;

    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;

    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;

    if (thread_row >= num_rows)
    {
        return;
    }
    const bool row_is_active = finished ? !finished[thread_row] : true;

    const InputType* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const InputType* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    float row_chunk[VPT];

    if constexpr (std::is_same_v<InputType, float>) {
        using VecType = AlignedArray<float, ELTS_PER_LDG>;
        VecType* row_chunk_vec_ptr = reinterpret_cast<VecType*>(&row_chunk);
        const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
        for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
            row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
        }
    } else if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
        if constexpr (ELTS_PER_LDG >= 2) {
            using VecType = AlignedArray<__nv_bfloat16, ELTS_PER_LDG>;
            float2* row_chunk_f2 = reinterpret_cast<float2*>(row_chunk);
            const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                VecType vec = vec_thread_read_ptr[ii * THREADS_PER_ROW];
                int base_idx_f2 = ii * ELTS_PER_LDG / 2;
#pragma unroll
                for (int jj = 0; jj < ELTS_PER_LDG / 2; ++jj) {
                    row_chunk_f2[base_idx_f2 + jj] = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(vec.data + jj * 2)
                    );
                }
            }
        } else {
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                const __nv_bfloat16* scalar_ptr = thread_read_ptr + ii * THREADS_PER_ROW;
                row_chunk[ii] = __bfloat162float(*scalar_ptr);
            }
        }
    } else if constexpr (std::is_same_v<InputType, __half>) {
        if constexpr (ELTS_PER_LDG >= 2) {
            using VecType = AlignedArray<__half, ELTS_PER_LDG>;
            float2* row_chunk_f2 = reinterpret_cast<float2*>(row_chunk);
            const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                VecType vec = vec_thread_read_ptr[ii * THREADS_PER_ROW];
                int base_idx_f2 = ii * ELTS_PER_LDG / 2;
#pragma unroll
                for (int jj = 0; jj < ELTS_PER_LDG / 2; ++jj) {
                    row_chunk_f2[base_idx_f2 + jj] = __half22float2(
                        *reinterpret_cast<const __half2*>(vec.data + jj * 2)
                    );
                }
            }
        } else {
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                const __half* scalar_ptr = thread_read_ptr + ii * THREADS_PER_ROW;
                row_chunk[ii] = __half2float(*scalar_ptr);
            }
        }
    }

    float thread_max = row_chunk[0];
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii)
    {
        thread_max = max(thread_max, row_chunk[ii]);
    }
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        thread_max = max(thread_max, VLLM_SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW));
    }

    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = expf(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
    }
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        row_sum += VLLM_SHFL_XOR_SYNC_WIDTH(row_sum, mask, THREADS_PER_ROW);
    }

    const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
    }

    int start_col = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    float selected_sum = 0.f;
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        float max_val = row_chunk[0];
        int expert = start_col;
#pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG)
        {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii)
            {
                float val = row_chunk[ldg * ELTS_PER_LDG + ii];

                if (val > max_val)
                {
                    max_val = val;
                    expert = col + ii;
                }
            }
        }
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
        {
            float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(max_val, mask, THREADS_PER_ROW);
            int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(expert, mask, THREADS_PER_ROW);

            if (other_max > max_val || (other_max == max_val && other_expert < expert))
            {
                max_val = other_max;
                expert = other_expert;
            }
        }

        if (thread_group_idx == 0)
        {
            const bool node_uses_expert = expert >= start_expert && expert < end_expert;
            const bool should_process_row = row_is_active && node_uses_expert;

            const int idx = k * thread_row + k_idx;
            output[idx] = max_val;
            indices[idx] = should_process_row ? (expert - start_expert) : NUM_EXPERTS;
            source_rows[idx] = k_idx * num_rows + thread_row;
            if (renormalize) {
                selected_sum += max_val;
            }
        }

        if (k_idx + 1 < k)
        {
            const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
            const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

            if (thread_group_idx == thread_to_clear_in_group)
            {
                const int offset_for_expert = expert % ELTS_PER_LDG;
                row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.f;
            }
        }
    }

    if (renormalize) {
        if (thread_group_idx == 0)
        {
            const float denom = selected_sum > 0.f ? selected_sum : 1.f;
            for (int k_idx = 0; k_idx < k; ++k_idx)
            {
                const int idx = k * thread_row + k_idx;
                output[idx] = output[idx] / denom;
            }
        }
    }
}
