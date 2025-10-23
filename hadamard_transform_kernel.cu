#include <stdint.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda/annotated_ptr>
namespace hadacore {

#ifndef __CUDACC__
#define __launch_bounds__(x,y)
#endif

#define MAX_WARPS_PER_SM 48
#define MIN(a, b) ((a) < (b) ? (a) : (b))

using b16 = uint16_t;
using b32 = uint32_t;

template <torch::ScalarType dtype>
__device__ __forceinline__ void mma_m16_n8_k16_b16_b16_b16_noacc(b32 a0, b32 a1, b32 a2, b32 a3, b32 b0, b32 b1, b32& c0, b32& c1){
    static_assert(dtype == torch::ScalarType::Half || dtype == torch::ScalarType::BFloat16);
    b32 zero = 0;
    if constexpr(dtype == torch::ScalarType::Half) {
        asm (
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
            "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n\t"
            : "=r"(c0), "=r"(c1) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(zero), "r"(zero)
        );
    } else {
        b32 temp0, temp1, temp2, temp3;
        asm (
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n\t"
            : "=r"(temp0), "=r"(temp1), "=r"(temp2), "=r"(temp3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(zero), "r"(zero), "r"(zero), "r"(zero)
        );
        asm ("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t" : "=r"(c0) : "r"(temp1), "r"(temp0));
        asm ("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t" : "=r"(c1) : "r"(temp3), "r"(temp2));
    }
}

template <torch::ScalarType dtype>
__device__ __forceinline__ void mma_m16_n16_k16_b16_b16_b16_noacc(b32 a0, b32 a1, b32 a2, b32 a3, b32 b0, b32 b1, b32 b2, b32 b3, b32& c0, b32& c1, b32& c2, b32& c3){
    mma_m16_n8_k16_b16_b16_b16_noacc<dtype>(a0, a1, a2, a3, b0, b1, c0, c1);
    mma_m16_n8_k16_b16_b16_b16_noacc<dtype>(a0, a1, a2, a3, b2, b3, c2, c3);
}

__device__ __forceinline__ void matrix_transpose_m8_n8_b16_inplace(b32& a0) {
    asm (
        "movmatrix.sync.aligned.m8n8.trans.b16 "
        "%0, %1;\n\t"
        : "=r"(a0) : "r"(a0)
    );
}

#define p_p(i) ((val_1p[i] & 0x0000FFFF) | val_1p[i] << 16)
#define p_n(i) ((val_1p[i] & 0x0000FFFF) | val_1n[i] << 16)
#define n_p(i) ((val_1n[i] & 0x0000FFFF) | val_1p[i] << 16)
#define n_n(i) ((val_1n[i] & 0x0000FFFF) | val_1n[i] << 16)

template<int64_t num_chunks, int64_t warps_per_block, int64_t log_had_size, int64_t blocks_per_sm, bool enable_mask, torch::ScalarType dtype>
__global__ void __launch_bounds__(32 * warps_per_block, blocks_per_sm)
hadamard_transform_kernel(b16* a, b16* out, int total_num_chunks) {
    static_assert(dtype == torch::ScalarType::Half || dtype == torch::ScalarType::BFloat16, "Only fp16 and bf16 supported currently");

    b32 b_frag_all[num_chunks][4];

    int64_t blockid = blockIdx.x * warps_per_block + threadIdx.x / 32;
    int64_t threadid = threadIdx.x % 32;
    extern __shared__ b32 bfrag_arr[];
    int64_t real_num_chunks = ((blockid + 1) * num_chunks) > total_num_chunks ? (total_num_chunks - (blockid * num_chunks)) : num_chunks;
    int64_t diff_num_chunks = real_num_chunks - num_chunks;

    b32* a_start_ptr = (b32*) (a + blockid * num_chunks * 256);
    b32* out_start_ptr = (b32*) (out + blockid * num_chunks * 256);
    b32* a_ptr = a_start_ptr + threadid * 4;
    b32* b_frag_ptr = bfrag_arr + (blockid % warps_per_block) * num_chunks * 128 + threadid * 4;

    #if (__CUDA_ARCH__ < 900)
    uint64_t cache_policy;
    asm volatile(
        "createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n"
        : "=l"(cache_policy)
    );
    #endif

    #pragma unroll
    for (int64_t k = 0; k < num_chunks; k++) {
        size_t shared_ptr = __cvta_generic_to_shared(b_frag_ptr);
        #if (__CUDA_ARCH__ >= 900)
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\n"
                "cp.async.commit_group;\n"
                :: "l"(shared_ptr), "l"(a_ptr)
            );
        #else
            asm volatile(
                "cp.async.cg.shared.global.L2::cache_hint.L2::256B [%0], [%1], 16, %2;\n"
                "cp.async.commit_group;\n"
                :: "l"(shared_ptr), "l"(a_ptr), "l"(cache_policy)
            );
        #endif

        a_ptr += 128;
        b_frag_ptr += 128;
    }

    constexpr b16 fp16_1p[4] = {0b0011100110101000, 0b0011100000000000, 0b0011010110101000, 0b0011010000000000};
    constexpr b16 fp16_1n[4] = {0b1011100110101000, 0b1011100000000000, 0b1011010110101000, 0b1011010000000000};
    constexpr b16 bf16_1p[4] = {0b0011111100110101, 0b0011111100000000, 0b0011111010110101, 0b0011111010000000};
    constexpr b16 bf16_1n[4] = {0b1011111100110101, 0b1011111100000000, 0b1011111010110101, 0b1011111010000000};

    #define val_type_1p(i) (((dtype) == torch::ScalarType::Half) ? (fp16_1p[i]) : (bf16_1p[i]))
    #define val_type_1n(i) (((dtype) == torch::ScalarType::Half) ? (fp16_1n[i]) : (bf16_1n[i]))
    constexpr b16 val_1p[4] = {val_type_1p(0), val_type_1p(1), val_type_1p(2), val_type_1p(3)};
    constexpr b16 val_1n[4] = {val_type_1n(0), val_type_1n(1), val_type_1n(2), val_type_1n(3)};

    constexpr b32 p_p[4] = {p_p(0), p_p(1), p_p(2), p_p(3)};
    constexpr b32 p_n[4] = {p_n(0), p_n(1), p_n(2), p_n(3)};
    constexpr b32 n_p[4] = {n_p(0), n_p(1), n_p(2), n_p(3)};
    constexpr b32 n_n[4] = {n_n(0), n_n(1), n_n(2), n_n(3)};
    const b32 had_16_p1[4][4] = {
        {
            0b10001000010001000010001000010001,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b10001000010001000010001000010001
        },
        {
            0b11001100100010000011001100100010,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b11001100100010000011001100100010
        },
        {
            0b11111111101010101100110010011001,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b11111111101010101100110010011001
        },
        {
            0b11111111101010101100110010011001,
            0b11111111101010101100110010011001,
            0b11111111101010101100110010011001,
            0b00000000010101010011001101100110
        }
    };
    const b32 had_16_p2[4][4] = {
        {
            0b10000000010000000010000000010000,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b10000000010000000010000000010000
        },
        {
            0b11000000100001000011000000100001,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b11000000100001000011000000100001
        },
        {
            0b11110000101001011100001110010110,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b11110000101001011100001110010110
        },
        {
            0b11110000101001011100001110010110,
            0b11110000101001011100001110010110,
            0b11110000101001011100001110010110,
            0b00001111010110100011110001101001
        }
    };
    const b32 had_16_mask[3][4] = {
        {
            0b10001000010001000010001000010001,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b10001000010001000010001000010001
        },
        {
            0b11001100110011000011001100110011,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b11001100110011000011001100110011
        },
        {
            0b11111111111111111111111111111111,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b11111111111111111111111111111111
        }
    };
    b32 had_frag[8];
    #pragma unroll
    for (int64_t i = 0; i < 2; i++) {
        int64_t c_log_h = (i == 0) ? MIN(4, log_had_size) : log_had_size % 4;
        #pragma unroll
        for (int64_t j = 0; j < 4; j++) {
            if (c_log_h < 4) {
                bool mask = had_16_mask[c_log_h - 1][j] & (1 << (31 - threadid));
                if (!mask) {
                    had_frag[i * 4 + j] = 0;
                    continue;
                }
            }
            bool pred1 = had_16_p1[c_log_h - 1][j] & (1 << (31 - threadid));
            bool pred2 = had_16_p2[c_log_h - 1][j] & (1 << (31 - threadid));
            b32 val = pred1 ? (pred2 ? p_p[c_log_h - 1] : p_n[c_log_h - 1]) : (pred2 ? n_p[c_log_h - 1] : n_n[c_log_h - 1]);
            had_frag[i * 4 + j] = val;
        }
        if constexpr(log_had_size <= 4 || log_had_size % 4 == 0) break;
    }

    constexpr int64_t part8_log_had_size = log_had_size - 8;

    b32* a_chunk_ptr = a_start_ptr;
    b32* out_chunk_ptr = out_start_ptr;

    #pragma unroll
    for (int64_t l = 0; l < 2; l++) {
        if constexpr(log_had_size <= 8) {
            b_frag_ptr = bfrag_arr + (blockid % warps_per_block) * num_chunks * 128;
        } else {
            b_frag_ptr = bfrag_arr + (blockid % warps_per_block) * num_chunks * (l == 0 ? 128 : (128 >> part8_log_had_size));
        }

        if (l == 1) {
            if constexpr(log_had_size > 8) {
                __syncthreads();

                if constexpr(log_had_size >= 12) {
                    b32* store = bfrag_arr + (128 >> part8_log_had_size) * (num_chunks * (blockid % warps_per_block));

                    #pragma unroll
                    for (int64_t j = 0; j < 4; j++) {
                        #pragma unroll
                        for (int64_t k = 0; k < num_chunks; k++) {
                            uint64_t real_chunk_num = (num_chunks - (threadid % num_chunks) + k) % num_chunks;

                            int64_t real_thread_id = (threadid / num_chunks) * num_chunks + k;
                            int64_t chunk_idx = 128 * real_chunk_num;
                            int64_t thread_group_idx = (real_thread_id / 4) * 16;
                            int64_t thread_idx = (real_thread_id % 4) * 2;
                            int64_t reg_idx = (j / 2) * 8 + (j % 2);
                            int64_t idx = chunk_idx + thread_group_idx + thread_idx + reg_idx;

                            int64_t rowidx = idx % (1 << part8_log_had_size);
                            int64_t colidx = idx >> part8_log_had_size;

                            b32 data = store[rowidx * 128 + colidx];

                            #pragma unroll
                            for (uint64_t i = 0; i < num_chunks; i++) {
                                asm volatile (
                                    "{\n\t"
                                    "  .reg .pred p0;\n\t"
                                    "  setp.eq.s64 p0, %1, %2;\n\t"
                                    "  @p0 mov.b32 %0, %3;\n\t"
                                    "}\n\t"
                                    : "+r"(b_frag_all[i][j])
                                    : "l"(real_chunk_num), "l"(i), "r"(data)
                                );
                            }
                        }
                    }

                    #pragma unroll
                    for (int64_t j = 0; j < 4; j++) {
                        #pragma unroll
                        for (int64_t k = 1; k < num_chunks; k++) {
                            int64_t threadid_contig = threadid % num_chunks;
                            int64_t threadid_mul = threadid / num_chunks;
                            int64_t threadid2 = (threadid_contig + num_chunks - k) % num_chunks + threadid_mul * num_chunks;
                            b_frag_all[k][j] = __shfl_sync(0xFFFFFFFF, b_frag_all[k][j], threadid2);
                        }
                    }
                }
            }
        }

        #pragma unroll
        for (int64_t k = 0; k < num_chunks; k++) {
            if constexpr(enable_mask) {
                if (k >= real_num_chunks)
                    break;
            }
            if (l == 0) {
                #define SWITCH_WAIT_ASYNC_LOAD_GROUP(i) case i: asm volatile("cp.async.wait_group %0;\n" :: "n"(num_chunks - i - 1)); break;
                if constexpr(enable_mask) {
                    switch(k + diff_num_chunks) {
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(0)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(1)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(2)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(3)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(4)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(5)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(6)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(7)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(8)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(9)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(10)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(11)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(12)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(13)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(14)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(15)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(16)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(17)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(18)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(19)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(20)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(21)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(22)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(23)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(24)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(25)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(26)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(27)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(28)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(29)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(30)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(31)
                    }
                } else {
                    switch(k) {
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(0)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(1)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(2)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(3)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(4)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(5)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(6)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(7)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(8)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(9)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(10)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(11)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(12)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(13)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(14)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(15)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(16)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(17)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(18)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(19)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(20)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(21)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(22)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(23)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(24)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(25)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(26)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(27)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(28)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(29)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(30)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(31)
                    }
                }
            }

            if (l == 0) {
                #pragma unroll
                for (int64_t j = 0; j < 4; j++) {
                    int64_t reg = ((threadid & 16) == 0) ? j : (j / 2 * 2 + (1 - j % 2));
                    int64_t real_thread_id = (reg == 0 || reg == 2) ? threadid : (threadid ^ 16);
                    int64_t real_row = real_thread_id % 4;
                    int64_t real_col = real_thread_id / 4;
                    b_frag_all[k][j] = b_frag_ptr[(real_row + (reg % 2) * 4) + (real_col + (j / 2) * 8) * 8];
                }

                if ((threadid & 16) != 0) {
                    b32 temp = b_frag_all[k][0];
                    b_frag_all[k][0] = b_frag_all[k][1];
                    b_frag_all[k][1] = temp;

                    temp = b_frag_all[k][2];
                    b_frag_all[k][2] = b_frag_all[k][3];
                    b_frag_all[k][3] = temp;
                }

                #pragma unroll
                for (int64_t j = 1; j < 4; j += 2) {
                    b_frag_all[k][j] = __shfl_xor_sync(0xFFFFFFFF, b_frag_all[k][j], 16);
                }
            } else if constexpr(log_had_size > 8) {
                if constexpr(log_had_size < 12) {
                    constexpr int64_t xor_val = log_had_size == 9 ? 16 : 1;

                    #pragma unroll
                    for (int64_t j = 0; j < 4; j++) {
                        int64_t reg = ((threadid & xor_val) == 0) ? j : (j + 2) % 4;
                        int64_t real_thread_id = reg < 2 ? threadid : (threadid ^ xor_val);
                        int64_t idx = (real_thread_id / 4 * 16) + (real_thread_id % 4 * 2) + (reg / 2 * 8) + (reg % 2);
                        int64_t rowidx = idx % (1 << part8_log_had_size);
                        int64_t colidx = idx >> part8_log_had_size;
                        b_frag_all[k][j] = b_frag_ptr[rowidx * 128 + colidx];
                    }

                    if ((threadid & xor_val) != 0) {
                        b32 temp = b_frag_all[k][0];
                        b_frag_all[k][0] = b_frag_all[k][2];
                        b_frag_all[k][2] = temp;

                        temp = b_frag_all[k][1];
                        b_frag_all[k][1] = b_frag_all[k][3];
                        b_frag_all[k][3] = temp;
                    }

                    #pragma unroll
                    for (int64_t j = 2; j < 4; j++) {
                        b_frag_all[k][j] = __shfl_xor_sync(0xFFFFFFFF, b_frag_all[k][j], xor_val);
                    }
                }
            }

            if (l == 1) {
                b32 f0 = ((b_frag_all[k][1] & 0xFFFF) << 16) | (b_frag_all[k][0] & 0xFFFF);
                b32 f1 = ((b_frag_all[k][3] & 0xFFFF) << 16) | (b_frag_all[k][2] & 0xFFFF);
                b32 f2 = (b_frag_all[k][1] & 0xFFFF0000) | (b_frag_all[k][0] >> 16);
                b32 f3 = (b_frag_all[k][3] & 0xFFFF0000) | (b_frag_all[k][2] >> 16);
                b_frag_all[k][0] = f0;
                b_frag_all[k][1] = f1;
                b_frag_all[k][2] = f2;
                b_frag_all[k][3] = f3;
            }

            #pragma unroll
            for(int64_t i = 0, remaining_log_had_size = log_had_size - l * 8; i < 2 && remaining_log_had_size > 0; i++) {
                int64_t had_off = ((remaining_log_had_size < 4) && !(log_had_size <= 4 || log_had_size % 4 == 0)) ? 4 : 0;
                mma_m16_n16_k16_b16_b16_b16_noacc<dtype>(had_frag[had_off + 0], had_frag[had_off + 1], had_frag[had_off + 2], had_frag[had_off + 3], b_frag_all[k][0], b_frag_all[k][1], b_frag_all[k][2], b_frag_all[k][3], b_frag_all[k][0], b_frag_all[k][1], b_frag_all[k][2], b_frag_all[k][3]);

                remaining_log_had_size -= 4;
                if (remaining_log_had_size <= 0 && i == 0) {
                    matrix_transpose_m8_n8_b16_inplace(b_frag_all[k][0]);
                    matrix_transpose_m8_n8_b16_inplace(b_frag_all[k][1]);
                    matrix_transpose_m8_n8_b16_inplace(b_frag_all[k][2]);
                    matrix_transpose_m8_n8_b16_inplace(b_frag_all[k][3]);
                } else {
                    b32 temp = b_frag_all[k][1];
                    b_frag_all[k][1] = b_frag_all[k][2];
                    b_frag_all[k][2] = temp;
                }
            }

            if (l == 1) {
                b32 f0 = ((b_frag_all[k][2] & 0xFFFF) << 16) | (b_frag_all[k][0] & 0xFFFF);
                b32 f1 = (b_frag_all[k][2] & 0xFFFF0000) | (b_frag_all[k][0] >> 16);
                b32 f2 = ((b_frag_all[k][3] & 0xFFFF) << 16) | (b_frag_all[k][1] & 0xFFFF);
                b32 f3 = (b_frag_all[k][3] & 0xFFFF0000) | (b_frag_all[k][1] >> 16);
                b_frag_all[k][0] = f0;
                b_frag_all[k][1] = f1;
                b_frag_all[k][2] = f2;
                b_frag_all[k][3] = f3;
            }

            if (l == 0) {
                #pragma unroll
                for (int64_t j = 1; j < 4; j += 2) {
                    b_frag_all[k][j] = __shfl_xor_sync(0xFFFFFFFF, b_frag_all[k][j], 16);
                }

                if ((threadid & 16) != 0) {
                    b32 temp = b_frag_all[k][0];
                    b_frag_all[k][0] = b_frag_all[k][1];
                    b_frag_all[k][1] = temp;

                    temp = b_frag_all[k][2];
                    b_frag_all[k][2] = b_frag_all[k][3];
                    b_frag_all[k][3] = temp;
                }

                b32* store = (log_had_size <= 8) ? out_chunk_ptr : b_frag_ptr;

                #pragma unroll
                for (int64_t j = 0; j < 4; j++) {
                    int64_t reg = ((threadid & 16) == 0) ? j : (j / 2 * 2 + (1 - j % 2));
                    int64_t real_thread_id = (reg == 0 || reg == 2) ? threadid : (threadid ^ 16);
                    int64_t real_row = real_thread_id % 4;
                    int64_t real_col = real_thread_id / 4;
                    store[(real_row + (reg % 2) * 4) + (real_col + (reg / 2) * 8) * 8] = b_frag_all[k][j];
                }
            } else if constexpr(log_had_size > 8) {
                if (log_had_size < 12) {
                    constexpr int xor_val = log_had_size == 9 ? 16 : 1;
                    #pragma unroll
                    for (int64_t j = 2; j < 4; j++) {
                        b_frag_all[k][j] = __shfl_xor_sync(0xFFFFFFFF, b_frag_all[k][j], xor_val);
                    }

                    if ((threadid & xor_val) != 0) {
                        b32 temp = b_frag_all[k][0];
                        b_frag_all[k][0] = b_frag_all[k][2];
                        b_frag_all[k][2] = temp;

                        temp = b_frag_all[k][1];
                        b_frag_all[k][1] = b_frag_all[k][3];
                        b_frag_all[k][3] = temp;
                    }

                    b32* store = (b32*)(out + (blockid / warps_per_block) * (num_chunks * warps_per_block) * 256 + (256 >> part8_log_had_size) * (num_chunks * (blockid % warps_per_block) + k));
                    #pragma unroll
                    for (int64_t j = 0; j < 4; j++) {
                        int64_t reg = ((threadid & xor_val) == 0) ? j : (j + 2) % 4;
                        b32 data = b_frag_all[k][j];
                        int64_t real_thread_id = reg < 2 ? threadid : (threadid ^ xor_val);
                        int64_t idx = (real_thread_id / 4 * 16) + (real_thread_id % 4 * 2) + (reg / 2 * 8) + (reg % 2);
                        int64_t rowidx = idx % (1 << part8_log_had_size);
                        int64_t colidx = idx >> part8_log_had_size;
                        store[rowidx * 128 + colidx] = data;
                    }
                }
            }

            a_chunk_ptr += 128;
            out_chunk_ptr += 128;
            if constexpr(log_had_size > 8) {
                b_frag_ptr += (l == 0 ? 128 : (128 >> part8_log_had_size));
            } else {
                b_frag_ptr += 128;
            }
        }
        if (log_had_size <= 8)
            break;
    }

    if constexpr(log_had_size >= 12) {
        #pragma unroll
        for (int64_t j = 0; j < 4; j++) {
            #pragma unroll
            for (int64_t k = 1; k < num_chunks; k++) {
                int64_t threadid_contig = threadid % num_chunks;
                int64_t threadid_mul = threadid / num_chunks;
                int64_t threadid2 = (threadid_contig + k) % num_chunks + threadid_mul * num_chunks;
                b_frag_all[k][j] = __shfl_sync(0xFFFFFFFF, b_frag_all[k][j], threadid2);
            }
        }

        b32* store = bfrag_arr + (128 >> part8_log_had_size) * (num_chunks * (blockid % warps_per_block));

        #pragma unroll
        for (int64_t j = 0; j < 4; j++) {
            #pragma unroll
            for (int64_t k = 0; k < num_chunks; k++) {
                int64_t real_chunk_num = (num_chunks - (threadid % num_chunks) + k) % num_chunks;

                b32 data;
                #pragma unroll
                for (int64_t i = 0; i < num_chunks; i++) {
                    if (real_chunk_num == i) data = b_frag_all[i][j];
                }
                
                int64_t real_thread_id = (threadid / num_chunks) * num_chunks + k;
                int64_t chunk_idx = 128 * real_chunk_num;
                int64_t thread_group_idx = (real_thread_id / 4) * 16;
                int64_t thread_idx = (real_thread_id % 4) * 2;
                int64_t reg_idx = (j / 2) * 8 + (j % 2);
                int64_t idx = chunk_idx + thread_group_idx + thread_idx + reg_idx;

                int64_t rowidx = idx % (1 << part8_log_had_size);
                int64_t colidx = idx >> part8_log_had_size;

                store[rowidx * 128 + colidx] = data;
            }
        }

        __syncthreads();
        store = ((b32*) out) + (blockid / warps_per_block) * (num_chunks * warps_per_block) * 128;
        int4* store4 = (int4*) store;
        int4* bfrag_arr4 = (int4*) bfrag_arr;
        #pragma unroll
        for (int64_t warp_off = 0; warp_off < (num_chunks * warps_per_block * 128 / 4); warp_off += 32 * warps_per_block) {
            int64_t total_off = warp_off + threadid + (blockid % warps_per_block) * 32;
            store4[total_off] = bfrag_arr4[total_off];
        }
    }

}

}  // namespace hadacore
