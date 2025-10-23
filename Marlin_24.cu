#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

template <typename T>
inline std::string str(T x) {
  return std::to_string(x);
}

namespace marlin_24 {

static constexpr int THREADS = 256;
static constexpr int STAGES = 4;

static constexpr int min_thread_n = 128;

static constexpr int tile_size = 16;
static constexpr int max_par = 64;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

template <const int num_bits,
          const int threads,
          const int thread_m_blocks,
          const int thread_n_blocks,
          const int thread_k_blocks,
          const int stages,
          const int group_blocks = -1>
__global__ void Marlin_24(
    const int4* __restrict__ A,
    const int4* __restrict__ B,
    const int4* __restrict__ meta,
    int4* __restrict__ C,
    const int4* __restrict__ s,
    int prob_m,
    int prob_n,
    int prob_k,
    int* locks
) {}

#else

template <const int num_bits,
          const int threads,
          const int thread_m_blocks,
          const int thread_n_blocks,
          const int thread_k_blocks,
          const int stages,
          const int group_blocks = -1>
__global__ void Marlin_24(
    const int4* __restrict__ A,
    const int4* __restrict__ B,
    const int4* __restrict__ meta,
    int4* __restrict__ C,
    const int4* __restrict__ s,
    int prob_m,
    int prob_n,
    int prob_k,
    int* locks
) {
  int parallel = 1;
  if (prob_m > 16 * thread_m_blocks) {
    parallel = prob_m / (16 * thread_m_blocks);
    prob_m = 16 * thread_m_blocks;
  }

  int k_tiles = prob_k / 32 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  int iters = ceildiv(k_tiles * n_tiles * parallel, gridDim.x);

  if (group_blocks != -1)
    iters = (group_blocks / thread_k_blocks) *
            ceildiv(iters, (group_blocks / thread_k_blocks));

  int slice_row = (iters * blockIdx.x) % k_tiles;
  int slice_col_par = (iters * blockIdx.x) / k_tiles;
  int slice_col = slice_col_par;
  int slice_iters;
  int slice_count = 0;
  int slice_idx;

  if (slice_col_par >= n_tiles) {
    A += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_k / 8;
    C += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 8;
    locks += (slice_col_par / n_tiles) * n_tiles;
    slice_col = slice_col_par % n_tiles;
  }

  auto init_slice = [&]() {
    slice_iters =
        iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
    if (slice_iters < 0 || slice_col_par >= n_tiles * parallel) slice_iters = 0;
    if (slice_iters == 0) return;
    if (slice_row + slice_iters > k_tiles) slice_iters = k_tiles - slice_row;
    slice_count = 1;
    slice_idx = 0;
    int col_first = iters * ceildiv(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = ceildiv(k_tiles - col_off, iters);
      if (col_off > 0) slice_count++;
      int delta_first = iters * blockIdx.x - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;
      else {
        slice_idx = slice_count - 1 - delta_first / iters;
        if (col_off > 0) slice_idx--;
      }
    }
    if (slice_col == n_tiles) {
      A += 16 * thread_m_blocks * prob_k / 8;
      C += 16 * thread_m_blocks * prob_n / 8;
      locks += n_tiles;
      slice_col = 0;
    }
  };
  init_slice();

  int a_gl_stride = prob_k / 8;

  constexpr int a_sh_stride = 32 * thread_k_blocks / 8;
  constexpr int a_gl_rd_delta_o = 32 * thread_k_blocks / 8;
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);
  constexpr int a_sh_rd_delta_o = 4 * ((threads / 32) / (thread_n_blocks / 4));
  constexpr int a_sh_rd_delta_i = a_sh_stride * 16;
  constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks);
  constexpr int a_sh_wr_iters = ceildiv(a_sh_stage, a_sh_wr_delta);

  constexpr int pack_factor = 32 / num_bits;

  int b_gl_stride = 16 * prob_n / (pack_factor * 4);
  constexpr int b_sh_stride = ((thread_n_blocks * 16) * 16 / pack_factor) / 4;
  constexpr int b_thread_vecs = num_bits == 4 ? 1 : 2;
  constexpr int b_sh_stride_threads = b_sh_stride / b_thread_vecs;
  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
  int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride_threads);
  constexpr int b_sh_wr_delta = threads * b_thread_vecs;
  constexpr int b_sh_rd_delta = threads * b_thread_vecs;
  constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

  int m_gl_stride = 2 * prob_n / 8;
  constexpr int m_sh_stride =
      (16 * thread_n_blocks) / 4;
  int m_gl_rd_delta_o = m_gl_stride * thread_k_blocks;
  int m_gl_rd_delta_i = m_gl_stride * (threads / m_sh_stride);
  constexpr int m_sh_wr_delta = threads / 2;
  constexpr int m_sh_rd_delta = threads / 2;
  constexpr int m_sh_stage = m_sh_stride * thread_k_blocks;
  constexpr int m_sh_iters = ceildiv(m_sh_stage, m_sh_wr_delta);

  int s_gl_stride = prob_n / 8;
  constexpr int s_sh_stride = 16 * thread_n_blocks / 8;
  constexpr int s_sh_stage = s_sh_stride;
  int s_gl_rd_delta = s_gl_stride;

  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  a_gl_rd += a_gl_rd_delta_o * slice_row;
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  int a_sh_rd =
      a_sh_stride * ((threadIdx.x % 32) % 16) + (threadIdx.x % 32) / 16;
  a_sh_rd += 4 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

  int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride_threads) +
                (threadIdx.x % b_sh_stride_threads) * b_thread_vecs;
  b_gl_rd += b_sh_stride * slice_col;
  b_gl_rd += b_gl_rd_delta_o * slice_row;
  auto b_sh_wr = threadIdx.x * b_thread_vecs;
  auto b_sh_rd = threadIdx.x * b_thread_vecs;

  int m_gl_rd = m_gl_stride * (threadIdx.x / (m_sh_stride)) +
                (threadIdx.x % (m_sh_stride));
  m_gl_rd += (m_sh_stride)*slice_col;
  m_gl_rd += m_gl_rd_delta_o * slice_row;
  auto m_sh_wr = threadIdx.x;
  auto m_sh_rd = threadIdx.x % 16 + (threadIdx.x / 32) * 16;

  int s_gl_rd;
  if constexpr (group_blocks == -1) {
    s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
  } else {
    s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
              s_sh_stride * slice_col + threadIdx.x;
  }

  auto s_sh_wr = threadIdx.x;
  int s_sh_rd;
  s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
            (threadIdx.x % 32) / 4;

  bool a_sh_wr_pred[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++) {
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;
  }
  bool s_sh_wr_pred = threadIdx.x < s_sh_stride;

  auto transform_a = [&](int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;
  };
  int a_sh_wr_trans[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
  int a_sh_rd_trans[2][b_sh_wr_iters][thread_m_blocks];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
  #pragma unroll
    for (int j = 0; j < thread_m_blocks; j++) {
      a_sh_rd_trans[0][i][j] =
          transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);
      a_sh_rd_trans[1][i][j] =
          transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd + 2);
    }
  }

  const int4* B_ptr[b_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++)
    B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

  bool m_sh_wr_pred = threadIdx.x < m_sh_wr_delta;
  const int4* meta_ptr[m_sh_iters];
  #pragma unroll
  for (int i = 0; i < m_sh_iters; i++)
    meta_ptr[i] = meta + m_gl_rd_delta_i * i + m_gl_rd;

  extern __shared__ int4 sh[];
  int4* sh_a = sh;
  int4* sh_b = sh_a + (stages * a_sh_stage);
  int4* sh_s = sh_b + (stages * b_sh_stage);
  int4* sh_m = sh_s + (stages * s_sh_stage);
  FragA frag_a[2][thread_m_blocks][2];
  I4 frag_b_quant[2][b_thread_vecs];
  FragM frag_m[2][2];
  FragC frag_c[thread_m_blocks][4][2];
  FragS frag_s[2][4];

  auto zero_accums = [&]() {
  #pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
      reinterpret_cast<float*>(frag_c)[i] = 0;
  };

  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
    if (pred) {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++) {
        cp_async4_pred(
            &sh_a_stage[a_sh_wr_trans[i]],
            &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
            a_sh_wr_pred[i]);
      }
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++) {
  #pragma unroll
        for (int j = 0; j < b_thread_vecs; j++) {
          cp_async4(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr + j], B_ptr[i] + j);
        }
        B_ptr[i] += b_gl_rd_delta_o;
      }
      int4* sh_meta_stage = sh_m + m_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < m_sh_iters; i++) {
        if (m_sh_wr_pred)
          cp_async4(&sh_meta_stage[m_sh_wr_delta * i + m_sh_wr], meta_ptr[i]);
        meta_ptr[i] += m_gl_rd_delta_o;
      }
      if constexpr (group_blocks != -1) {
        static_assert(group_blocks >= thread_k_blocks);
        if (pipe % (group_blocks / thread_k_blocks) == 0) {
          int4* sh_s_stage = sh_s + s_sh_stage * pipe;
          if (s_sh_wr_pred) cp_async4(&sh_s_stage[s_sh_wr], &s[s_gl_rd]);
          s_gl_rd += s_gl_rd_delta;
        }
      }
    }
    cp_async_fence();
  };

  auto wait_for_stage = [&]() {
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  auto fetch_to_registers = [&](int k, int pipe) {
    if constexpr (group_blocks != -1) {
      static_assert(group_blocks >= thread_k_blocks);
      int4* sh_s_stage =
          sh_s + s_sh_stage * ((group_blocks / thread_k_blocks) *
                               (pipe / (group_blocks / thread_k_blocks)));
      reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
    }
    int4* sh_a_stage = sh_a + a_sh_stage * pipe;
  #pragma unroll
    for (int i = 0; i < thread_m_blocks; i++) {
      ldsm4(frag_a[k % 2][i][0],
            &sh_a_stage[a_sh_rd_trans[0][k % b_sh_wr_iters][i]]);
      ldsm4(frag_a[k % 2][i][1],
            &sh_a_stage[a_sh_rd_trans[1][k % b_sh_wr_iters][i]]);
    }

    int4* sh_b_stage = sh_b + b_sh_stage * pipe;
  #pragma unroll
    for (int i = 0; i < b_thread_vecs; i++) {
      frag_b_quant[k % 2][i] = *reinterpret_cast<I4*>(
          &sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd + i]);
    }

    int4* sh_m_stage = sh_m + m_sh_stage * pipe;
    ldsm4_m(frag_m[k % 2][0],
            &sh_m_stage[m_sh_rd_delta * (k % m_sh_iters) + m_sh_rd]);
  };

  auto matmul = [&](int k) {
  #pragma unroll
    for (int j = 0; j < 4; j++) {
      FragB frag_b0;
      FragB frag_b1;

      if constexpr (num_bits == 4) {
        int b_quant = frag_b_quant[k % 2][0][j];
        int b_quant_shift = b_quant >> 8;

        frag_b0 = dequant_4bit(b_quant);
        frag_b1 = dequant_4bit(b_quant_shift);

      } else {
        int* frag_b_quant_ptr = reinterpret_cast<int*>(frag_b_quant[k % 2]);
        int b_quant_0 = frag_b_quant_ptr[j * 2 + 0];
        int b_quant_1 = frag_b_quant_ptr[j * 2 + 1];

        frag_b0 = dequant_8bit(b_quant_0);
        frag_b1 = dequant_8bit(b_quant_1);
      }

      if constexpr (group_blocks != -1) {
        scale(frag_b0, frag_s[k % 2][j], 0);
      }
      if constexpr (group_blocks != -1) {
        scale(frag_b1, frag_s[k % 2][j], 1);
      }

  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        mma_sp(frag_b0, frag_b1, frag_a[k % 2][i][0], frag_c[i][j][0],
               frag_m[k % 2][j / 2], j % 2);
      }
    }
  };

  auto thread_block_reduce = [&]() {
    constexpr int red_off = threads / b_sh_stride_threads / 2;
    if (red_off >= 1) {
      auto red_idx = threadIdx.x / b_sh_stride_threads;
      constexpr int red_sh_stride = b_sh_stride_threads * 4 * 2;
      constexpr int red_sh_delta = b_sh_stride_threads;
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride_threads) +
                      (threadIdx.x % b_sh_stride_threads);

  #pragma unroll
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
  #pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
  #pragma unroll
            for (int j = 0; j < 4 * 2; j++) {
              int red_sh_wr =
                  red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                float* c_rd =
                    reinterpret_cast<float*>(&sh[red_sh_delta * j + red_sh_rd]);
                float* c_wr = reinterpret_cast<float*>(&sh[red_sh_wr]);
  #pragma unroll
                for (int k = 0; k < 4; k++)
                  reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + j][k] +=
                      c_rd[k] + c_wr[k];
              }
              sh[red_sh_wr] =
                  reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j];
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
  #pragma unroll
          for (int i = 0; i < 4 * 2; i++) {
            float* c_rd =
                reinterpret_cast<float*>(&sh[red_sh_delta * i + red_sh_rd]);
  #pragma unroll
            for (int j = 0; j < 4; j++)
              reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j] +=
                  c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  };

  auto global_reduce = [&](bool first = false, bool last = false) {
    constexpr int active_threads = 32 * thread_n_blocks / 4;
    if (threadIdx.x < active_threads) {
      int c_gl_stride = prob_n / 8;
      int c_gl_wr_delta_o = 2 * 4 * c_gl_stride;
      int c_gl_wr_delta_i =
          c_gl_stride;
      int c_gl_wr = 2 * c_gl_stride * (threadIdx.x % 4) +
                    8 * (threadIdx.x / 32) + (threadIdx.x % 32) / 4;
      c_gl_wr += (2 * thread_n_blocks) * slice_col;
      constexpr int c_sh_wr_delta = active_threads;
      auto c_sh_wr = threadIdx.x;

      int col = 2 * ((threadIdx.x % 32) % 4);

      if (!first) {
  #pragma unroll
        for (int i = 0; i < thread_m_blocks * 4; i++) {
          cp_async4_pred(&sh[c_sh_wr + c_sh_wr_delta * i],
                         &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                            c_gl_wr_delta_i * (i % 2)],
                         i < (thread_m_blocks - 1) * 4 ||
                             8 * (i / 2) + col + (i % 2) < prob_m);
        }
        cp_async_fence();
        cp_async_wait<0>();
      }

  #pragma unroll
      for (int i = 0; i < thread_m_blocks * 4; i++) {
        if (i < (thread_m_blocks - 1) * 4 ||
            8 * (i / 2) + col + (i % 2) < prob_m) {
          if (!first) {
            int4 c_red = sh[c_sh_wr + i * c_sh_wr_delta];
  #pragma unroll
            for (int j2 = 0; j2 < 2; j2++) {
  #pragma unroll
              for (int j1 = 0; j1 < 4; j1++) {
                reinterpret_cast<float*>(
                    &frag_c)[4 * 2 * 4 * (i / 4) + 8 * j1 + 2 * j2 +
                             4 * ((i % 4) / 2) + i % 2] +=
                    __half2float(
                        reinterpret_cast<__half*>(&c_red)[(j2 * 4 + j1)]);
              }
            }
          }
          if (!last) {
            int4 c;
  #pragma unroll
            for (int j2 = 0; j2 < 2; j2++) {
  #pragma unroll
              for (int j1 = 0; j1 < 4; j1++) {
                reinterpret_cast<__half*>(&c)[(j2 * 4 + j1)] =
                    __float2half(reinterpret_cast<float*>(
                        &frag_c)[4 * 2 * 4 * (i / 4) + 8 * j1 + 2 * j2 +
                                 4 * ((i % 4) / 2) + i % 2]);
              }
            }
            C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)] =
                c;
          }
        }
      }
    }
  };

  auto write_result = [&]() {
    int c_gl_stride = prob_n / 8;

    constexpr int c_sh_stride = 2 * thread_n_blocks;
    constexpr int c_sh_stride_2 = 2 * c_sh_stride + 2;
    constexpr int c_sh_stride_3 = 2 * (2 * thread_n_blocks) + 2;

    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;

    int c_sh_wr = c_sh_stride_2 * ((threadIdx.x % 32) % 4) +
                  ((threadIdx.x % 32) / 4);
    c_sh_wr += 8 * (threadIdx.x / 32);

    constexpr int c_sh_rd_delta =
        c_sh_stride_3 * (threads / (2 * 2 * thread_n_blocks));
    int c_sh_rd = c_sh_stride_3 * (threadIdx.x / (2 * 2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * 2 * thread_n_blocks));

    int c_gl_wr_end = c_gl_stride * prob_m;

    auto write = [&](int idx, float c0, float c1, float c2, float c3, FragS& s0,
                     float c4, float c5, float c6, float c7, FragS& s1) {
      uint2 res[2];
      res[0] = to_half4(c0, c1, c2, c3);
      res[1] = to_half4(c4, c5, c6, c7);
      half2* tmp = (half2*)&res;
      if constexpr (group_blocks == -1 && num_bits == 4) {
        tmp[0] = __hmul2(tmp[0], s0[0]);
        tmp[1] = __hmul2(tmp[1], s0[1]);
        tmp[2] = __hmul2(tmp[2], s1[0]);
        tmp[3] = __hmul2(tmp[3], s1[1]);
      }
      ((int4*)sh)[idx] = *((int4*)&res[0]);
    };

    if (threadIdx.x / 32 < thread_n_blocks / 4) {
  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        int wr = c_sh_wr;
        write(wr, frag_c[i][0][0][0], frag_c[i][1][0][0], frag_c[i][2][0][0],
              frag_c[i][3][0][0], frag_s[0][0], frag_c[i][0][0][2],
              frag_c[i][1][0][2], frag_c[i][2][0][2], frag_c[i][3][0][2],
              frag_s[0][2]);
        write(wr + c_sh_stride, frag_c[i][0][0][1], frag_c[i][1][0][1],
              frag_c[i][2][0][1], frag_c[i][3][0][1], frag_s[0][0],
              frag_c[i][0][0][3], frag_c[i][1][0][3], frag_c[i][2][0][3],
              frag_c[i][3][0][3], frag_s[0][2]);
        write(wr + 4 * c_sh_stride_2, frag_c[i][0][1][0], frag_c[i][1][1][0],
              frag_c[i][2][1][0], frag_c[i][3][1][0], frag_s[0][0],
              frag_c[i][0][1][2], frag_c[i][1][1][2], frag_c[i][2][1][2],
              frag_c[i][3][1][2], frag_s[0][2]);
        write(wr + 4 * c_sh_stride_2 + c_sh_stride, frag_c[i][0][1][1],
              frag_c[i][1][1][1], frag_c[i][2][1][1], frag_c[i][3][1][1],
              frag_s[0][0], frag_c[i][0][1][3], frag_c[i][1][1][3],
              frag_c[i][2][1][3], frag_c[i][3][1][3], frag_s[0][2]);

        c_sh_wr += 8 * c_sh_stride_2;
      }
    }
    __syncthreads();

  #pragma unroll
    for (int i = 0;
         i < ceildiv(16 * thread_m_blocks, threads / (2 * thread_n_blocks));
         i++) {
      if (c_gl_wr < c_gl_wr_end) {
        C[c_gl_wr] = sh[c_sh_rd];
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
  };

  auto start_pipes = [&]() {
  #pragma unroll
    for (int i = 0; i < stages - 1; i++) fetch_to_shared(i, i, i < slice_iters);
    zero_accums();
    wait_for_stage();
    fetch_to_registers(0, 0);
    a_gl_rd += a_gl_rd_delta_o * (stages - 1);
  };
  start_pipes();

  while (slice_iters) {
  #pragma unroll
    for (int pipe = 0; pipe < stages;) {
      fetch_to_shared((pipe + stages - 1) % stages, pipe,
                      slice_iters >= stages);
      matmul(pipe);
      wait_for_stage();

      fetch_to_registers(pipe + 1, (pipe + 1) % stages);

      pipe++;
      slice_iters--;
      if (slice_iters == 0) break;
    }
    a_gl_rd += a_gl_rd_delta_o * stages;

    if (slice_iters == 0) {
      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;
      if constexpr (group_blocks == -1) {
        if constexpr (num_bits == 8) {
          if (s_sh_wr_pred) cp_async4(&sh_s[s_sh_wr], &s[s_gl_rd]);
          cp_async_fence();
        } else {
          if (last) {
            if (s_sh_wr_pred) cp_async4(&sh_s[s_sh_wr], &s[s_gl_rd]);
            cp_async_fence();
          }
        }
      }
      thread_block_reduce();

      if constexpr (group_blocks == -1) {
        if constexpr (num_bits == 8) {
          cp_async_wait<0>();
          __syncthreads();
          if (threadIdx.x / 32 < thread_n_blocks / 4) {
            *(float4*)(frag_s) = *(float4*)(&sh_s[s_sh_rd]);
          }
        } else {
          if (last) {
            cp_async_wait<0>();
            __syncthreads();
            if (threadIdx.x / 32 < thread_n_blocks / 4) {
              *(float4*)(frag_s) = *(float4*)(&sh_s[s_sh_rd]);
            }
          }
        }
      }

      if constexpr (group_blocks == -1 && num_bits == 8) {
        if (threadIdx.x / 32 < thread_n_blocks / 4) {
  #pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
            scale_floats(&frag_c[i][0][0][0], &frag_c[i][1][0][0],
                         &frag_c[i][2][0][0], &frag_c[i][3][0][0], frag_s[0][0],
                         &frag_c[i][0][0][2], &frag_c[i][1][0][2],
                         &frag_c[i][2][0][2], &frag_c[i][3][0][2],
                         frag_s[0][2]);

            scale_floats(&frag_c[i][0][0][1], &frag_c[i][1][0][1],
                         &frag_c[i][2][0][1], &frag_c[i][3][0][1], frag_s[0][0],
                         &frag_c[i][0][0][3], &frag_c[i][1][0][3],
                         &frag_c[i][2][0][3], &frag_c[i][3][0][3],
                         frag_s[0][2]);

            scale_floats(&frag_c[i][0][1][0], &frag_c[i][1][1][0],
                         &frag_c[i][2][1][0], &frag_c[i][3][1][0], frag_s[0][0],
                         &frag_c[i][0][1][2], &frag_c[i][1][1][2],
                         &frag_c[i][2][1][2], &frag_c[i][3][1][2],
                         frag_s[0][2]);

            scale_floats(&frag_c[i][0][1][1], &frag_c[i][1][1][1],
                         &frag_c[i][2][1][1], &frag_c[i][3][1][1], frag_s[0][0],
                         &frag_c[i][0][1][3], &frag_c[i][1][1][3],
                         &frag_c[i][2][1][3], &frag_c[i][3][1][3],
                         frag_s[0][2]);
          }
        }
      }

      if (slice_count > 1) {
        barrier_acquire(&locks[slice_col], slice_idx);
        global_reduce(slice_idx == 0, last);
        barrier_release(&locks[slice_col], last);
      }
      if (last)
        write_result();

      slice_row = 0;
      slice_col_par++;
      slice_col++;
      init_slice();
      if (slice_iters) {
        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                  (threadIdx.x % a_gl_rd_delta_o);
  #pragma unroll
        for (int i = 0; i < b_sh_wr_iters; i++)
          B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
  #pragma unroll
        for (int i = 0; i < m_sh_iters; i++)
          meta_ptr[i] += (m_sh_stride)-m_gl_rd_delta_o * k_tiles;
        if (slice_col == 0) {
  #pragma unroll
          for (int i = 0; i < b_sh_wr_iters; i++) B_ptr[i] -= b_gl_stride;
  #pragma unroll
          for (int i = 0; i < m_sh_iters; i++) meta_ptr[i] -= m_gl_stride;
        }
        s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
        start_pipes();
      }
    }
  }
}

#endif

}  // namespace marlin_24
