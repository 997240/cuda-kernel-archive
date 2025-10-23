#include <assert.h>
#include <cuda.h>
#include <torch/all.h>

__device__ int64_t save_blocks(int* block_offset, int64_t range_start,
                               int64_t range_end, int64_t block_size,
                               int64_t input_block_count, int64_t kv_seqlen) {
  if (range_start >= kv_seqlen) {
    return input_block_count;
  }
  if (range_end > kv_seqlen) {
    range_end = kv_seqlen;
  }
  int64_t current_block_count = input_block_count;
  for (int idx = range_start; idx < range_end; idx += block_size) {
    block_offset[current_block_count++] = idx;
  }
  return current_block_count;
}

__global__ void convert_vertical_slash_indexes_kernel(
    const int* q_seqlens,
    const int* kv_seqlens,
    const int* vertical_indexes,
    const int* slash_indexes,
    int* block_count,
    int* block_offset,
    int* column_count,
    int* column_index,
    int64_t N_HEADS,
    int64_t N_ROWS,
    int64_t BLOCK_SIZE_M,
    int64_t BLOCK_SIZE_N,
    int64_t NNZ_V,
    int64_t NNZ_S,
    bool causal) {
  const int batch_idx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int group_idx = blockIdx.z;

  int64_t q_seqlen = q_seqlens[batch_idx];
  int64_t kv_seqlen = kv_seqlens[batch_idx];
  int64_t block_idx_m = group_idx * blockDim.x + threadIdx.x;
  int64_t start_m = block_idx_m * BLOCK_SIZE_M;
  if (start_m >= q_seqlen) {
    return;
  }
  int64_t end_m = start_m + BLOCK_SIZE_M;
  vertical_indexes += (batch_idx * N_HEADS + head_idx) * NNZ_V;
  slash_indexes += (batch_idx * N_HEADS + head_idx) * NNZ_S;
  int64_t row_offset = (batch_idx * N_HEADS + head_idx) * N_ROWS + block_idx_m;
  block_count += row_offset;
  block_offset += row_offset * NNZ_S;
  column_count += row_offset;
  column_index += row_offset * NNZ_V;

  bool has_slash = true;
  int64_t tmp_col_cnt = 0, tmp_blk_cnt = 0;
  int64_t s = 0, v = 0;
  int64_t v_idx = vertical_indexes[v++];
  int64_t s_idx = slash_indexes[s++];
  if (causal) {
    while (s_idx >= end_m + (kv_seqlen - q_seqlen) && s < NNZ_S) {
      s_idx = slash_indexes[s++];
    }
    if (s_idx > end_m + (kv_seqlen - q_seqlen)) has_slash = false;
    s_idx = max((kv_seqlen - q_seqlen) + end_m - s_idx, BLOCK_SIZE_M);
  } else {
    while (s_idx >= end_m + kv_seqlen && s < NNZ_S) {
      s_idx = slash_indexes[s++];
    }
    if (s_idx > end_m + kv_seqlen) has_slash = false;
    s_idx = max(kv_seqlen + end_m - s_idx, BLOCK_SIZE_M);
  }

  int64_t range_start = s_idx - BLOCK_SIZE_M, range_end = s_idx;
  if (!has_slash) {
    if (causal) {
      range_start = (kv_seqlen - q_seqlen) + end_m;
      range_end = (kv_seqlen - q_seqlen) + end_m + BLOCK_SIZE_N;
    } else {
      range_start = kv_seqlen;
      range_end = kv_seqlen + BLOCK_SIZE_N;
    }
  }

  bool slash_finished = false;
  while (1) {
    if (v_idx < range_end) {
      if (v_idx < range_start) {
        column_index[tmp_col_cnt++] = v_idx;
      }
      if (v < NNZ_V) {
        v_idx = vertical_indexes[v++];
      } else {
        if (causal)
          v_idx = end_m + BLOCK_SIZE_N + (kv_seqlen - q_seqlen);
        else
          v_idx = end_m + BLOCK_SIZE_N + kv_seqlen;
      }
    } else {
      if ((s < NNZ_S && causal) ||
          (s < NNZ_S && !causal && slash_indexes[s] >= start_m)) {
        if (causal)
          s_idx = max((kv_seqlen - q_seqlen) + end_m - slash_indexes[s++],
                      BLOCK_SIZE_M);
        else
          s_idx = max(kv_seqlen + end_m - slash_indexes[s++], BLOCK_SIZE_M);
      } else {
        if (v == NNZ_V || (v_idx > range_start && causal)) {
          if (v == NNZ_V && !causal && v_idx < kv_seqlen) {
            column_index[tmp_col_cnt++] = v_idx;
          }
          tmp_blk_cnt = save_blocks(block_offset, range_start, range_end,
                                    BLOCK_SIZE_N, tmp_blk_cnt, kv_seqlen);
          break;
        } else {
          if (causal) {
            range_start = (kv_seqlen - q_seqlen) + end_m;
            range_end = (kv_seqlen - q_seqlen) + end_m + BLOCK_SIZE_N;
          } else {
            tmp_blk_cnt = save_blocks(block_offset, range_start, range_end,
                                      BLOCK_SIZE_N, tmp_blk_cnt, kv_seqlen);
            range_start = kv_seqlen;
            range_end = kv_seqlen + BLOCK_SIZE_N;
          }
          slash_finished = true;
        }
      }
      if (!slash_finished) {
        if (s_idx > range_end + BLOCK_SIZE_M) {
          tmp_blk_cnt = save_blocks(block_offset, range_start, range_end,
                                    BLOCK_SIZE_N, tmp_blk_cnt, kv_seqlen);
          range_start = s_idx - BLOCK_SIZE_M;
          range_end = s_idx;
        } else if (s_idx > range_end) {
          range_end += BLOCK_SIZE_M;
        }
      }
    }
  }

  block_count[0] = tmp_blk_cnt;
  column_count[0] = tmp_col_cnt;
}
