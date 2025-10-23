template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda>
static __global__ void moe_vec_q(const void* __restrict__ vx,
                                 const void* __restrict__ vy,
                                 scalar_t* __restrict__ dst,
                                 const int* topk_ids, const int topk,
                                 const int ncols, const int nrows,
                                 const int token_stride) {
  const auto row = blockIdx.x * blockDim.y + threadIdx.y;

  const auto token = blockIdx.z / topk;
  const auto expert = (topk_ids)[blockIdx.z];

  if (row >= nrows) {
    return;
  }

  const int blocks_per_row = ncols / qk;
  const int blocks_per_warp = vdr * WARP_SIZE / qi;

  // partial sum for each thread
  float tmp = 0.0f;

  const block_q_t* x = ((const block_q_t*)vx) + expert * nrows * blocks_per_row;
  const block_q8_1* y =
      (const block_q8_1*)(((const int*)vy) + token * token_stride);

  for (auto i = threadIdx.x / (qi / vdr); i < blocks_per_row;
       i += blocks_per_warp) {
    const int ibx = row * blocks_per_row + i;  // x block index

    const int iby = i * (qk / QK8_1);  // y block index that aligns with ibx

    const int iqs =
        vdr *
        (threadIdx.x %
         (qi / vdr));  // x block quant index when casting the quants to int

    tmp += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
  }

  // sum up partial sums and write back result
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    tmp += VLLM_SHFL_XOR_SYNC(tmp, mask);
  }

  if (threadIdx.x == 0) {
    dst[blockIdx.z * nrows + row] = tmp;
  }
}
