template<typename dst_t>
static __global__ void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q6_K * x = (const block_q6_K *) vx;

    const auto i = blockIdx.x;

    const auto tid = threadIdx.x;
    const int ip  = tid/32;
    const int il  = tid - 32*ip;
    const int is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

    const half d = x[i].d;

    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t   qh = x[i].qh[32*ip + il];
    const int8_t  * sc = x[i].scales + is;

    y[ 0] = convert_from_half<dst_t>(__hmul(d, __int2half_rn(sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32))));
    y[32] = convert_from_half<dst_t>(__hmul(d, __int2half_rn(sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32))));
    y[64] = convert_from_half<dst_t>(__hmul(d, __int2half_rn(sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32))));
    y[96] = convert_from_half<dst_t>(__hmul(d, __int2half_rn(sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32))));
}