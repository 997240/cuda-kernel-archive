static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q5_K * x = (const block_q5_K *) vx;

    const auto i = blockIdx.x;

    const auto tid = threadIdx.x;
    const int il  = tid/16;
    const int ir  = tid%16;
    const int is  = 2*il;

    dst_t * y = yy + i*QK_K + 64*il + 2*ir;

    const half dall = __low2half(x[i].dm);
    const half dmin = __high2half(x[i].dm);

    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const half d1 = __hmul(dall, __int2half_rn(sc)); const half m1 = __hmul(dmin, __int2half_rn(m));
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const half d2 = __hmul(dall, __int2half_rn(sc)); const half m2 = __hmul(dmin, __int2half_rn(m));

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = convert_from_half<dst_t>(__hsub(__hmul(d1, __int2half_rn((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0))), m1));
    y[ 1] = convert_from_half<dst_t>(__hsub(__hmul(d1, __int2half_rn((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0))), m1));
    hm <<= 1;
    y[32] = convert_from_half<dst_t>(__hsub(__hmul(d2, __int2half_rn((ql[0] >>  4) + (qh[0] & hm ? 16 : 0))), m2));
    y[33] = convert_from_half<dst_t>(__hsub(__hmul(d2, __int2half_rn((ql[1] >>  4) + (qh[1] & hm ? 16 : 0))), m2));
}