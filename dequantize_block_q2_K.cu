template<typename dst_t>
static __global__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const auto i   = blockIdx.x;
    const block_q2_K * x = (const block_q2_K *) vx;

    const auto tid = threadIdx.x;
    const int n   = tid/32;
    const int l   = tid - 32*n;
    const int is  = 8*n + l/16;

    const uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    half dall = __low2half(x[i].dm);
    half dmin = __high2half(x[i].dm);
    y[l+ 0] = convert_from_half<dst_t>(__hsub(__hmul(dall, __int2half_rn((x[i].scales[is+0] & 0xF) * ((q >> 0) & 3))), __hmul(dmin,  __int2half_rn(x[i].scales[is+0] >> 4))));
    y[l+32] = convert_from_half<dst_t>(__hsub(__hmul(dall, __int2half_rn((x[i].scales[is+2] & 0xF) * ((q >> 2) & 3))), __hmul(dmin,  __int2half_rn(x[i].scales[is+2] >> 4))));
    y[l+64] = convert_from_half<dst_t>(__hsub(__hmul(dall, __int2half_rn((x[i].scales[is+4] & 0xF) * ((q >> 4) & 3))), __hmul(dmin,  __int2half_rn(x[i].scales[is+4] >> 4))));
    y[l+96] = convert_from_half<dst_t>(__hsub(__hmul(dall, __int2half_rn((x[i].scales[is+6] & 0xF) * ((q >> 6) & 3))), __hmul(dmin,  __int2half_rn(x[i].scales[is+6] >> 4))));
}