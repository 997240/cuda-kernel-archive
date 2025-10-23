#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME marlin_moe_wna16
#endif

#define MARLIN_KERNEL_PARAMS \
    const int4* __restrict__ A_ptr, \
    const int4* __restrict__ B_ptr, \
    int4* __restrict__ C_ptr, \
    int4* __restrict__ C_tmp_ptr, \
    const int4* __restrict__ bias_ptr, \
    const void* __restrict__ s_ptr, \
    const uint16_t* __restrict__ s2_ptr, \
    const int4* __restrict__ zp_ptr, \
    const int* __restrict__ g_idx_ptr, \
    const int32_t* __restrict__ sorted_token_ids_ptr, \
    const int32_t* __restrict__ expert_ids_ptr, \
    const int32_t* __restrict__ num_tokens_past_padded_ptr, \
    const float* __restrict__ topk_weights_ptr, \
    int top_k, \
    bool mul_topk_weights, \
    bool is_ep, \
    int num_groups, \
    int prob_m, \
    int prob_n, \
    int prob_k, \
    int* locks, \
    bool has_bias, \
    bool use_atomic_add, \
    bool use_fp32_reduce, \
    int max_shared_mem

namespace MARLIN_NAMESPACE_NAME {

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS) {}

} // namespace MARLIN_NAMESPACE_NAME
