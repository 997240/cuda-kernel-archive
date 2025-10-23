#include <cmath>

// CUDA compatibility definitions
#ifndef VLLM_LDG
#define VLLM_LDG(ptr) __ldg(ptr)
#endif

namespace vllm {

template <typename T>
__device__ __forceinline__ T swigluoai_and_mul(const T& gate, const T& up,
                                               float alpha, float limit) {
  // clamp gate: min=None, max=limit
  const float gate_f = (float)gate;
  const float clamped_gate = gate_f > limit ? limit : gate_f;

  // clamp up: min=-limit, max=limit
  const float up_f = (float)up;
  const float clamped_up =
      up_f > limit ? limit : (up_f < -limit ? -limit : up_f);

  // glu = gate * sigmoid(gate * alpha)
  const float sigmoid_val = 1.0f / (1.0f + expf(-clamped_gate * alpha));
  const float glu = clamped_gate * sigmoid_val;

  // (up + 1) * glu
  return (T)((clamped_up + 1.0f) * glu);
}

template <typename scalar_t,
          scalar_t (*ACT_FN)(const scalar_t&, const scalar_t&, const float,
                             const float)>
__global__ void swigluoai_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d, const float alpha, const float limit) {
  const int64_t token_idx = blockIdx.x;
  // TODO: Vectorize loads and stores.
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    // gate = x[..., ::2]  (even indices)
    const scalar_t gate = VLLM_LDG(&input[token_idx * 2 * d + 2 * idx]);
    // up = x[..., 1::2]   (odd indices)
    const scalar_t up = VLLM_LDG(&input[token_idx * 2 * d + 2 * idx + 1]);

    out[token_idx * d + idx] = ACT_FN(gate, up, alpha, limit);
  }
}

}  // namespace vllm