#include <cutlass/arch/arch.h>
#include <cute/tensor.hpp>
#include <cassert>
#include <cstdint>

using namespace cute;

template <typename ElementAB, typename ElementC, typename ElementSF,
          typename ElementAccumulator, typename LayoutSFA, typename LayoutSFB,
          typename ScaleConfig>
__global__ void __get_group_gemm_starts(
    ElementAB** a_offsets, ElementAB** b_offsets, ElementC** out_offsets,
    ElementSF** a_scales_offsets, ElementSF** b_scales_offsets,
    ElementAccumulator** alpha_offsets, LayoutSFA* layout_sfa_base_as_int,
    LayoutSFB* layout_sfb_base_as_int, ElementAB* a_base_as_int,
    ElementAB* b_base_as_int, ElementC* out_base_as_int,
    ElementSF* a_scales_base_as_int, ElementSF* b_scales_base_as_int,
    ElementAccumulator* alphas_base_as_int, const int32_t* expert_offsets,
    const int32_t* sf_offsets, const int32_t* problem_sizes_as_shapes,
    const int K, const int N) {
  int64_t expert_id = threadIdx.x;
  if (expert_id >= gridDim.x * blockDim.x) {
    return;
  }
  // Originally int32_t but upcasting to int64_t to avoid overflow
  // during offset calculations
  int64_t expert_offset = static_cast<int64_t>(expert_offsets[expert_id]);
  int64_t sf_offset = static_cast<int64_t>(sf_offsets[expert_id]);
  // size for block in block scale.
  int64_t group_size = 16;
  int64_t m = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3]);
  int64_t n = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3 + 1]);
  int64_t k = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3 + 2]);
  assert((m >= 0 && n == N && k == K && k % 2 == 0) &&
         "unexpected problem sizes");

  int64_t half_k = static_cast<int64_t>(k / 2);
  int64_t group_k = static_cast<int64_t>(k / group_size);
  // Shape of A as uint8/byte = [M, K // 2]
  // Shape of B as uint8/byte = [E, N, K // 2]
  a_offsets[expert_id] = a_base_as_int + expert_offset * half_k;

  b_offsets[expert_id] = b_base_as_int + expert_id * n * half_k;
  // Shape of C = [M, N]
  out_offsets[expert_id] = out_base_as_int + expert_offset * n;
  // Shape of a_scale = [sum(sf_sizes), K // group_size]
  a_scales_offsets[expert_id] = a_scales_base_as_int + sf_offset * group_k;

  assert((reinterpret_cast<uintptr_t>(a_scales_offsets[expert_id]) % 128) ==
             0 &&
         "TMA requires 128-byte alignment");

  // Shape of B scale = [E, N, K // group_size]
  b_scales_offsets[expert_id] = b_scales_base_as_int + expert_id * n * group_k;
  assert((reinterpret_cast<uintptr_t>(b_scales_offsets[expert_id]) % 128) ==
             0 &&
         "TMA requires 128-byte alignment");
  // Shape of alpha = [E]
  alpha_offsets[expert_id] = alphas_base_as_int + expert_id;

  LayoutSFA* layout_sfa_ptr = layout_sfa_base_as_int + expert_id;
  LayoutSFB* layout_sfb_ptr = layout_sfb_base_as_int + expert_id;

  *layout_sfa_ptr = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
  *layout_sfb_ptr = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
}
