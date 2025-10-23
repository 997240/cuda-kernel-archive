namespace machete {

template <int threads, typename PrepackedLayoutB, typename BInTensor,
          typename ElementB>
static __global__ void prepack_B_kernel(BInTensor B_in, ElementB* B_out_ptr) {
  auto constexpr block_size =
      Int<size(typename PrepackedLayoutB::PPBlockShape_NK{})>{};
  auto constexpr eles_per_thread = Int<block_size / threads>{};
  static_assert(block_size % threads == 0,
                "block_size must be divisible by the number of threads");

  auto blk_coord = make_coord(blockIdx.x, blockIdx.y, blockIdx.z);
  auto tB_in = local_tile(
      B_in, append(typename PrepackedLayoutB::PPBlockShape_NK{}, _1{}),
      blk_coord);

  auto bNbKL_to_offset = PrepackedLayoutB::bNbKL_to_offset(shape(B_in));

  auto tB_out_linear =
      make_tensor(get_logical_ptr(B_out_ptr) + bNbKL_to_offset(blk_coord),
                  make_layout(make_shape(block_size)));
  auto tB_in_linear = make_tensor(
      tB_in.data(),
      tB_in.layout()
          .compose(right_inverse(PrepackedLayoutB::ppblock_ilvd_NK_to_offset()))
          .with_shape(make_shape(block_size)));

  auto thr_tB_in_linear =
      local_tile(tB_in_linear, make_shape(eles_per_thread), threadIdx.x);
  auto thr_tB_out_linear =
      local_tile(tB_out_linear, make_shape(eles_per_thread), threadIdx.x);

  auto fragment = make_tensor<ElementB>(shape(thr_tB_in_linear));

  copy(thr_tB_in_linear, fragment);
  copy(Copy_Atom<DefaultCopy, uint8_t>{}, fragment, thr_tB_out_linear);
}

}  // namespace machete
