#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/std/limits>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace vllm {
namespace moe {

constexpr unsigned FULL_WARP_MASK = 0xffffffff;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t BLOCK_SIZE = 512;
constexpr int32_t NUM_WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

namespace warp_topk {

template <int size, typename T>
__host__ __device__ constexpr T round_up_to_multiple_of(T len) {
  if (len == 0) {
    return 0;
  }
  return ((len - 1) / size + 1) * size;
}

template <typename T>
constexpr __host__ __device__ bool isPowerOf2(T v) {
  return (v && !(v & (v - 1)));
}

template <bool greater, typename T>
__forceinline__ __device__ bool is_better_than(T val, T baseline) {
  return (val > baseline && greater) || (val < baseline && !greater);
}

template <bool greater, typename T, typename idxT>
__forceinline__ __device__ bool is_better_than(T val, T baseline, idxT index,
                                               idxT baseline_index) {
  bool res = (val > baseline && greater) || (val < baseline && !greater);
  if (val == baseline) {
    res = (index < baseline_index && greater) ||
          (index < baseline_index && !greater);
  }
  return res;
}

template <typename T, typename idxT>
int calc_smem_size_for_block_wide(int num_of_warp, int64_t k) {
  int64_t cache_topk = (sizeof(T) + sizeof(idxT)) * num_of_warp * k;
  int64_t n = std::max<int>(num_of_warp / 2 * k, num_of_warp * WARP_SIZE);
  return max(cache_topk,
             round_up_to_multiple_of<256>(n * sizeof(T)) + n * sizeof(idxT));
}

template <int size, bool ascending, bool reverse, typename T, typename idxT,
          bool is_stable>
struct BitonicMerge {
  __device__ static void merge(T* __restrict__ val_arr,
                               idxT* __restrict__ idx_arr) {
    static_assert(isPowerOf2(size));
    static_assert(size >= 2 * WARP_SIZE);
    constexpr int arr_len = size / WARP_SIZE;

    constexpr int stride = arr_len / 2;
    for (int i = 0; i < stride; ++i) {
      int const other_i = i + stride;
      T& val = val_arr[i];
      T& other_val = val_arr[other_i];
      bool is_better;
      if constexpr (is_stable) {
        is_better = is_better_than<ascending>(val, other_val, idx_arr[i],
                                              idx_arr[other_i]);
      } else {
        is_better = is_better_than<ascending>(val, other_val);
      }

      if (is_better) {
        T tmp = val;
        val = other_val;
        other_val = tmp;

        idxT tmp2 = idx_arr[i];
        idx_arr[i] = idx_arr[other_i];
        idx_arr[other_i] = tmp2;
      }
    }

    BitonicMerge<size / 2, ascending, reverse, T, idxT, is_stable>::merge(
        val_arr, idx_arr);
    BitonicMerge<size / 2, ascending, reverse, T, idxT, is_stable>::merge(
        val_arr + arr_len / 2, idx_arr + arr_len / 2);
  }
};

template <int size, bool ascending, typename T, typename idxT, bool is_stable>
struct BitonicSort {
  __device__ static void sort(T* __restrict__ val_arr,
                              idxT* __restrict__ idx_arr) {
    static_assert(isPowerOf2(size));
    static_assert(size >= 2 * WARP_SIZE);
    constexpr int arr_len = size / WARP_SIZE;

    BitonicSort<size / 2, true, T, idxT, is_stable>::sort(val_arr, idx_arr);
    BitonicSort<size / 2, false, T, idxT, is_stable>::sort(
        val_arr + arr_len / 2, idx_arr + arr_len / 2);
    BitonicMerge<size, ascending, ascending, T, idxT, is_stable>::merge(
        val_arr, idx_arr);
  }
};

template <bool ascending, typename T, typename idxT, bool is_stable>
struct BitonicSort<32, ascending, T, idxT, is_stable> {
  __device__ static void sort(T* __restrict__ val_arr,
                              idxT* __restrict__ idx_arr) {
    int const lane = threadIdx.x % WARP_SIZE;

    for (int stage = 0; stage < 4; ++stage) {
      for (int stride = (1 << stage); stride > 0; stride /= 2) {
        bool reverse = (lane >> stage) & 2;
        bool is_second = lane & stride;

        T other = __shfl_xor_sync(FULL_WARP_MASK, *val_arr, stride);
        idxT other_idx = __shfl_xor_sync(FULL_WARP_MASK, *idx_arr, stride);

        bool is_better;
        if constexpr (is_stable) {
          if constexpr (ascending) {
            is_better = ((*val_arr > other) ||
                         ((*val_arr == other) && (*idx_arr < other_idx))) !=
                        (reverse != is_second);
          } else {
            is_better = ((*val_arr > other) ||
                         ((*val_arr == other) && (*idx_arr > other_idx))) !=
                        (reverse != is_second);
          }
        } else {
          is_better = (*val_arr != other &&
                       (*val_arr > other) != (reverse != is_second));
        }
        if (is_better) {
          *val_arr = other;
          *idx_arr = other_idx;
        }
      }
    }

    BitonicMerge<32, ascending, ascending, T, idxT, is_stable>::merge(val_arr,
                                                                      idx_arr);
  }
};

template <bool ascending, bool reverse, typename T, typename idxT,
          bool is_stable>
struct BitonicMerge<32, ascending, reverse, T, idxT, is_stable> {
  __device__ static void merge(T* __restrict__ val_arr,
                               idxT* __restrict__ idx_arr) {
    int const lane = threadIdx.x % WARP_SIZE;
    for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
      bool is_second = lane & stride;
      T& val = *val_arr;
      T other = __shfl_xor_sync(FULL_WARP_MASK, val, stride);
      idxT& idx = *idx_arr;
      idxT other_idx = __shfl_xor_sync(FULL_WARP_MASK, idx, stride);

      bool is_better;
      if constexpr (is_stable) {
        if constexpr (ascending) {
          is_better = ((*val_arr > other) ||
                       ((*val_arr == other) && (*idx_arr < other_idx))) ==
                      (reverse != is_second);
        } else {
          is_better = ((*val_arr > other) ||
                       ((*val_arr == other) && (*idx_arr > other_idx))) ==
                      (reverse != is_second);
        }
      } else {
        is_better =
            (val != other && ((val > other) == (ascending != is_second)));
      }

      if (is_better) {
        val = other;
        idx = other_idx;
      }
    }
  }
};

template <int capacity, bool greater, typename T, typename idxT, bool is_stable>
class WarpSort {
 public:
  __device__ WarpSort(idxT k, T dummy)
      : lane_(threadIdx.x % WARP_SIZE), k_(k), dummy_(dummy) {
    static_assert(capacity >= WARP_SIZE && isPowerOf2(capacity));

    for (int i = 0; i < max_arr_len_; ++i) {
      val_arr_[i] = dummy_;
      idx_arr_[i] = 0;
    }
  }

  __device__ void load_sorted(T const* __restrict__ in,
                              idxT const* __restrict__ in_idx, idxT start) {
    idxT idx = start + WARP_SIZE - 1 - lane_;
    for (int i = max_arr_len_ - 1; i >= 0; --i, idx += WARP_SIZE) {
      if (idx < start + k_) {
        T t = in[idx];
        bool is_better;
        if constexpr (is_stable) {
          is_better =
              is_better_than<greater>(t, val_arr_[i], in_idx[idx], idx_arr_[i]);
        } else {
          is_better = is_better_than<greater>(t, val_arr_[i]);
        }
        if (is_better) {
          val_arr_[i] = t;
          idx_arr_[i] = in_idx[idx];
        }
      }
    }

    BitonicMerge<capacity, greater, !greater, T, idxT, is_stable>::merge(
        val_arr_, idx_arr_);
  }

  __device__ void dump(T* __restrict__ out, idxT* __restrict__ out_idx) const {
    for (int i = 0; i < max_arr_len_; ++i) {
      idxT out_i = i * WARP_SIZE + lane_;
      if (out_i < k_) {
        out[out_i] = val_arr_[i];
        out_idx[out_i] = idx_arr_[i];
      }
    }
  }

  __device__ void dumpIdx(idxT* __restrict__ out_idx) const {
    for (int i = 0; i < max_arr_len_; ++i) {
      idxT out_i = i * WARP_SIZE + lane_;
      if (out_i < k_) {
        out_idx[out_i] = idx_arr_[i];
      }
    }
  }

 protected:
  static constexpr int max_arr_len_ = capacity / WARP_SIZE;

  T val_arr_[max_arr_len_];
  idxT idx_arr_[max_arr_len_];

  int const lane_;
  idxT const k_;
  T const dummy_;
};

template <int capacity, bool greater, typename T, typename idxT, bool is_stable>
class WarpSelect : public WarpSort<capacity, greater, T, idxT, is_stable> {
 public:
  __device__ WarpSelect(idxT k, T dummy)
      : WarpSort<capacity, greater, T, idxT, is_stable>(k, dummy),
        k_th_(dummy),
        k_th_lane_((k - 1) % WARP_SIZE) {
    extern __shared__ char smem_buf[];

    int const num_of_warp = blockDim.x / WARP_SIZE;
    int const warp_id = threadIdx.x / WARP_SIZE;
    val_smem_ = reinterpret_cast<T*>(smem_buf);
    val_smem_ += warp_id * WARP_SIZE;
    idx_smem_ = reinterpret_cast<idxT*>(
        smem_buf +
        round_up_to_multiple_of<256>(num_of_warp * sizeof(T) * WARP_SIZE));
    idx_smem_ += warp_id * WARP_SIZE;
  }

  __device__ void add(T const* in, idxT start, idxT end) {
    idxT const end_for_fullwarp =
        round_up_to_multiple_of<WARP_SIZE>(end - start) + start;
    for (idxT i = start + lane_; i < end_for_fullwarp; i += WARP_SIZE) {
      T val = (i < end) ? in[i] : dummy_;
      add(val, i);
    }
  }

  __device__ void add(T val, idxT idx) {
    bool do_add;
    if constexpr (is_stable) {
      do_add = is_better_than<greater>(val, k_th_, idx, k_th_idx_);
    } else {
      do_add = is_better_than<greater>(val, k_th_);
    }

    uint32_t mask = __ballot_sync(FULL_WARP_MASK, do_add);
    if (mask == 0) {
      return;
    }

    int pos = smem_buf_len_ + __popc(mask & ((0x1u << lane_) - 1));
    if (do_add && pos < WARP_SIZE) {
      val_smem_[pos] = val;
      idx_smem_[pos] = idx;
      do_add = false;
    }
    smem_buf_len_ += __popc(mask);
    if (smem_buf_len_ >= WARP_SIZE) {
      __syncwarp();
      merge_buf_(val_smem_[lane_], idx_smem_[lane_]);
      smem_buf_len_ -= WARP_SIZE;
    }
    if (do_add) {
      pos -= WARP_SIZE;
      val_smem_[pos] = val;
      idx_smem_[pos] = idx;
    }
    __syncwarp();
  }

  __device__ void done() {
    if (smem_buf_len_) {
      T val = (lane_ < smem_buf_len_) ? val_smem_[lane_] : dummy_;
      idxT idx = (lane_ < smem_buf_len_) ? idx_smem_[lane_] : 0;
      merge_buf_(val, idx);
    }

    __syncthreads();
  }

 private:
  __device__ void set_k_th_() {
    k_th_ = __shfl_sync(FULL_WARP_MASK, val_arr_[max_arr_len_ - 1], k_th_lane_);
    if constexpr (is_stable) {
      k_th_idx_ =
          __shfl_sync(FULL_WARP_MASK, idx_arr_[max_arr_len_ - 1], k_th_lane_);
    }
  }

  __device__ void merge_buf_(T val, idxT idx) {
    BitonicSort<WARP_SIZE, greater, T, idxT, is_stable>::sort(&val, &idx);

    T& old = val_arr_[max_arr_len_ - 1];

    bool is_better;
    if constexpr (is_stable) {
      is_better =
          is_better_than<greater>(val, old, idx, idx_arr_[max_arr_len_ - 1]);
    } else {
      is_better = is_better_than<greater>(val, old);
    }

    if (is_better) {
      old = val;
      idx_arr_[max_arr_len_ - 1] = idx;
    }

    BitonicMerge<capacity, greater, !greater, T, idxT, is_stable>::merge(
        val_arr_, idx_arr_);

    set_k_th_();
  }

  using WarpSort<capacity, greater, T, idxT, is_stable>::max_arr_len_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::val_arr_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::idx_arr_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::lane_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::k_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::dummy_;

  T* val_smem_;
  idxT* idx_smem_;
  int smem_buf_len_ = 0;

  T k_th_;
  idxT k_th_idx_;
  int const k_th_lane_;
};
}  // namespace warp_topk

template <typename T_OUT, typename T_IN>
__device__ inline T_OUT cuda_cast(T_IN val) {
  return val;
}

template <>
__device__ inline float cuda_cast<float, __nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}

template <typename T>
__device__ inline T neg_inf() {
  return cuda_cast<T, float>(-cuda::std::numeric_limits<float>::infinity());
}

template <typename T>
__device__ inline bool is_finite(const T val) {
#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120800)
  return cuda::std::isfinite(val);
#else
  return isfinite(cuda_cast<float, T>(val));
#endif
}

template <typename T, typename IdxT>
__global__ void group_idx_and_topk_idx_kernel(
    T* scores, T const* group_scores, T* topk_values, IdxT* topk_indices,
    T* scores_with_bias, int64_t const num_tokens, int64_t const n_group,
    int64_t const topk_group, int64_t const topk, int64_t const num_experts,
    int64_t const num_experts_per_group, bool renormalize,
    double routed_scaling_factor) {
  int32_t warp_id = threadIdx.x / WARP_SIZE;
  int32_t lane_id = threadIdx.x % WARP_SIZE;
  int32_t case_id =
      blockIdx.x * NUM_WARPS_PER_BLOCK + warp_id;
  scores_with_bias += case_id * num_experts;
  scores += case_id * num_experts;
  group_scores += case_id * n_group;
  topk_values += case_id * topk;
  topk_indices += case_id * topk;

  int32_t align_num_experts_per_group =
      warp_topk::round_up_to_multiple_of<WARP_SIZE>(num_experts_per_group);

  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

  extern __shared__ char smem_buf[];
  int32_t* s_topk_idx = reinterpret_cast<int32_t*>(smem_buf);
  T* s_topk_value =
      reinterpret_cast<T*>(s_topk_idx + NUM_WARPS_PER_BLOCK * topk) +
      warp_id * topk;
  s_topk_idx += warp_id * topk;

  T value = neg_inf<T>();
  T topk_group_value = neg_inf<T>();
  int32_t num_equalto_topkth_group;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  if (case_id < num_tokens) {
    int32_t target_num_min = WARP_SIZE - n_group + topk_group;
    if (lane_id < n_group && is_finite(group_scores[lane_id])) {
      value = group_scores[lane_id];
    }

    int count_equal_to_top_value = WARP_SIZE - n_group;
    int pre_count_equal_to_top_value = 0;
    while (count_equal_to_top_value < target_num_min) {
      __syncwarp();
      topk_group_value = cg::reduce(tile, value, cg::greater<T>());
      if (value == topk_group_value) {
        value = neg_inf<T>();
      }
      pre_count_equal_to_top_value = count_equal_to_top_value;
      count_equal_to_top_value =
          __popc(__ballot_sync(FULL_WARP_MASK, (value == neg_inf<T>())));
    }
    num_equalto_topkth_group = target_num_min - pre_count_equal_to_top_value;
  }
  __syncthreads();

  warp_topk::WarpSelect</*capability*/ WARP_SIZE, /*greater*/ true, T, int32_t,
                        /* is_stable */ true>
      queue((int32_t)topk, neg_inf<T>());

  int count_equalto_topkth_group = 0;
  bool if_proceed_next_topk = topk_group_value != neg_inf<T>();
  if (case_id < num_tokens && if_proceed_next_topk) {
    for (int i_group = 0; i_group < n_group; i_group++) {
      if ((group_scores[i_group] > topk_group_value) ||
          ((group_scores[i_group] == topk_group_value) &&
           (count_equalto_topkth_group < num_equalto_topkth_group))) {
        int32_t offset = i_group * num_experts_per_group;
        for (int32_t i = lane_id; i < align_num_experts_per_group;
             i += WARP_SIZE) {
          T candidates = (i < num_experts_per_group) &&
                                 is_finite(scores_with_bias[offset + i])
                             ? scores_with_bias[offset + i]
                             : neg_inf<T>();
          queue.add(candidates, offset + i);
        }
        if (group_scores[i_group] == topk_group_value) {
          count_equalto_topkth_group++;
        }
      }
    }
    queue.done();
    __syncwarp();
    queue.dumpIdx(s_topk_idx);
    __syncwarp();
  }

  float topk_sum = 1e-20;
  if (case_id < num_tokens && if_proceed_next_topk) {
    for (int i = lane_id;
         i < warp_topk::round_up_to_multiple_of<WARP_SIZE>(topk);
         i += WARP_SIZE) {
      T value =
          i < topk
              ? scores[s_topk_idx[i]]
              : cuda_cast<T, float>(0.0f);
      if (i < topk) {
        s_topk_value[i] = value;
      }
      topk_sum +=
          cg::reduce(tile, cuda_cast<float, T>(value), cg::plus<float>());
    }
  }

  __syncthreads();

  if (case_id < num_tokens) {
    if (if_proceed_next_topk) {
      for (int i = lane_id; i < topk; i += WARP_SIZE) {
        float value;
        if (renormalize) {
          value = cuda_cast<float, T>(s_topk_value[i]) / topk_sum *
                  routed_scaling_factor;
        } else {
          value = cuda_cast<float, T>(s_topk_value[i]) * routed_scaling_factor;
        }
        topk_indices[i] = s_topk_idx[i];
        topk_values[i] = cuda_cast<T, float>(value);
      }
    } else {
      for (int i = lane_id; i < topk; i += WARP_SIZE) {
        topk_indices[i] = i;
        topk_values[i] = cuda_cast<T, float>(1.0f / topk);
      }
    }
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

}  // namespace moe
}  // namespace vllm
