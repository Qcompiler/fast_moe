#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])

template <const int kWarpSize = 32>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
} 

template <const int kWarpSize = 32>
__device__ __forceinline__ half warp_reduce_sum_f16(half val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

