#include "jitcu/all.h"
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h> 
#include <sys/time.h>
#include <stdint.h>
#include <assert.h>
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])

template <const int kWarpSize = 32>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
} 



__device__ __forceinline__ uint32_t ld_gbl_cs(const __restrict__ uint32_t *addr) {
	uint32_t return_value;
	asm("ld.global.cs.u32 %0, [%1];" : "=r"(return_value) : "l"(addr));
	return return_value;
}


#define WARP_SIZE 32


__device__ __forceinline__ uint2 ld_cs_u32_v2(const uint2 *p_src)
{
  uint2 n_result;
  asm("ld.global.cs.v2.u32 {%0,%1}, [%2];"  : "=r"(n_result.x), "=r"(n_result.y) : "l"(p_src));
  return n_result;
}

template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) {
    return elems[i];
  }
};

using FragB = Vec<half2, 2>;


template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile(
    "lop3.b32 %0, %1, %2, %3, %4;\n"
    : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
  );
  return res;
}
__device__ inline void dequant(int q, half2* frag_b) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  frag_b[0] = __hsub2(
    *reinterpret_cast<half2*>(&lo),
    *reinterpret_cast<const half2*>(&SUB)
  );
  frag_b[1] = __hfma2(
    *reinterpret_cast<half2*>(&hi),
    *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)
  );


}




template < int WARP_PER_ROW , int TOTAL_WARPS>
__global__ void warp_specialized_gemv_kernel(
    int32_t* __restrict__ a,
    half* __restrict__ x,
    half* __restrict__ y,
    float * scales,
    int M, int K
) {
    // extern __shared__ half shmem_vector[];
    
    int warp_id = threadIdx.y;

    // 每个线程负责4个元素，一个warp覆盖128个元素
    int tx = threadIdx.x;         // 0~31
    int ty = threadIdx.y;         // 0~3
    int bx = blockIdx.x ;          // 0~M/4
    int lane = tx % WARP_SIZE;    // 0~31
    int m = (blockDim.y * bx + ty) * WARP_PER_ROW ; // (0~M/4) * 4 + (0~3)
  
    int total_iterations = K / 8  * ((sizeof(int32_t) / sizeof(int8_t)) * 2);
    // 每个warp需要处理的迭代次数

    const int TOTAL_WARPS_ = 8;
    int iterations_per_warp = (total_iterations + TOTAL_WARPS_ - 1) / TOTAL_WARPS_;
    
    // 计算当前warp的起始迭代索引
    int start = warp_id * iterations_per_warp;
    // 计算当前warp的结束迭代索引（不包含）
    int end = min((warp_id + 1) * iterations_per_warp, total_iterations);
    
    // 每个warp内的线程以WARP_SIZE为步长进行迭代
    // if (warp_id < TOTAL_WARPS_)
    // for (int i = start + lane; i < end; i += WARP_SIZE) {
    //     *(((int4 *)shmem_vector) + i) = *(((int4 *)x) + i);
    // }

    const int NUM_PER_THREAD = 2;
    int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + NUM_PER_THREAD - 1) / NUM_PER_THREAD;
    float sum[WARP_PER_ROW];

    for (int i = 0; i < WARP_PER_ROW; ++i){
        sum[i] = 0.0;
    }

    

    __syncthreads();

  if (m < M) {
    

      #pragma unroll  4
        for (int w = 0; w < NUM_WARPS ; ++w) {
        int k = (w * WARP_SIZE + lane) * NUM_PER_THREAD;

        
            half2 reg_x_0[4];
            half2 reg_x_1[4];
            *(((int4 *)reg_x_0)) =  *(((int4 *)x) +  k);
            *(((int4 *)reg_x_1)) =  *(((int4 *)x) +  k + 1);


            for (int i = 0; i < WARP_PER_ROW; ++i){
               
                uint2 reg = ld_cs_u32_v2((uint2*)&a[( m + i ) * K + k + 0]);
                int reg_a_0 = *(reinterpret_cast<int *>(&reg)  ); 
                int reg_a_1 = *(reinterpret_cast<int *>(&reg) + 1 );
                int reg_a_0_shift = reg_a_0 >> 8;
                int reg_a_1_shift2 = reg_a_1 >> 8;                
                half2 frag_b0[4];
                half2 frag_b1[4];
                dequant(reg_a_0, frag_b0);
                dequant(reg_a_0_shift, frag_b0 + 2);
                dequant(reg_a_1, frag_b1);
                dequant(reg_a_1_shift2, frag_b1 + 2);

                #pragma unroll  4
                for (int kk = 0; kk < 4; ++kk){
                  sum[i] += (float(reg_x_0[kk].x) * float(frag_b0[kk].x) + float(reg_x_0[kk].y) * float(frag_b0[kk].y) +
                        float( reg_x_1[kk].x) * float(frag_b1[kk].x) + float(reg_x_1[kk].y) * float(frag_b1[kk].y));
                }
            }
        
      }

      // for (int i = 0; i < WARP_PER_ROW; ++i)
      //    sum[i] = warp_reduce_sum_f32<WARP_SIZE>(sum[i]);
      uint mask[1];
      half t0[1];
      half red_buf0[1];
      red_buf0[0] = __float2half(sum[0]);
      mask[0] = __activemask();
      t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 16, 32);
      red_buf0[0] = (red_buf0[0] + t0[0]);
      t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
      red_buf0[0] = (red_buf0[0] + t0[0]);
      t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
      red_buf0[0] = (red_buf0[0] + t0[0]);
      t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
      red_buf0[0] = (red_buf0[0] + t0[0]);
      t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
      red_buf0[0] = (red_buf0[0] + t0[0]);
      red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], (((int)threadIdx.y) * 32), 32);

        
      if (lane == 0){
        for (int i = 0; i < WARP_PER_ROW; ++i)
            y[m + i] = (__float2half)( __half2float (red_buf0[0]) * scales[m + i]);
      }
    
    }


}

void warp_specialized_gemv( int32_t* d_A,  half* d_B, half* d_C, float* scales,
   int M, int K, cudaStream_t stream) {

    const int NUM_WARP = 4;
    const int WARP_PER_ROW = 1;
    dim3 block(32, NUM_WARP);
    dim3 grid((M + NUM_WARP * WARP_PER_ROW - 1) / (NUM_WARP * WARP_PER_ROW), 1);


    // int sharedMemSize = K *  sizeof(half) * 8; // Shared memory for A
    
    warp_specialized_gemv_kernel<WARP_PER_ROW, NUM_WARP><<<grid, block, 0 , stream>>>( d_A, d_B,  d_C, scales,  M, K);
}
extern "C" {

void warp_specialized_gemv_host(cudaStream_t stream, jc::Tensor& weight, 
          const jc::Tensor& vector, const jc::Tensor& output, const jc::Tensor& scales) {

  int M = weight.size(0);
  int K = weight.size(1);

  warp_specialized_gemv( weight.data_ptr<int32_t>(), 
  vector.data_ptr<half>(), 
  output.data_ptr<half>(), scales.data_ptr<float>(),  M, K, stream);
  CUDA_CHECK_KERNEL_LAUNCH();
}

}