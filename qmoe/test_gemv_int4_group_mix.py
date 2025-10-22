import torch
from jitcu import load_cuda_ops
from triton.testing import do_bench_cudagraph,do_bench

import torch
import numpy as np
import torch.nn as nn
seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)


code = r"""

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





template < int WARP_PER_ROW , int TOTAL_WARPS, int N_Outliers_Per_Thread>
__global__ void warp_specialized_gemv_kernel_group_128(
    int32_t* __restrict__ a,
    half* __restrict__ x,
    half* __restrict__ y,
    float * scales,
    half *__restrict__ fpweight,
    const int32_t*  ind,
    int M, int K
) {
    extern __shared__ half shmem_vector[];
    
    int warp_id = threadIdx.y;

    // 每个线程负责4个元素，一个warp覆盖128个元素
    int tx = threadIdx.x;         // 0~31
    int ty = threadIdx.y;         // 0~3
    int bx = blockIdx.x ;          // 0~M/4
    int lane = tx % WARP_SIZE;    // 0~31
    int m = (blockDim.y * bx + ty) * WARP_PER_ROW ; // (0~M/4) * 4 + (0~3)
  
    int total_iterations = K / 8  * ((sizeof(int32_t) / sizeof(int8_t)) * 2);
    // 每个warp需要处理的迭代次数
    int iterations_per_warp = (total_iterations + TOTAL_WARPS - 1) / TOTAL_WARPS;
    
    // 计算当前warp的起始迭代索引
    int start = warp_id * iterations_per_warp;
    // 计算当前warp的结束迭代索引（不包含）
    int end = min((warp_id + 1) * iterations_per_warp, total_iterations);
    
    // 每个warp内的线程以WARP_SIZE为步长进行迭代
    for (int i = start + lane; i < end; i += WARP_SIZE) {
        *(((int4 *)shmem_vector) + i) = *(((int4 *)x) + i);
    }

    const int NUM_PER_THREAD = 2;
    int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + NUM_PER_THREAD - 1) / NUM_PER_THREAD;
    float sum[WARP_PER_ROW];


    half2 outlier[2];
    *(((uint64_t *) outlier)) =  *( (uint64_t *)fpweight + ( m ) * (128 / 4)  + lane);
    int4 ind_local =  *( (int4 *) ind + lane);
    half v1 = x[ind_local.x];
    half v2 = x[ind_local.y];
    half v3 = x[ind_local.z];
    half v4 = x[ind_local.w];
    sum[0] =   float(v1) * float(outlier[0].x)  + float(v2) * float(outlier[0].y) + 
        float(v3) * float(outlier[1].x) + float(v4) * float(outlier[1].y) ;
    __syncthreads();

    float* scales_ptr =  (float *) (  (int4 *) scales + m * ( (K  * 8) / ( 128 * 4 )) ) ;


  if (m < M) {
    

      #pragma unroll  4
        for (int w = 0; w < NUM_WARPS ; ++w) {
              int k = (w * WARP_SIZE + lane) * NUM_PER_THREAD;
              float tmp = 0.0;

              half2 reg_x_0[4];
              half2 reg_x_1[4];
              *(((int4 *)reg_x_0)) =  *(((int4 *)shmem_vector) +  k);
              *(((int4 *)reg_x_1)) =  *(((int4 *)shmem_vector) +  k + 1);
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
                    tmp += (float(reg_x_0[kk].x) * float(frag_b0[kk].x) + float(reg_x_0[kk].y) * float(frag_b0[kk].y) +
                          float( reg_x_1[kk].x) * float(frag_b1[kk].x) + float(reg_x_1[kk].y) * float(frag_b1[kk].y));
                  }
              }

              for (int i = 0; i < WARP_PER_ROW; ++i){
                    tmp *=   scales_ptr[  (k * 8 ) / 128];
                    sum[i] +=  tmp;
                }    
                     
      }
      for (int i = 0; i < WARP_PER_ROW; ++i)
          sum[i] = warp_reduce_sum_f32<WARP_SIZE>(sum[i]);
      if (lane == 0){
        for (int i = 0; i < WARP_PER_ROW; ++i)
            y[m + i] = (__float2half)(sum[i]);
      }
    
    }


}

template < int WARP_PER_ROW , int TOTAL_WARPS, int N_Outliers_Per_Thread>
__global__ void warp_specialized_gemv_kernel_group_1(
    int32_t* __restrict__ a,
    half* __restrict__ x,
    half* __restrict__ y,
    float * scales,
     half *__restrict__ fpweight,
    const int32_t*  ind,
    int M, int K
) {
    extern __shared__ half shmem_vector[];
    
    int warp_id = threadIdx.y;

    // 每个线程负责4个元素，一个warp覆盖128个元素
    int tx = threadIdx.x;         // 0~31
    int ty = threadIdx.y;         // 0~3
    int bx = blockIdx.x ;          // 0~M/4
    int lane = tx % WARP_SIZE;    // 0~31
    int m = (blockDim.y * bx + ty) * WARP_PER_ROW ; // (0~M/4) * 4 + (0~3)
  
    int total_iterations = K / 8  * ((sizeof(int32_t) / sizeof(int8_t)) * 2);
    // 每个warp需要处理的迭代次数
    int iterations_per_warp = (total_iterations + TOTAL_WARPS - 1) / TOTAL_WARPS;
    
    // 计算当前warp的起始迭代索引
    int start = warp_id * iterations_per_warp;
    // 计算当前warp的结束迭代索引（不包含）
    int end = min((warp_id + 1) * iterations_per_warp, total_iterations);
    
    // 每个warp内的线程以WARP_SIZE为步长进行迭代
    for (int i = start + lane; i < end; i += WARP_SIZE) {
        *(((int4 *)shmem_vector) + i) = *(((int4 *)x) + i);
    }

    const int NUM_PER_THREAD = 2;
    int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + NUM_PER_THREAD - 1) / NUM_PER_THREAD;
    float sum[WARP_PER_ROW];
    
    half2 outlier[2];
    *(((uint64_t *) outlier)) =  *( (uint64_t *)fpweight + ( m ) * (128 / 4)  + lane);
    int4 ind_local =  *( (int4 *) ind + lane);
    half v1 = x[ind_local.x];
    half v2 = x[ind_local.y];
    half v3 = x[ind_local.z];
    half v4 = x[ind_local.w];
    sum[0] =   float(v1) * float(outlier[0].x)  + float(v2) * float(outlier[0].y) + 
        float(v3) * float(outlier[1].x) + float(v4) * float(outlier[1].y) ;
    __syncthreads();

  if (m < M) {
    

      #pragma unroll  4
        for (int w = 0; w < NUM_WARPS ; ++w) {
        int k = (w * WARP_SIZE + lane) * NUM_PER_THREAD;

        
            half2 reg_x_0[4];
            half2 reg_x_1[4];
            *(((int4 *)reg_x_0)) =  *(((int4 *)shmem_vector) +  k);
            *(((int4 *)reg_x_1)) =  *(((int4 *)shmem_vector) +  k + 1);
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
      for (int i = 0; i < WARP_PER_ROW; ++i)
          sum[i] = warp_reduce_sum_f32<WARP_SIZE>(sum[i]);
      if (lane == 0){
        for (int i = 0; i < WARP_PER_ROW; ++i)
            y[m + i] = (__float2half)(sum[i] * scales[m + i]);
      }
    
    }


}

void warp_specialized_gemv_mix( int32_t* d_A,  half* d_B, half* d_C,
      float* scales, half* fpweight, int32_t * ind, 
      int M, int K, cudaStream_t stream, int groupsize) {

    const int NUM_WARP = 8;
    const int WARP_PER_ROW = 1;
    dim3 block(32, NUM_WARP);
    dim3 grid((M + NUM_WARP * WARP_PER_ROW - 1) / (NUM_WARP * WARP_PER_ROW), 1);

    int sharedMemSize;
     
    sharedMemSize = K *  sizeof(half) * 8 ; // Shared memory for A

  
    const int N_Outliers_Per_Thread = 4;
    switch (groupsize) {
      case 128:
        warp_specialized_gemv_kernel_group_128<WARP_PER_ROW, NUM_WARP, N_Outliers_Per_Thread><<<grid, block, 
        sharedMemSize, stream>>>( d_A, d_B,  d_C, scales, fpweight, ind,  M, K);
        break;
      case -1:
        warp_specialized_gemv_kernel_group_1<WARP_PER_ROW, NUM_WARP, N_Outliers_Per_Thread><<<grid, block, 
        sharedMemSize, stream>>>( d_A, d_B,  d_C, scales, fpweight, ind, M, K);
        break;
      default:
        printf("groupsize not support\n");   
        exit(0); 
    }

}
extern "C" {

void warp_specialized_gemv_host_mix(cudaStream_t stream, jc::Tensor& weight, 
          const jc::Tensor& vector, const jc::Tensor& output, 
          const jc::Tensor& scales, const jc::Tensor& fpweight,
          const jc::Tensor& ind, int groupsize) {

  int M = weight.size(0);
  int K = weight.size(1);

  warp_specialized_gemv_mix( weight.data_ptr<int32_t>(), 
  vector.data_ptr<half>(), 
  output.data_ptr<half>(), scales.data_ptr<float>(), fpweight.data_ptr<half>(), 
   ind.data_ptr<int32_t>(), M, K, stream, groupsize);
  CUDA_CHECK_KERNEL_LAUNCH();

}

}

"""




lib = load_cuda_ops(
  name="test",
  sources=code,
  func_names=["warp_specialized_gemv_host_mix"],
  func_params=["t_t_t_t_t_t"],
  arches=[ "89", "90a"],
  extra_include_paths=["3rd/cutlass/include"],
  build_directory="./build",
)

device = torch.cuda.current_device()


import argparse
parser = argparse.ArgumentParser(description='Calculate volume of a cylinder')
# 添加参数
parser.add_argument('--marlin', type=int, default=0)
parser.add_argument('--g', type=int, default=-1)
# 解析参数
args = parser.parse_args()

from common.common import generate_randint, gen_quant4, gen_quant4_my
for (out_dim, k) in [ (2048, 2048), (2048, 4096), (4096, 4096), (12288, 4096), (8192, 8192) ]:
  dtype = torch.float16

  groupsize = args.g


  c = torch.zeros((1, out_dim), dtype=dtype, device=device)

  weight, vector = generate_randint(k, out_dim, device)
  grand = torch.mm(vector, weight.T)

  #-------------------------------------marlin----------------------------
  import marlin
  workspace = torch.zeros(out_dim // 128 * 16, device=device)
  C_i4mar = torch.zeros((1, out_dim), dtype=dtype, device=device)
  thread_k = 64
  thread_n = 256
  


  _, B, s = gen_quant4(k, out_dim, weight.t().contiguous(),   groupsize =  groupsize)  

  marlin.mul(vector, B, C_i4mar, s, workspace, thread_k, thread_n, -1)
  torch.cuda.synchronize()

  
  
  n_outliers = 128
  ind = torch.as_tensor(range(n_outliers)).to(torch.int32).cuda()
  weight_cache =   weight[:,ind].contiguous()
  weight[:,ind] = 0

  q_weight, scales  = gen_quant4_my(out_dim, k, torch.clone(weight),   groupsize = groupsize, tile = 1)

  if groupsize == -1:
    weight_cache = weight_cache / scales
  scales = scales.to(torch.float32)


  

  lib.warp_specialized_gemv_host_mix(q_weight, vector, c, scales,  weight_cache, ind, groupsize)

  # print(c)
  # print(grand)
  # print(C_i4mar)

  #------------------------------------mixq----------------------------------
  # print((grand - C_i4mar).max())
  torch.testing.assert_close(grand, C_i4mar, rtol=1e-2, atol=1e-1)
  torch.testing.assert_close(c, C_i4mar, rtol=1e-2, atol=1e-1)
 

  torch.cuda.synchronize()
  with torch.cuda.stream(torch.cuda.Stream()):
    if args.marlin == 1:
      ms = do_bench(lambda: marlin.mul(vector, B, C_i4mar, s, workspace, thread_k, thread_n, -1))
    else:
      ms = do_bench(lambda: lib.warp_specialized_gemv_host_mix(q_weight, vector, c, scales, weight_cache, ind, groupsize))
    
    # ms = do_bench(lambda: torch.mm(vector, weight.T))


  
  # double gflops = (2.0 * M * N) / (kernel_time * 1e9);  // 每个元素需要1次乘法和1次加法
  # double bandwidth_gb_s = ((M * N + M + N) * sizeof(half) * 1e-9) / kernel_time;
  gb = (out_dim * k + out_dim + k) * 2 / 1024 / 1024 / 1024
  bandwidth_gb_s = (gb) / ms * 1000
  print(f"{bandwidth_gb_s=} GB/s, {gb} GB, {ms} ms")