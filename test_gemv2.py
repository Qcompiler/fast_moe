import torch
from jitcu import load_cuda_ops
from triton.testing import do_bench_cudagraph,do_bench

"export PYTHONPATH=/home/chenyidong/newstart/bandwidth/jitcu"


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

template <const int kWarpSize = 32>
__device__ __forceinline__ half warp_reduce_sum_f16(half val) {
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


template < int WARP_PER_ROW , int TOTAL_WARPS>
__global__ void warp_specialized_gemv_kernel(
    half* __restrict__ a,
    half* __restrict__ x,
    half* __restrict__ y,
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
  
    int total_iterations = K / 8;
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

    const int NUM_PER_THREAD = 4;
    int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + NUM_PER_THREAD - 1) / NUM_PER_THREAD;
    float sum[WARP_PER_ROW];

    for (int i = 0; i < WARP_PER_ROW; ++i){
        sum[i] = 0.0;
    }

    

    __syncthreads();

  if (m < M) {
    

      #pragma unroll  4
        for (int w = 0; w < NUM_WARPS; ++w) {
        int k = (w * WARP_SIZE + lane) * NUM_PER_THREAD;

        for (int p = 0; p < NUM_PER_THREAD / 4; ++p){
            half2 reg_x_0 = HALF2(shmem_vector[k + p * 4]);
            half2 reg_x_1 = HALF2(shmem_vector[k + 2 + p * 4]);


            for (int i = 0; i < WARP_PER_ROW; ++i){
                uint2 reg = ld_cs_u32_v2((uint2*)&a[( m + i ) * K + k + 0 + p * 4]);

                half2 reg_a_0 = *(reinterpret_cast<half2 *>(&reg)  ); 
                half2 reg_a_1 = *(reinterpret_cast<half2 *>(&reg) + 1 );
                sum[i] += (float(reg_x_0.x) * float(reg_a_0.x) + float(reg_x_0.y) * float(reg_a_0.y) +
                      float( reg_x_1.x) * float(reg_a_1.x) + float(reg_x_1.y) * float(reg_a_1.y));
            }
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

void warp_specialized_gemv( half* d_A,  half* d_B, half* d_C, int M, int K, cudaStream_t stream) {

    const int NUM_WARP = 8;
    const int WARP_PER_ROW = 1;
    dim3 block(32, NUM_WARP);
    dim3 grid((M + NUM_WARP * WARP_PER_ROW - 1) / (NUM_WARP * WARP_PER_ROW), 1);


    int sharedMemSize = K *  sizeof(half); // Shared memory for A
    
    warp_specialized_gemv_kernel<WARP_PER_ROW, NUM_WARP><<<grid, block, 
    sharedMemSize, stream>>>( d_A, d_B,  d_C,   M, K);
}


extern "C" {

void warp_specialized_gemv_host(cudaStream_t stream, jc::Tensor& activation, 
          const jc::Tensor& vector, const jc::Tensor& output) {

  int M = activation.size(0);
  int K = activation.size(1);
  warp_specialized_gemv( activation.data_ptr<half>(), 
  vector.data_ptr<half>(), 
  output.data_ptr<half>(), M, K, stream);
  CUDA_CHECK_KERNEL_LAUNCH();
}

}

"""

lib = load_cuda_ops(
  name="test",
  sources=code,
  func_names=["warp_specialized_gemv_host"],
  func_params=["t_t_t"],
  arches=[ "90a"],
  extra_include_paths=["3rd/cutlass/include"],
  build_directory="./build",
)

device = torch.cuda.current_device()



for (m, k) in [(2048, 2048), (4096, 4096), (4096, 12288), (4096, 11008 * 2) ]:
  dtype = torch.float16

  activation = torch.randn((m, k), dtype=dtype, device=device) 
  vector = torch.randn((1, k), dtype=dtype, device=device) 
  c = torch.zeros((1, m), dtype=dtype, device=device)
  # print(c)
  lib.warp_specialized_gemv_host(activation, vector, c)

  # print(c)
  # print(torch.mm(vector, activation.T))
  torch.cuda.synchronize()
  torch.testing.assert_close(c, torch.mm(vector, activation.T), rtol=1e-2, atol=1e-1)


  torch.cuda.synchronize()
  with torch.cuda.stream(torch.cuda.Stream()):
    ms = do_bench(lambda: lib.warp_specialized_gemv_host(activation, vector, c))
    # ms = do_bench(lambda: torch.mm(vector, activation.T))


  
  # double gflops = (2.0 * M * N) / (kernel_time * 1e9);  // 每个元素需要1次乘法和1次加法
  # double bandwidth_gb_s = ((M * N + M + N) * sizeof(half) * 1e-9) / kernel_time;
  gb = (m * k + m + k) * 2 / 1024 / 1024 / 1024
  bandwidth_gb_s = (gb) / ms * 1000
  print(f"{bandwidth_gb_s=} GB/s, {gb} GB, {ms} ms")