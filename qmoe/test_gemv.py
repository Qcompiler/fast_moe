import torch
from jitcu import load_cuda_ops
from triton.testing import do_bench_cudagraph,do_bench

import torch
import triton
import triton.language as tl
torch.manual_seed(0)

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
    int bx = blockIdx.x;          // 0~M/4
    int lane = tx % WARP_SIZE;    // 0~31
    int m = blockDim.y * bx + ty; // (0~M/4) * 4 + (0~3)
  
    const int TOTAL_WARPS = 2; // 
    int total_iterations = K / 8;
    // 每个warp需要处理的迭代次数
    int iterations_per_warp = (total_iterations + TOTAL_WARPS - 1) / TOTAL_WARPS;
    
    // 计算当前warp的起始迭代索引
    int start = warp_id * iterations_per_warp;
    // 计算当前warp的结束迭代索引（不包含）
    int end = min((warp_id + 1) * iterations_per_warp, total_iterations);
    
    // 每个warp内的线程以WARP_SIZE为步长进行迭代

    if (warp_id < TOTAL_WARPS)
      for (int i = start + lane; i < end; i += WARP_SIZE) {
          *(((int4 *)shmem_vector) + i) = *(((int4 *)x) + i);
      }

    int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;
    float sum = 0.0f;

    

    __syncthreads();
  // H100每个SM有256KB寄存器，支持多达65,536个寄存器（每个线程最多255个）
  // 256 KB / 128 =  2 KB
  // 2 KB =  2048 字节
  if (m < M) {
    
    // process 4 * WARP_SIZE elements per warp.
    
    #pragma unroll  4
      for (int w = 0; w < NUM_WARPS; ++w) {
      int k = (w * WARP_SIZE + lane) * 4;
      half2 reg_x_0 = HALF2(shmem_vector[k + 0]);
      half2 reg_x_1 = HALF2(shmem_vector[k + 2]);

      
      auto reg = ld_cs_u32_v2((uint2*)&a[m * K + k + 0]);
    //   uint32_t regb = ld_gbl_cs((uint32_t*)&a[m * K + k + 2]);
      half2 reg_a_0 = *(reinterpret_cast<half2 *>(&reg)  ); 
      half2 reg_a_1 = *(reinterpret_cast<half2 *>(&reg) + 1 );
      sum += (float(reg_x_0.x) * float(reg_a_0.x) + float(reg_x_0.y) * float(reg_a_0.y) +
             float( reg_x_1.x) * float(reg_a_1.x) + float(reg_x_1.y) * float(reg_a_1.y));
    }


    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (lane == 0)
      y[m] = (__float2half)(sum);

  
  }


}

void warp_specialized_gemv( half* d_A,  half* d_B, half* d_C, int M, int K, cudaStream_t stream) {

    const int NUM_WARP = 8;
    dim3 block(32, NUM_WARP);
    dim3 grid((M + NUM_WARP - 1) / NUM_WARP, 1);


    int sharedMemSize = K *  sizeof(half); // Shared memory for A
    
    warp_specialized_gemv_kernel<<<grid, block, 
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
  arches=[ "89", "90a", "120a"],
  extra_include_paths=["3rd/cutlass/include"],
  build_directory="./build",
)

device = torch.cuda.current_device()





@triton.jit
def gemv_kernel(
    A_ptr, x_ptr, y_ptr,
    m, n,
    stride_am, stride_an,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    acc = 0.0
    
    for off in range(0, n, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n
        A_offset = row_id * stride_am + cols * stride_an
        x_offset = cols
        a = tl.load(A_ptr + A_offset, mask=mask)
        x = tl.load(x_ptr + x_offset, mask=mask)
        acc += tl.sum(a * x, axis=0)
    
    tl.store(y_ptr + row_id, acc)

  


def gemv(A: torch.Tensor, vector: torch.Tensor, output, ptx = 0):
    n, k = A.shape
    device = A.device
    stride_ak, stride_an = A.stride()
    assert vector.shape[1] == A.shape[1], "Vector and input tensor shape mismatch"
    assert A.device == device and vector.device == device and output.device == device, "Tensors must be on CUDA"
    grid = lambda meta: (n, )
    
    k = gemv_kernel[grid](A, vector, output, n, k, stride_ak, stride_an,  BLOCK_SIZE = 1024)
         
    return k

import argparse
parser = argparse.ArgumentParser(description='Calculate volume of a cylinder')
# 添加参数
parser.add_argument('--marlin', type=int, default=0)

parser.add_argument('--ptx', type=int, default=0)
# 解析参数
args = parser.parse_args()



test_cases = [
      ("torch", "torch.mm"),
      ("mygemv", "warp_specialized_gemv"),
      ("triton_gemv", "gemv")
  ]

for test_name, func_name in test_cases:

  print(f"测试 {func_name}")

  for (m, k) in [(2048, 4096), (4096, 4096), (4096, 12288), (4096, 11008 * 2) ]:
    dtype = torch.float16

    activation  = torch.randint(low=-1, high=2, size=(m, k)).to(torch.float16).to(device)
    vector =  torch.randint(low=-1, high=2, size=(1, k)).to(torch.float16).to(device)

    # activation  = torch.ones((m, k)).to(torch.float16).to(device)
    # vector =  torch.ones((1, k)).to(torch.float16).to(device)
    c = torch.zeros((1, m), dtype=dtype, device=device)

    c_triton = torch.zeros((1, m), dtype=dtype, device=device)
    # print(c)
    lib.warp_specialized_gemv_host(activation, vector, c)
    kernel = gemv(activation, vector, c_triton)

    import os
    if args.ptx == 1:
  
        import os
        f = open("gemv.ptx","w")
        f.writelines(kernel.asm['ptx'])
        f.close()

    # print(c_triton)
    # print(c)
    # exit()
    torch.cuda.synchronize()
    torch.testing.assert_close(c, torch.mm(vector, activation.T), rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(c, c_triton , rtol=1e-2, atol=1e-1)


    torch.cuda.synchronize()
    
    with torch.cuda.stream(torch.cuda.Stream()):
        if test_name == "torch":
            ms = do_bench(lambda: torch.mm(vector, activation.T))
        if test_name == "mygemv":
            ms = do_bench(lambda: lib.warp_specialized_gemv_host(activation, vector, c))
        
        if test_name == 'triton_gemv':
           ms =  do_bench(lambda: gemv(activation, vector, c_triton))
        
        
        gb = (m * k + m + k) * 2 / 1024 / 1024 / 1024
        bandwidth_gb_s = (gb) / ms * 1000
        print(f"{bandwidth_gb_s=} GB/s, {gb} GB, {ms} ms")