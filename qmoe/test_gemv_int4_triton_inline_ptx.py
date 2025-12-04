import torch
from jitcu import load_cuda_ops
from triton.testing import do_bench, do_bench_cudagraph

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




template < int WARP_PER_ROW , int TOTAL_WARPS>
__global__ void warp_specialized_gemv_kernel(
    int32_t* __restrict__ a,
    half* __restrict__ x,
    half* __restrict__ y,
    float * scales,
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

    const int TOTAL_WARPS_ = 8;
    int iterations_per_warp = (total_iterations + TOTAL_WARPS_ - 1) / TOTAL_WARPS_;
    
    // 计算当前warp的起始迭代索引
    int start = warp_id * iterations_per_warp;
    // 计算当前warp的结束迭代索引（不包含）
    int end = min((warp_id + 1) * iterations_per_warp, total_iterations);
    
    // 每个warp内的线程以WARP_SIZE为步长进行迭代
    if (warp_id < TOTAL_WARPS_)
    for (int i = start + lane; i < end; i += WARP_SIZE) {
        *(((int4 *)shmem_vector) + i) = *(((int4 *)x) + i);
    }

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

void warp_specialized_gemv( int32_t* d_A,  half* d_B, half* d_C, float* scales, int M, int K, cudaStream_t stream) {

    const int NUM_WARP = 8;
    const int WARP_PER_ROW = 1;
    dim3 block(32, NUM_WARP);
    dim3 grid((M + NUM_WARP * WARP_PER_ROW - 1) / (NUM_WARP * WARP_PER_ROW), 1);


    int sharedMemSize = K *  sizeof(half) * 8; // Shared memory for A
    
    warp_specialized_gemv_kernel<WARP_PER_ROW, NUM_WARP><<<grid, block, 
    sharedMemSize, stream>>>( d_A, d_B,  d_C, scales,  M, K);
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

"""

import torch, math, random, copy
from torch import Tensor
import triton
import triton.language as tl


 




@triton.jit
def gemv_int4_kernel(
    A_ptr, x_ptr, y_ptr,
    m, k,
    stride_am, stride_an,
    BLOCK_SIZE: tl.constexpr,
    unpack_mask: tl.constexpr,
    a_evict: tl.constexpr = 'evict_last',
    b_evict: tl.constexpr = 'evict_first',
):
    row_id = tl.program_id(0)
    acc = 0.0
    elements_per_sample = 8
    W_nbits = 4
    
    offs_k =  tl.arange(0, BLOCK_SIZE)
    A_offset = row_id * stride_am + (offs_k // 8) * stride_an
    
    for kk in range(0, tl.cdiv(k, BLOCK_SIZE)):
          
      mask = offs_k < k
      
      a = tl.load(A_ptr + A_offset,  eviction_policy=a_evict)

      
      q_shift = ((offs_k % elements_per_sample) * W_nbits).to(tl.int32)
      a = ((a >> q_shift) & unpack_mask) - 8 
      a = a.to(tl.float16)
      x = tl.load(x_ptr + offs_k, mask = mask, other=0.0)
      acc += tl.sum(a * x, axis=0) 

      offs_k +=  BLOCK_SIZE
      A_offset += (BLOCK_SIZE // 8) * stride_an

    tl.store(y_ptr + row_id, acc)






def gemv_int4(A: torch.Tensor, vector: torch.Tensor, output, ptx = 0):
    n, _ = A.shape
    k = vector.shape[1] # [m, k ]
    device = A.device
    stride_ak, stride_an = A.stride()
    assert vector.shape[1] == A.shape[1] * 8, "Vector and input tensor shape mismatch"
    assert A.device == device and vector.device == device and output.device == device, "Tensors must be on CUDA"
    grid = lambda meta: (n, 1)
    
    
    kernel = gemv_int4_kernel[grid](A, vector, output, n, k, stride_ak, stride_an,  
                                BLOCK_SIZE = 1024, unpack_mask = 15)
         




    return kernel



@triton.jit
def dequanti(b):
    x1, x2 = tl.inline_asm_elementwise(
        asm="""
            {
            .reg .b32 	r<16>;
            .reg .b32  r_high<2>, r_low<2>;
	          .reg .b64 	rd<2>;
            .reg .u16 tmp1, tmp2, tmp3, tmp4;            
            mov.u32 r2, $2;
            mov.u32 	r3, 983055;
            mov.u32 	r8, 1677747200;
            lop3.b32 r1, r2, r3, r8, 234;
            mov.u32 	r7, 15728880;
            lop3.b32 r5, r2, r7, r8, 234;
            mov.u32 	r11, 1678271496;
            mov.u32 	r14, 738208768;
            mov.u32 	r15, -729754496;
            fma.rn.f16x2 r12,r5,r14,r15;
            sub.f16x2 r9,r1,r11;            
            shr.s32   r_high1, r9, 16;
            cvt.u16.u32   tmp1, r_high1;
            and.b32       r_low1, r9, 0xFFFF;
            cvt.u16.u32   tmp2, r_low1;
            shr.s32   r_high1, r12, 16;
            cvt.u16.u32   tmp3, r_high1;
            and.b32       r_low1, r12, 0xFFFF;
            cvt.u16.u32   tmp4, r_low1;
            mov.b32 $1, {tmp4, tmp3};   
            mov.b32 $0, {tmp2, tmp1};  
            }
        """,
        constraints=(
            "=r,=r,r"
        ),
        args=[b], #输入
        dtype=(tl.uint32, tl.uint32), #输出
        is_pure=False,
        pack=1,
    )
    return x1, x2




@triton.jit
def sum_4_half(x1, x2, vec1, vec2):
    # x1 : int32 x2: int32
    # vec1 : int32 vec2: int32
    # x1 * vec1 + x2 * vec2
    y = tl.inline_asm_elementwise(
        asm="""
            {
              .reg .b32 vec1, vec2, vec_sum;
              .reg .b16 h_low, h_high, h_final;
              .reg .b32 x1, x2;

              mov.b32 vec1, $1;
              mov.b32 vec2, $2;
              mov.b32 x1, $3;
              mov.b32 x2, $4;

              mul.f16x2 vec1, vec1, x1;
              mul.f16x2 vec2, vec2, x2;
              add.f16x2 vec_sum, vec1, vec2;
              mov.b32 {h_high, h_low}, vec_sum; 
              add.f16 h_final, h_high, h_low;              
              cvt.u16.u16 $0,  h_final;
               
            }
        """,
        constraints=(
            "=f,r,r,r,r"
        ),
        args=[x1, x2, vec1, vec2],  # 参数 1 2 3 4
        dtype=(tl.float16), #参数 0
        is_pure=False,
        pack=1,
    )

    return y


@triton.jit
def load_v4_b32(ptr):
    return tl.inline_asm_elementwise(
        asm="ld.global.v4.u32 {$0,$1,$2,$3}, [$4];",
        constraints=("=r,=r,=r,=r,l"),
        args=[ptr],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=False,
        pack=1
    )



# def get_autotune_config():
#     configs = []
#     # for evict in ['evict_last', 'evict_first', None]:
#     for block_size in [128, 256, 512]:
#           configs.append(triton.Config({ 'BLOCK_SIZE' : block_size}))

#     return configs

# @triton.autotune(
#     configs=get_autotune_config(),
#     key = ['m', 'k', 'int4_k'],
#     use_cuda_graph = False
# )

# bandwidth_gb_s=2771.1846914627185 GB/s, 0.0312652587890625 GB, 0.011282271761020632 ms
# bandwidth_gb_s=1781.0717729268401 GB/s, 0.015636444091796875 GB, 0.008779233004238489 ms
# bandwidth_gb_s=3476.3029263603253 GB/s, 0.06252288818359375 GB, 0.017985454521092313 ms
# bandwidth_gb_s=4039.256917884022 GB/s, 0.093780517578125 GB, 0.02321726978120823 ms
# bandwidth_gb_s=4197.0196543086395 GB/s, 0.125030517578125 GB, 0.029790310238306673 ms

# bandwidth_gb_s=2711.641957021821 GB/s, 0.0312652587890625 GB, 0.011530009966139091 ms
# bandwidth_gb_s=1844.975788655275 GB/s, 0.015636444091796875 GB, 0.008475148664792842 ms
# bandwidth_gb_s=3276.7513242093223 GB/s, 0.06252288818359375 GB, 0.019080754685795533 ms
# bandwidth_gb_s=3878.28912972846 GB/s, 0.093780517578125 GB, 0.024180898958580502 ms
# bandwidth_gb_s=4084.921795699678 GB/s, 0.125030517578125 GB, 0.03060781180921222 ms
@triton.jit
def test_dequant_kernel(
    A_ptr, x_ptr, y_ptr,
    m, k, int4_k,
    stride_am, 
    BLOCK_SIZE : tl.constexpr = 256,
    evict : tl.constexpr = 'evict_last'
):



    row_id = tl.program_id(0)
    acc = 0
    acc = acc.to(tl.float16)


    offs_k =  tl.arange(0, BLOCK_SIZE)
    A_offset = row_id * stride_am + (offs_k)
    
    A_ptr = A_ptr + A_offset
    x_ptr = x_ptr + (offs_k * 2)
    # x_ptr_int = tl.cast(x_ptr, (tl.uint32), bitcast = True)
    for kk in range(0, tl.cdiv(int4_k, BLOCK_SIZE)):
      mask = offs_k < int4_k
      x1, x2, x3, x4 = load_v4_b32(x_ptr)
      a = tl.load(A_ptr,  eviction_policy = evict, mask = mask)
      a1, a2 = dequanti(a)  
      all1 = sum_4_half(a1, a2, x1, x2)     
      a = a >> 8
      a5, a6 = dequanti(a)      
      all2 = sum_4_half(a5, a6, x3, x4) 
      
      acc += tl.sum(all1 + all2, axis=0) 
      offs_k +=  BLOCK_SIZE 
      A_ptr += (BLOCK_SIZE) 
      x_ptr += (BLOCK_SIZE * 2)

    tl.store(y_ptr + row_id, acc)


def test_dequant(A: torch.Tensor, vector: torch.Tensor, output):
    n, _ = A.shape
    k = vector.shape[1] # [m, k ]
    device = A.device
    stride_ak, stride_an = A.stride()
    stride_cm, stride_cn = output.stride()
    assert vector.shape[1] == A.shape[1] * 8, "Vector and input tensor shape mismatch"
    assert A.device == device and vector.device == device and output.device == device, "Tensors must be on CUDA"
    grid = lambda meta: (n, 1)
    
    storage = vector.untyped_storage()
    # k = vector.numel()  # 元素数量
    uint64_tensor = torch.tensor([], dtype=torch.uint64,device=device).set_(storage, 0, (k // 4,))
    int4_k = int(A.shape[1])
    
    # if out_dim > 4096:
    #   evict = 'evict_first'
    # else:
    #   evict = 'evict_last'
    
    # ret = triton.compile(kernel, signature="*fp32,i32,*fp32,i32", constants={"BLOCK_M": 64, "BLOCK_N": 64})
    # kernel = triton.compile(test_dequant_kernel)
    kernel = test_dequant_kernel[grid](A, uint64_tensor, output, n, k, int4_k,  
                            stride_ak)
    
    
    return kernel



def test_dequant(A: torch.Tensor, vector: torch.Tensor, output, ptx = 0):
    n, _ = A.shape
    k = vector.shape[1] # [m, k ]
    device = A.device
    stride_ak, stride_an = A.stride()
    stride_cm, stride_cn = output.stride()
    assert vector.shape[1] == A.shape[1] * 8, "Vector and input tensor shape mismatch"
    assert A.device == device and vector.device == device and output.device == device, "Tensors must be on CUDA"
    grid = lambda meta: (n, 1)
    
    storage = vector.untyped_storage()
    # k = vector.numel()  # 元素数量
    uint64_tensor = torch.tensor([], dtype=torch.uint64,device=device).set_(storage, 0, (k // 4,))
    int4_k = int(A.shape[1])
    
    if out_dim > 4096:
      evict = 'evict_first'
    else:
      evict = 'evict_last'
    
    # ret = triton.compile(kernel, signature="*fp32,i32,*fp32,i32", constants={"BLOCK_M": 64, "BLOCK_N": 64})
    # kernel = triton.compile(test_dequant_kernel)
    kernel = test_dequant_kernel[grid](A, uint64_tensor, output, n, k, int4_k,  
                            stride_ak)
    
    
    return kernel


def run_kernel(A: torch.Tensor, vector: torch.Tensor, output, kernel):
    n, _ = A.shape
    k = vector.shape[1] # [m, k ]
    device = A.device
    stride_ak, stride_an = A.stride()
   
    storage = vector.untyped_storage()
    uint64_tensor = torch.tensor([], dtype=torch.uint64,device=device).set_(storage, 0, (k // 4,))
    int4_k = int(A.shape[1])    
    
 
    kernel[n, 1, 1](A, uint64_tensor, output, n, k, int4_k,  
                            stride_ak)
    

# ret = compile_kernel()
# print(ret)
lib = load_cuda_ops(
  name="test",
  sources=code,
  func_names=["warp_specialized_gemv_host"],
  func_params=["t_t_t_t"],
  arches=[ "89", "90a"],
  extra_include_paths=["3rd/cutlass/include"],
  build_directory="./build",
)

device = torch.cuda.current_device()


import argparse
parser = argparse.ArgumentParser(description='Calculate volume of a cylinder')
# 添加参数
parser.add_argument('--marlin', type=int, default=0)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--triton', type=int, default=0)
parser.add_argument('--bitblas', type=int, default=0)
parser.add_argument('--gemlite', type=int, default=0)
parser.add_argument('--micro', type=int, default=0)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--tilelang', type=int, default=0)


# 解析参数
args = parser.parse_args()


if args.bitblas == 1:
  import bitblas

from common.common import generate_randint, gen_quant4, gen_quant4_my_no_reorder
from common.common import generate_randint, gen_quant4, gen_quant4_my

if args.tilelang == 1:
  import tilelang
  from tilelang import language as T
  from typing import Optional, Callable, Any
  import torch
  from tilelang import DataType
  from tilelang.quantize import (
      _tir_packed_int_to_int_convert,)


  def dequantize_gemv(
      M: int,
      N: int,
      K: int,
      in_dtype: str,
      out_dtype: str,
      accum_dtype: str,
      num_bits: int = 4,
      storage_dtype: str = "int8",
      source_format: str = "uint",
      n_partition: int = 4,
      reduce_thread: int = 32,
      fast_decoding: bool = False,
      trans_A: bool = False,
      trans_B: bool = True,
      group_size: int = -1,
      with_scaling: bool = False,
  ) -> Callable[..., Any]:

      assert n_partition is not None, "n_partition must be provided"
      assert reduce_thread is not None, (
          "reduce_thread must be provided currently, as related bitblas.gpu.gemv.GEMV"
          "sch_outer_reduction_with_config is not implemented")

      assert trans_A is False, "Dequantize only implement for trans_A=False currently"
      assert trans_B is True, "Dequantize only implement for trans_B=TRue currently"
      storage_type = "".join(c for c in storage_dtype if not c.isdigit())
      storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))
      num_elems_per_byte = storage_nbit // num_bits

      MAX_TRANSACTION_SIZE_IN_BITS = 128
      micro_size_k = MAX_TRANSACTION_SIZE_IN_BITS // DataType(in_dtype).bits
      micro_size_k_compressed = micro_size_k // num_elems_per_byte
      block_K = reduce_thread * micro_size_k

      if group_size == -1:
          group_size = K

      A_shape = (M, K)
      B_shape = (N, K // storage_nbit * num_bits)
      C_shape = (M, N)

      dp4a_size = 4
      use_dp4a = in_dtype == "int8" and accum_dtype == "int32"

      import_source: Optional[str] = None
      func_name: str = ""
      if fast_decoding is True:
          # Lazy import to decrease the startup time
          # as intrin registry may take a while to load
          from tilelang.quantize import get_lop3_intrin_group

          lop3_intrin_info = get_lop3_intrin_group(
              out_dtype=in_dtype,
              source_format=source_format,
              source_bit=num_bits,
              storage_dtype=storage_dtype,
              with_scaling=with_scaling,
              with_zeros=False,
          )
          import_source = lop3_intrin_info["c_source"]
          func_name = lop3_intrin_info["func_name"]
          assert import_source is not None, "lop3_intrin_info is not found"
          assert func_name is not None, "lop3_intrin_info is not found"
          import_source = import_source

      @T.prim_func
      def main(
          A: T.Tensor[A_shape, in_dtype],
          B: T.Tensor[B_shape, storage_dtype],
          C: T.Tensor[C_shape, out_dtype],
      ):
          with T.Kernel(
                  T.ceildiv(N, n_partition),
                  M,
                  threads=(reduce_thread, n_partition),
          ) as (
                  bx,
                  by,
          ):
              A_local = T.alloc_local((micro_size_k,), in_dtype)
              B_quant_local = T.alloc_local([micro_size_k_compressed], storage_dtype)
              B_dequantize_local = T.alloc_local([micro_size_k], in_dtype)
              accum_res = T.alloc_local((1,), accum_dtype)
              reduced_accum_res = T.alloc_local((1,), accum_dtype)

              kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
              ni = T.thread_binding(0, n_partition, thread="threadIdx.y")

              T.import_source(import_source)

              T.clear(accum_res)
              for ko in T.serial(T.ceildiv(K, block_K)):
                  for v in T.vectorized(micro_size_k):
                      A_local[v] = A[by, ko * block_K + kr * micro_size_k + v]

                  for v in T.vectorized(micro_size_k_compressed):
                      B_quant_local[v] = B[
                          bx * n_partition + ni,
                          ko * (reduce_thread * micro_size_k_compressed) +
                          kr * micro_size_k_compressed + v,
                      ]

                  if fast_decoding:
                      T.call_extern(
                          func_name,
                          T.address_of(B_quant_local[0]),
                          T.address_of(B_dequantize_local[0]),
                          dtype=in_dtype,
                      )
                  else:
                      for ki in T.serial(micro_size_k):
                          B_dequantize_local[ki] = _tir_packed_int_to_int_convert(
                              storage_type,
                              storage_nbit)(num_bits, B_quant_local[ki // num_elems_per_byte],
                                            ki % num_elems_per_byte, in_dtype)

                  if use_dp4a:
                      for ki in T.serial(micro_size_k // dp4a_size):
                          T.dp4a(
                              A_local[ki * dp4a_size],
                              B_dequantize_local[ki * dp4a_size],
                              accum_res[0],
                          )
                  else:
                      for ki in T.serial(micro_size_k):
                          accum_res[0] += A_local[ki] * B_dequantize_local[ki]

              with T.attr(
                      T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),
                      "reduce_scope",
                      T.reinterpret(T.uint64(0), dtype="handle"),
              ):
                  T.evaluate(
                      T.tvm_thread_allreduce(
                          T.uint32(1),
                          accum_res[0],
                          True,
                          reduced_accum_res[0],
                          kr,
                          dtype="handle",
                      ))
              if kr == 0:
                  C[by, bx * n_partition + ni] = reduced_accum_res[0]

      return main

for (out_dim, k) in [ (4096, 4096), (2048, 4096), (4096, 8192), (12288, 4096), (8192, 8192) ]:
# for (out_dim, k) in [  (8192, 8192) ]:
# for (out_dim, k) in [  (4096, 8192)]:
  dtype = torch.float16


  c = torch.zeros((1, out_dim), dtype=dtype, device=device)

  weight, vector = generate_randint(k, out_dim, device)


  #-------------------------------------marlin----------------------------
  import marlin
  workspace = torch.zeros(out_dim // 128 * 16, device=device)
  C_i4mar = torch.zeros((1, out_dim), dtype=dtype, device=device)
  thread_k = 64
  thread_n = 256
  _, B, s = gen_quant4(k, out_dim, weight.t().contiguous(),   groupsize=-1)  

  marlin.mul(vector, B, C_i4mar, s, workspace, thread_k, thread_n, -1)


  torch.cuda.synchronize()

  #------------------------------------marlin------------------------------
  q_weight, scales  = gen_quant4_my(out_dim, k, torch.clone(weight),   groupsize = -1, tile = 1)
  scales = scales.to(torch.float32)
  
  lib.warp_specialized_gemv_host(q_weight, vector, c, scales)

  #------------------------------------mixq---------------------------------
  c_triton = torch.zeros((1, out_dim), dtype=dtype, device=device)


  q_weight_triton, scales_trion  = gen_quant4_my_no_reorder(out_dim, k, torch.clone(weight),   groupsize = -1, tile = 1)
  scales_trion = scales_trion.to(torch.float32)
  gemv_int4(q_weight_triton, vector, c_triton)

  # print(torch.mm(vector, weight.t()))
  c_triton_2 = torch.zeros((1, out_dim), dtype=dtype, device=device)

  
  kernel = test_dequant(q_weight, vector, c_triton_2)
  # print(c_triton_2)
  from pathlib import Path
  tmp = Path("/home/chenyidong/newstart/bandwidth/qmoe")


  filename = "test.ptx"
  temp_file = (tmp / filename)
  # print(kernel.asm.keys())
  # exit()
  
  temp_file.write_text(kernel.asm['ptx'])


  filename = "test.ttgir"
  temp_file = (tmp / filename)
  # print(kernel.asm.keys())
  # exit()
  
  temp_file.write_text(kernel.asm['ttgir'])
  kernel = triton.compile(str(temp_file))
  # print(kernel)
  c_triton_2 = torch.zeros((1, out_dim), dtype=dtype, device=device)
  run_kernel(q_weight, vector, c_triton_2, kernel)
  # print(c_triton_2)
 
  # print(c_triton)
  # print(c_triton_2)
  # print(c)
  # exit()
  #------------------------------------mixq----------------------------------
  torch.testing.assert_close(c, C_i4mar, rtol=1e-2, atol=1e-2)
  torch.testing.assert_close(c, c_triton * scales_trion.T.to(torch.float16), rtol=1e-2, atol=1e-2)
  
  if not args.debug:
    torch.testing.assert_close(c, c_triton_2 * scales.T.to(torch.float16), rtol=1e-2, atol=1e-2)

  # exit()
  
  torch.cuda.synchronize()

  if args.bitblas == 1:
    matmul_config = bitblas.MatmulConfig(
      M=1,  # M dimension
      N=out_dim,  # N dimension
      K=k,  # K dimension
      A_dtype="float16",  # activation A dtype
      W_dtype="int4",  # weight W dtype
      accum_dtype="float16",  # accumulation dtype
      out_dtype="float16",  # output dtype
      layout="nt",  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
      with_bias=False,  # bias
      group_size=-1,  # setting for grouped quantization
      with_scaling=False,  # setting for scaling factor
      with_zeros=False,  # setting for zeros
      zeros_mode=None,  # setting for how to calculating zeros
    )
    matmul = bitblas.Matmul(config=matmul_config, enable_tuning = False)



    input_shape = (1, k)
    weight_shape = (out_dim, k)
    scaling_shape = (out_dim, k // 128)
    zeros_shape = (out_dim, k // 128)
    output_shape = (1, out_dim)
    scaling = torch.rand(scaling_shape, dtype=torch.float16).cuda()
    zeros = torch.rand(zeros_shape, dtype=torch.float16).cuda()

    # Create input tensor
    input_tensor = torch.rand(input_shape, dtype=torch.float16).cuda()

    # Create and transform weight tensor
    weight_tensor = torch.randint(0, 7, weight_shape, dtype=torch.int8).cuda()
    weight_tensor_int4 = matmul.transform_weight(weight_tensor)

    output_tensor = matmul(input_tensor, weight_tensor_int4, scale=scaling, zeros=zeros)


  if args.gemlite == 1:
    from gemlite.helper import *
    in_features = k
    out_features = out_dim
    device, dtype = 'cuda:0', torch.float16
    group_size = 128
    linear = torch.nn.Linear(in_features, out_features, bias=False, device=None, dtype=torch.float16).cuda()

    from gemlite.helper import GemLiteLinearTriton, DType
    from hqq.core.quantize import HQQLinear, BaseQuantizeConfig

    linear.weight.data =  weight
    orig_shape   = (out_features, in_features)
    quant_config   = BaseQuantizeConfig(nbits=4, group_size=group_size)

    # print(quant_config)
    hqq_layer    = HQQLinear(linear, quant_config=quant_config,
                              compute_dtype=torch.float16, device=device, 
                              del_orig=False) 


    gemlite_linear = GemLiteLinearTriton(W_nbits=4, 
                                        group_size=group_size, in_features=in_features, out_features=out_features, 
                                        input_dtype=DType.FP16, output_dtype=DType.FP16)

    # print(hqq_layer.meta['scale'])
    # exit()
    gemlite_linear.pack(hqq_layer.unpack(dtype=torch.uint8).view(orig_shape),
                        hqq_layer.meta['scale'].clone(), 
                        hqq_layer.meta['zero'].clone(), bias=None)
    
    matmul_type = "GEMV_REVSPLITK"
    # matmul_type = "GEMM"
    # matmul_type = "GEMV_SPLITK"
    for i in range(500):
      output = gemlite_linear.forward_manual(vector, matmul_type=matmul_type)


  if args.tilelang == 1:
        M = 1
        N = out_dim
        K = k
        in_dtype = "float16"
        out_dtype = "float16"
        accum_dtype = "float16"
        num_bits = 4
        storage_dtype = "int8"
        source_format = "uint"
        n_partition = 4
        reduce_thread = 32
        fast_decoding = True
        trans_A = False
        trans_B = True
        group_size = -1
        with_scaling = False

        program = dequantize_gemv(M, N, K, in_dtype, out_dtype, accum_dtype, num_bits, storage_dtype,
                                  source_format, n_partition, reduce_thread, fast_decoding, trans_A,
                                  trans_B, group_size, with_scaling)

        kernel = tilelang.compile(program)

        # print(kernel.get_kernel_source())
        # exit()

        storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))
        num_elems_per_byte = storage_nbit // num_bits
        # A = torch.rand(M, K, dtype=getattr(torch, in_dtype)).cuda()
        # qB = torch.randint(
        #     0, 127, (N, K // num_elems_per_byte), dtype=getattr(torch, storage_dtype)).cuda()
        C = torch.zeros(M, N, dtype=getattr(torch, accum_dtype)).cuda()

 
        if fast_decoding:
            from tilelang.quantize.utils import interleave_weight
            qB = interleave_weight(q_weight, num_bits, in_dtype)
        kernel(vector, qB, C)

  with torch.cuda.stream(torch.cuda.Stream()):
    if args.marlin == 1:
      ms = do_bench(lambda: marlin.mul(vector, B, C_i4mar, s, workspace, thread_k, thread_n, -1))
    if args.cuda == 1:
      ms = do_bench(lambda: lib.warp_specialized_gemv_host(q_weight, vector, c, scales))
    if args.triton == 1:
      ms = do_bench(lambda: gemv_int4(q_weight, vector, c_triton))
    if args.bitblas == 1:
      ms = do_bench(lambda: matmul(input_tensor, weight_tensor_int4, scale=scaling, zeros=zeros))
    if args.gemlite == 1:

      ms = do_bench_cudagraph(lambda: gemlite_linear.forward_manual(vector, matmul_type=matmul_type))

    if args.micro == 1:

      ms = do_bench(lambda: run_kernel(q_weight, vector, c_triton_2, kernel))

    if args.tilelang == 1:
      ms = do_bench(lambda: kernel(vector, qB, c_triton))
      
    # ms = do_bench(lambda: torch.mm(vector, weight.T))


  
  # double gflops = (2.0 * M * N) / (kernel_time * 1e9);  // 每个元素需要1次乘法和1次加法
  # double bandwidth_gb_s = ((M * N + M + N) * sizeof(half) * 1e-9) / kernel_time;
  gb = (out_dim * k + out_dim + k) * 2 / 1024 / 1024 / 1024
  bandwidth_gb_s = (gb) / ms * 1000
  print(f"{bandwidth_gb_s=} GB/s, {gb} GB, {ms} ms")