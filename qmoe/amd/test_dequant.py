import torch
from triton.testing import do_bench, do_bench_cudagraph


import torch
import numpy as np
import torch.nn as nn
seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)

import torch, math, random, copy
from torch import Tensor
import triton
import triton.language as tl
import time


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
def dequanti_tensorRT_llm(b):
  #x1 int32
  # 一共四个
  # 总共16字节
  # 8个half
    return  tl.inline_asm_elementwise(
        asm="""
                
            v_mov_b32_e32 $3, 0xf000f
            v_mov_b32_e32 $8, 0xf000f0
            v_mov_b32_e32 $6, 0x64006400
            v_bfi_b32 $3, $3, $9, $6
            v_mov_b32_e32 $7, 0xf000f0
            v_bfi_b32 $8, $8, $9, $6
            v_ashrrev_i32_e32 $9, 8, $9
            s_mov_b32 $4, 0xe408
            v_mov_b32_e32 $1, 0xf000f
            v_mov_b32_e32 $2, 0x64006400
            v_bfi_b32 $6, $1, $9, $6
            v_bfi_b32 $7, $7, $9, $2
            v_pk_add_f16 $0, $3, $4 op_sel_hi:[1,0]
            s_movk_i32 $5, 0x2c00
            v_mov_b32_e32 $3, 0xd480
            v_pk_fma_f16 $1, $8, $5, $3 op_sel_hi:[1,0,0]
            v_pk_add_f16 $2, $6, $4 op_sel_hi:[1,0]
            v_pk_fma_f16 $3, $7, $5, $3 op_sel_hi:[1,0,0]

        """,
        constraints=(
            "=v,=v,=v,=v,=s,=s,=v,=v,=v,v"
        ),
        args=[b], #输入 参数 最后
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32, tl.uint32, 
               tl.uint32, tl.uint32, tl.uint32, tl.uint32), #输出 #参数 0 1 2 3 
        is_pure=False,
        pack=1,
    )
    





 
@triton.jit
def test_dequant_kernel(
    A_ptr, x_ptr, y_ptr,
    scales_ptr,
    m, k, int4_k,
    stride_am, 
    BLOCK_SIZE : tl.constexpr = 256,
    evict : tl.constexpr = 'evict_first',
    evict_scales : tl.constexpr = None,
    new : tl.constexpr = True
):



    row_id = tl.program_id(0)
    acc = 0
    acc = acc.to(tl.float16)
    all1 =   tl.zeros([BLOCK_SIZE], dtype=tl.float16)



    offs_k =  tl.arange(0, BLOCK_SIZE)
    A_offset = row_id * stride_am + (offs_k)

    A_ptr = A_ptr + A_offset

    y_ptr = y_ptr + (A_offset) * 4
    for kk in range(0,  tl.cdiv(int4_k, BLOCK_SIZE)):
       
 

        a = tl.load(A_ptr)
        x1, x2, x3, x4, x5, x6, x7, x8, x9 = dequanti_tensorRT_llm(a) 
     

         
        tl.store(y_ptr, x1)
        tl.store(y_ptr + 1, x2)
        tl.store(y_ptr + 2, x3)
        tl.store(y_ptr + 3, x4)

        A_ptr += (BLOCK_SIZE ) 
        y_ptr += (BLOCK_SIZE * 4)
 
        


def test_dequant(A: torch.Tensor, vector: torch.Tensor, output, scales):
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


    storage2 = output.untyped_storage()
    # k = vector.numel()  # 元素数量
    uint32_tensor = torch.tensor([], dtype=torch.int32,device=device).set_(storage2, 0, (1, k // 2,))
     
    kernel = test_dequant_kernel[grid](A, uint64_tensor, uint32_tensor, scales, n, k, int4_k,  
                            stride_ak)
    

    return kernel






def run_kernel(A: torch.Tensor, vector: torch.Tensor, output, kernel, scales):
    n, _ = A.shape
    k = vector.shape[1] # [m, k ]
    device = A.device
    stride_ak, stride_an = A.stride()
   
    storage = vector.untyped_storage()
    uint64_tensor = torch.tensor([], dtype=torch.uint64,device=device).set_(storage, 0, (k // 4,))
    int4_k = int(A.shape[1])    
    
 
    kernel[n, 1, 1](A, uint64_tensor, output, scales, n, k, int4_k,  
                            stride_ak, stride_an)
    



device  = triton.runtime.driver.active.get_active_torch_device()
print("hip version")
print(torch.version.hip)

import argparse
parser = argparse.ArgumentParser(description='Calculate volume of a cylinder')
# 添加参数
parser.add_argument('--triton', type=int, default=0)
parser.add_argument('--micro', type=int, default=0)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--torch', type=int, default=0)


# 解析参数
args = parser.parse_args()


from common.common import generate_randint, gen_quant4, gen_quant4_my_no_reorder
from common.common import generate_randint, gen_quant4, gen_quant4_my







# for (out_dim, k) in [  (4096, 4096) ]:
    # for (out_dim, k) in [  (8192, 8192) ]:
for (out_dim, k) in [  (4096, 4096), (8192, 8192) , (8192, 57344), (28672, 8192)]:
    dtype = torch.float16


    c = torch.zeros((1, out_dim), dtype=dtype, device=device)

    weight, vector = generate_randint(k, out_dim, device)

    c = torch.mm(vector, weight.T)


    #------------------------------------marlin------------------------------
    q_weight, scales  = gen_quant4_my(out_dim, k, torch.clone(weight),   groupsize = -1, tile = 1)
    scales = scales.to(torch.float32).to(device)
    q_weight = q_weight.to(device)
    c_triton = torch.zeros((1, out_dim), dtype=dtype, device=device)
    q_weight_triton, scales_trion  = gen_quant4_my_no_reorder(out_dim, k, torch.clone(weight),   groupsize = -1, tile = 1)
    scales_trion = scales_trion.to(torch.float32)


    # print(torch.mm(vector, weight.t()))
    c_triton_2 = torch.zeros((1, out_dim), dtype=dtype, device=device)

   
    scales = scales.to(torch.float16)
    c_triton_2 = torch.zeros_like(weight)
    kernel = test_dequant(q_weight, vector, c_triton_2, scales)

    torch.testing.assert_close(weight, c_triton_2, rtol=1e-2, atol=1e-2)



