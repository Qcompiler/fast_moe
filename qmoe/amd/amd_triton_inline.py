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
def sum_4_half(x1, x2, x3, x4):
    # 一共8个half
    y = tl.inline_asm_elementwise(
        asm="""

        v_pk_add_f16 $1, $1, $2    ; $1 = [sum_hi1, sum_lo1]
        v_pk_add_f16 $3, $3, $4    ; $3 = [sum_hi2, sum_lo2]
        v_pk_add_f16 $1, $1, $3    ; $1 = [sum_hi1+sum_hi2, sum_lo1+sum_lo2]

        ; 提取高位分量并转换为f32
        v_mov_b32_e32 $2, $1
        v_lshrrev_b32_e32 $1, 16, $2
        v_cvt_f32_f16_e32 $1, $1
        ; 提取低位分量并累加
        v_and_b32_e32 $2, 0xffff, $2  ; 低位分量
        v_cvt_f32_f16_e32 $2, $2
        v_add_f32_e32 $0, $1, $2   ; 高低分量相加

        """,
        constraints=(
            "=v,v,v,v,v"
        ),
        args=[x1, x2, x3, x4],  # 参数 1 2 3 4  
        dtype=(tl.float32), #参数 0 
        is_pure=False,
        pack=1,
    )

    return y







@triton.jit
def dequanti_tensorRT_llm(b):
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
    





def get_autotune_config():
    configs = []
    for evict in ['evict_last', 'evict_first', None]:
            for BLOCK_SIZE in [64, 128, 256, 512]:
                 configs.append(triton.Config({ 'BLOCK_SIZE' :
                                                BLOCK_SIZE, 
                                                'evict' : evict }))

    return configs

@triton.autotune(
    configs=get_autotune_config(),
    key = ['m', 'k'],
    use_cuda_graph = False
)
@triton.jit
def test_dequant_kernel(
    A_ptr, x_ptr, y_ptr,
    scales_ptr,
    m, k, int4_k,
    stride_am, 
    BLOCK_SIZE : tl.constexpr,
    evict : tl.constexpr
):



    row_id = tl.program_id(0)
    offs_k =  tl.arange(0, BLOCK_SIZE)
    A_offset = row_id * stride_am + (offs_k)
    A_ptr = A_ptr + A_offset


    all1 =   tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for kk in range(0,  tl.cdiv(int4_k, BLOCK_SIZE)):
       
 

        a = tl.load(A_ptr,  eviction_policy= evict)
        x1, x2, x3, x4, x5, x6, x7, x8, x9 = dequanti_tensorRT_llm(a) 
     
        all1 += sum_4_half(x1, x2, x3, x4)

        A_ptr += (BLOCK_SIZE ) 
 
    acc = tl.sum(all1, axis=0) 
    
    tl.store(y_ptr + row_id, acc.to(tl.float16))


def test_reduction(A: torch.Tensor, vector: torch.Tensor, output, scales):
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

 
    kernel = test_dequant_kernel[grid](A, uint64_tensor, output, scales, n, k, int4_k,  
                            stride_ak)
    
    # gcn = kernel.asm['amdgcn']
    # f = open("test_reduction.gcn", "w")
    # f.write(gcn)
    # f.close()

    return kernel








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
for (out_dim, k) in [  (4096, 4096), (8192, 8192) , (8192, 16384), (16384, 8192)]:
    dtype = torch.float16


    c = torch.zeros((1, out_dim), dtype=dtype, device=device)

    weight, vector = generate_randint(k, out_dim, device)

    # print(vector)

    vector = torch.ones((1, k), dtype=dtype, device=device)
    c = torch.mm(vector, weight.T)
    # print(c)

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
    kernel = test_reduction(q_weight, vector, c_triton_2, scales)

    # print(scales.T)
    # print(c_triton_2)
    # print(c_triton_2.shape)
    # print(c.shape)
    # print(weight)
    torch.testing.assert_close(c, c_triton_2, rtol=1e-2, atol=1e-2)

    if args.torch == 1:
        ms = do_bench_cudagraph(lambda: torch.mm(vector, weight.T),  rep=2000)

    if args.triton == 1:
        ms = do_bench_cudagraph(lambda: gemv_int4(q_weight, vector, c_triton),  rep=2000)

    if args.micro == 1:
        ms = do_bench_cudagraph(lambda: test_reduction(q_weight, vector, c_triton_2, scales),  rep=2000)

    gb = (out_dim * k + out_dim + k) * 2 / 1024 / 1024 / 1024
    bandwidth_gb_s = (gb) / ms * 1000
    print(f"{bandwidth_gb_s=} GB/s, {gb} GB, {ms} ms")