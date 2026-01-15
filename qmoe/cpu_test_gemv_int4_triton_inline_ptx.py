import torch
from jitcu import load_cuda_ops
from triton.testing import do_bench, do_bench_cudagraph


import torch
import numpy as np
import torch.nn as nn
seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)

from qmoe.common.common import  import_code
import torch, math, random, copy
from torch import Tensor
import triton
import triton.language as tl
import time

 

def do_bench_cpu(fn, warmup=5, rep=20):
    assert rep > 0
    for _ in range(warmup):
        fn()
    durations = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        durations.append((t1 - t0) * 1000)
    # return the median time
    sorted_durations = sorted(durations)
    if rep % 2 == 0:
        return (sorted_durations[rep // 2 - 1] + sorted_durations[rep // 2]) / 2
    else:
        return sorted_durations[rep // 2]


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
    x1, x2, x3, x4 = tl.inline_asm_elementwise(
        asm="""
            {
            }
        """,
        constraints=(
            "=r,=r,=r,=r,r"
        ),
        args=[b], #输入
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32), #输出
        is_pure=False,
        pack=1,
    )
    return x1, x2, x3, x4


@triton.jit
def sum_2_half(x1, vec1):
    # x1 : int32 x2: int32
    # vec1 : int32 vec2: int32
    # x1 * vec1 
    y = tl.inline_asm_elementwise(
        asm="""
            {
            }
        """,
        constraints=(
            "=f,r,r"
        ),
        args=[x1, vec1],  # 参数 1 2
        dtype=(tl.float16), #参数 0
        is_pure=False,
        pack=1,
    )

    return y

@triton.jit
def sum_4_half(x1, x2, vec1, vec2):
    # x1 : int32 x2: int32
    # vec1 : int32 vec2: int32
    # x1 * vec1 + x2 * vec2
    y = tl.inline_asm_elementwise(
        asm="""
            {
               
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
        asm=" ",
        constraints=("=r,=r,=r,=r,l"),
        args=[ptr],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=False,
        pack=1
    )



def get_autotune_config():
    configs = []
    for evict in ['evict_last', 'evict_first', None]:
        for evict_scales in ['evict_last', 'evict_first', None]:
            configs.append(triton.Config({ 'evict' : evict, 'evict_scales' : evict_scales}))

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
    x_ptr = x_ptr + (offs_k * 2)
    for kk in range(0, tl.cdiv(int4_k, BLOCK_SIZE)):
       
        x1, x2, x3, x4 = load_v4_b32(x_ptr)
        a = tl.load(A_ptr,  eviction_policy = evict)
        a1, a2, a3, a4 = dequanti_tensorRT_llm(a) 
        all1 += sum_4_half(a1, a2, x1, x2)
        all1 += sum_4_half(a3, a4, x3, x4)
  
        offs_k +=  BLOCK_SIZE 
        A_ptr += (BLOCK_SIZE) 
        x_ptr += (BLOCK_SIZE * 2)

    acc = tl.sum(all1, axis=0) 
    scales = tl.load(scales_ptr + row_id, eviction_policy = evict_scales)
    acc = acc * scales
    tl.store(y_ptr + row_id, acc)


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
    
    kernel = test_dequant_kernel[grid](A, uint64_tensor, output, scales, n, k, int4_k,  
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
    



device = "hip"


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







for (out_dim, k) in [  (4096, 4096), (2048, 4096), (4096, 8192), (12288, 4096), (8192, 8192) ]:
    # for (out_dim, k) in [  (8192, 8192) ]:
    # for (out_dim, k) in [  (4096, 4096), (8192, 8192) , (8192, 57344), (28672, 8192)]:
    dtype = torch.float16


    c = torch.zeros((1, out_dim), dtype=dtype, device=device)

    weight, vector = generate_randint(k, out_dim, device)

    c = torch.mm(vector, weight.T)

    if args.torch == 1:
        ms = do_bench_cpu(lambda: torch.mm(vector, weight.T), warmup=500, rep=2000)
        gb = (out_dim * k + out_dim + k) * 2 / 1024 / 1024 / 1024
        bandwidth_gb_s = (gb) / ms * 1000
        print(f"{bandwidth_gb_s=} GB/s, {gb} GB, {ms} ms")
        continue 
    #------------------------------------marlin------------------------------
    q_weight, scales  = gen_quant4_my(out_dim, k, torch.clone(weight),   groupsize = -1, tile = 1)
    scales = scales.to(torch.float32).to(device)
    q_weight = q_weight.to(device)
    c_triton = torch.zeros((1, out_dim), dtype=dtype, device=device)
    q_weight_triton, scales_trion  = gen_quant4_my_no_reorder(out_dim, k, torch.clone(weight),   groupsize = -1, tile = 1)
    scales_trion = scales_trion.to(torch.float32)
    gemv_int4(q_weight_triton, vector, c_triton)

    # print(torch.mm(vector, weight.t()))
    c_triton_2 = torch.zeros((1, out_dim), dtype=dtype, device=device)

    if args.micro:
        scales = scales.to(torch.float16)
    # kernel = test_dequant(q_weight, vector, c_triton_2, scales)



    if not args.debug:
        torch.testing.assert_close(c, c_triton * scales_trion.T.to(torch.float16), rtol=1e-2, atol=1e-2)
        

        # torch.testing.assert_close(c, c_triton_2, rtol=1e-2, atol=1e-2)






    print("call bench!")
    if args.triton == 1:
        ms = do_bench_cpu(lambda: gemv_int4(q_weight, vector, c_triton), warmup=1, rep=1)

    if args.micro == 1:
        # print("call test dequant")
        ms = do_bench_cpu(lambda: test_dequant(q_weight, vector, c_triton_2, scales), warmup=500, rep=2000)

        
    gb = (out_dim * k + out_dim + k) * 2 / 1024 / 1024 / 1024
    bandwidth_gb_s = (gb) / ms * 1000
    print(f"{bandwidth_gb_s=} GB/s, {gb} GB, {ms} ms")