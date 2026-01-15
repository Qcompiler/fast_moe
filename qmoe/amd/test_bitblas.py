
from triton.testing import do_bench 

import torch
import numpy as np
import torch.nn as nn
seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)



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



from common.common import generate_randint, gen_quant4, gen_quant4_my
for (out_dim, k) in [ (4096, 4096), (2048, 4096), (4096, 8192), (12288, 4096), (8192, 8192) ]:
  dtype = torch.float16


  c = torch.zeros((1, out_dim), dtype=dtype, device=device)

  weight, vector = generate_randint(k, out_dim, device)



  torch.cuda.synchronize()

  if args.bitblas == 1:
    from hqq.backends.bitblas import patch_hqq_to_bitblas, HQQLinearBitBlas
    HQQLinearBitBlas.check = lambda hqq_layer: True
    HQQLinearBitBlas.BIT_TO_DTYPE = {8:"uint8", 4: "uint4", 2: "uint2", 1: "uint1"}
    from gemlite.helper import *
    from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
    group_size = k
    quant_config   = BaseQuantizeConfig(nbits=4, group_size=group_size)
    in_features = k
    out_features = out_dim
    linear = torch.nn.Linear(in_features, out_features, bias=False, device=None, dtype=torch.float16).cuda()

    linear.weight.data =  weight
    bitblas_linear = patch_hqq_to_bitblas(HQQLinear(linear, quant_config=quant_config, compute_dtype=torch.float16, device=device, del_orig=False), None)
    output = bitblas_linear(vector)
    # torch.testing.assert_close(c, output, rtol=1e-2, atol=1e-2)
    
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
    
    # matmul_type = "GEMV_REVSPLITK"
    matmul_type = "GEMV"
    # matmul_type = "GEMV_SPLITK"
    # for i in range(500):
    #   output = gemlite_linear.forward_manual(vector, matmul_type=matmul_type)


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

        # kernel = tilelang.compile(program, verbose = True)
        kernel = tilelang.compile(program)


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
        ms = do_bench(lambda: marlin.mul(vector, B, C_i4mar, s, workspace, thread_k, thread_n, -1), 
                      warmup=200, rep=1000)
    if args.cuda == 1:
        ms = do_bench(lambda: lib.warp_specialized_gemv_host(q_weight, vector, c, scales), warmup=200, rep=1000)
    if args.triton == 1:
        ms = do_bench(lambda: gemv_int4(q_weight, vector, c_triton), warmup=200, rep=1000)
    if args.bitblas == 1:
        ms = do_bench(lambda: bitblas_linear(vector), warmup=200, rep=1000)
    if args.gemlite == 1:
      # gemlite_linear.forward = lambda x: gemlite_linear.forward_manual(x, matmul_type="GEMV_REVSPLITK")
      # ms  = do_bench(lambda x: gemlite_linear(vector), {'x': vector.to(gemlite_linear.compute_dtype)}) 
        ms = do_bench(lambda: gemlite_linear.forward_manual(vector, matmul_type="GEMV_REVSPLITK"))

    if args.micro == 1:
      # print("call test dequant")
        # ms = do_bench(lambda: test_dequant(q_weight, vector, c_triton_2), warmup=200, rep=200)
        ms = do_bench(lambda: run_kernel(q_weight, vector, c_triton_2, kernel), warmup=200, rep=1000)

    if args.tilelang == 1:
        ms = do_bench(lambda: kernel(vector, qB, c_triton), warmup=200, rep=1000)
      
 
 
  gb = (out_dim * k + out_dim + k) * 2 / 1024 / 1024 / 1024
  bandwidth_gb_s = (gb) / ms * 1000
  print(f"{bandwidth_gb_s=} GB/s, {gb} GB, {ms} ms")