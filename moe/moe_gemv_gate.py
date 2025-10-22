import torch
from jitcu import load_cuda_ops

from common.common import  import_code
from triton.testing import do_bench_cudagraph,do_bench




device = torch.cuda.current_device()


import argparse
parser = argparse.ArgumentParser(description='Calculate volume of a cylinder')
# 添加参数
parser.add_argument('--torch', type=int, default=0)
parser.add_argument('--new', type=int, default=0)
parser.add_argument('--naive', type=int, default=0)
parser.add_argument('--quant', type=int, default=0)
parser.add_argument('--kernel_type', type=int, default=0)
parser.add_argument('--bf16',  action='store_true')

# 解析参数
args = parser.parse_args()

kernel_type = args.kernel_type


if args.quant == 1:

  if args.bf16:
    code = "generated/i4_bf16_moe_gemv_gate.cu"

  else:
    code = "generated/i4_moe_gemv_gate.cu"

  lib_quant = load_cuda_ops(
    name="i4_gate",
    sources=import_code(code),
    func_names=["warp_specialized_gemv_gate_host"],
    func_params=["t_t_t_t"],
    arches=[ "89", "90a"],
    extra_include_paths=["include"],
    build_directory="./build",
  )

code_name = "moe_gemv_gate.cu"
if args.bf16:
  code_name = "generated/bf16_" + code_name

  
  
code = import_code(code_name)


lib = load_cuda_ops(
  name="test",
  sources=code,
  func_names=["warp_specialized_gemv_gate_host"],
  func_params=["t_t_t_t"],
  arches=[ "89", "90a"],
  extra_include_paths=["include"],
  build_directory="./build",
)
from common.common import generate_randint_moe, compute_moe_gate_up_opt, gen_quant4, gen_quant4_my, compute_moe_gate_up
for (out_dim, k) in [(512,  2048),   (2048, 2048) , (2048, 4096), (4096, 4096), (4096, 8192) ]:

  num_experts = 16
  topk = 8

  if args.bf16:
    dtype = torch.bfloat16
  else:
    dtype = torch.float16


  gate_up_weight, vector, topk_ids = generate_randint_moe(num_experts, out_dim,  k, topk,  device, dtype)


  hidden = compute_moe_gate_up(vector, topk_ids, gate_up_weight)
  

  topk_ids = topk_ids.to(torch.int32)
  

  hidden2 = compute_moe_gate_up_opt(vector, topk_ids, gate_up_weight)

  hidden3 = torch.zeros_like(hidden2)

  lib.warp_specialized_gemv_gate_host( gate_up_weight, vector, hidden3, topk_ids, kernel_type)



  if args.quant == 1:
      group_size = 128
      from common.common import  gen_quant4_my

      n = gate_up_weight.shape[1]
      k = gate_up_weight.shape[2]
      all_weight = torch.empty((num_experts, n, k // 8), dtype=torch.int32 ,device = gate_up_weight.device)
      all_scales = torch.empty((num_experts, n, k // group_size), dtype=torch.float,device = gate_up_weight.device)
      for i in range(0,num_experts):
        
        
        q_weight, scales  = gen_quant4_my(n, k, 
                                          torch.clone(gate_up_weight[i,:,:].squeeze()),   
                                          groupsize = group_size, tile = 1)

        scales = scales.to(torch.float32)
        all_weight[i, :, :] = q_weight
        all_scales[i, :, :] = scales

        # print(scales)
        # exit()
      hidden4 = torch.zeros_like(hidden3)
      lib_quant.warp_specialized_gemv_gate_host(all_weight,  vector,  hidden4, 
                                                topk_ids , 
                                                all_scales, group_size, kernel_type)
      
      # print(all_scales)
      # print(hidden4)
      # print(hidden)

      torch.testing.assert_close((hidden+0.01).to(torch.float), (hidden4+0.01).to(torch.float), rtol = 0.1, atol=100)


  torch.testing.assert_close(hidden2, hidden, rtol=1e-2, atol=1e-2)
  torch.testing.assert_close(hidden3, hidden, rtol=1e-2, atol=1e-2)
  for i in range(1000):
    lib.warp_specialized_gemv_gate_host( gate_up_weight, vector, hidden3, topk_ids, kernel_type)

  torch.cuda.synchronize()
  with torch.cuda.stream(torch.cuda.Stream()):

    if args.quant == 1:
       ms = do_bench(lambda: lib_quant.warp_specialized_gemv_gate_host(all_weight,  vector,  hidden4, 
                                                topk_ids , 
                                                all_scales, group_size, kernel_type))
       
    else:
      if args.torch == 1:
        ms = do_bench(lambda: compute_moe_gate_up(vector, topk_ids, gate_up_weight))
      if args.naive == 1:
        ms = do_bench(lambda: compute_moe_gate_up_opt(vector, topk_ids, gate_up_weight))
      if args.new == 1:
        ms = do_bench(lambda: lib.warp_specialized_gemv_gate_host( gate_up_weight, vector, hidden3, topk_ids, kernel_type))

      
  intermediate_size = out_dim * 2
  gb = (intermediate_size * k * topk  # 读入的专家
        + intermediate_size * topk  # 写出的tensor
        + k # 读入的tensor
        ) * 2 / 1024 / 1024 / 1024
  bandwidth_gb_s = (gb) / ms * 1000
  print(f"{bandwidth_gb_s=} GB/s, {gb} GB, {ms} ms")