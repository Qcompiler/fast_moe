import torch
import numpy as np
import torch.nn as nn

seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)

from jitcu import load_cuda_ops
from qmoe.common.common import compute_moe_gate_up_opt,SiluAndMul,  generate_randint_moe,generate_randint_moe_down,  compute_moe_gate_up_down_opt, gen_quant4, gen_quant4_my, compute_moe_gate_up_down
from qmoe.common.common import  gen_quant4_my
from qmoe.common.common import  import_code
from triton.testing import do_bench_cudagraph, do_bench







device = torch.cuda.current_device()


import argparse
parser = argparse.ArgumentParser(description='Calculate volume of a cylinder')
# 添加参数
parser.add_argument('--torch', type=int, default=0)
parser.add_argument('--new', type=int, default=0)
parser.add_argument('--naive', type=int, default=0)
parser.add_argument('--kernel_type', type=int, default=0)
parser.add_argument('--quant', type=int, default=0)
parser.add_argument('--bf16',  action='store_true')

# 解析参数
args = parser.parse_args()


code_gate_name = "moe_gemv_gate.cu"
code_down_name = "moe_gemv_down.cu"
if args.bf16:
  code_gate_name = "generated/bf16_" + code_gate_name
  code_down_name = "generated/bf16_" + code_down_name


code_gate = import_code(code_gate_name)
code_down = import_code(code_down_name)

lib_gate = load_cuda_ops(
  name="gate",
  sources=code_gate,
  func_names=["warp_specialized_gemv_gate_host"],
  func_params=["t_t_t_t"],
  arches=[ "89", "90a"],
  extra_include_paths=["include"],
  build_directory="build",
)

lib_down = load_cuda_ops(
  name="down",
  sources=code_down,
  func_names=["warp_specialized_gemv_down_host"],
  func_params=["t_t_t_t"],
  arches=[ "89", "90a"],
  extra_include_paths=["include"],
  build_directory="./build",
)

if args.quant == 1:
      lib_quant = load_cuda_ops(
        name="i4_down",
        sources=import_code("generated/i4_moe_gemv_down.cu"),
        func_names=["warp_specialized_gemv_down_host"],
        func_params=["t_t_t_t"],
        arches=[ "89", "90a"],
        extra_include_paths=["include"],
        build_directory="./build",
      )
for (out_dim, k) in [ (512,  2048),   (2048, 2048) , (2048, 4096), (4096, 4096), (4096, 8192) ]:


  num_experts = 16
  topk = 8

  if args.bf16:
    dtype = torch.bfloat16
  else:
    dtype = torch.float16  


  gate_up_weight, vector, topk_ids = generate_randint_moe(num_experts, out_dim,  k, topk,  device, dtype)

  down_weight, topk_weight = generate_randint_moe_down(num_experts, out_dim,  k, topk,  device, dtype)


  hidden = compute_moe_gate_up_down(vector, topk_ids, gate_up_weight, down_weight, topk_weight)
  

  topk_ids = topk_ids.to(torch.int32)
  hidden2 = compute_moe_gate_up_down_opt(vector, topk_ids, gate_up_weight, down_weight, topk_weight)


  # ----------------计算 mix---------------- 
  kernel_type = args.kernel_type
  tmp = compute_moe_gate_up_opt(vector, topk_ids, gate_up_weight)
  out = torch.zeros_like(tmp)
  lib_gate.warp_specialized_gemv_gate_host( gate_up_weight, vector, out, topk_ids, 1)
  act_fn = SiluAndMul()
  act = act_fn(out)
  hidden3 = torch.zeros_like(hidden2)
  lib_down.warp_specialized_gemv_down_host( down_weight,  act,  hidden3, topk_weight,  topk_ids , kernel_type)

  import moe_gemm
  final_hidden_states_ = torch.zeros_like(hidden2)
  moe_gemm.moe_gemv_down(act,
                final_hidden_states_,
                down_weight,
                topk_weight,
                topk_ids,
                32, 
                4 )


  torch.testing.assert_close(final_hidden_states_, hidden, rtol=1e-1, atol=5e-1)
  torch.testing.assert_close(hidden2, hidden, rtol=1e-2, atol=5e-1)


  torch.testing.assert_close(hidden3, hidden,  rtol=1e-1, atol=5e-1)
  # print(final_hidden_states_)
  # print(hidden)
  # print(hidden3)
  if args.quant == 1:

      group_size = 128
      

      n = down_weight.shape[1]
      k = down_weight.shape[2]

      # down_weight = (down_weight / down_weight) / 100
      all_weight = torch.empty((num_experts, n, k // 8), dtype=torch.int32 ,device = down_weight.device)
      all_scales = torch.empty((num_experts, n, k // group_size), dtype=torch.float,device = down_weight.device)
      for i in range(0,num_experts):
        
        
        q_weight, scales  = gen_quant4_my(n, k, 
                                          torch.clone(down_weight[i,:,:].squeeze()),   
                                          groupsize = group_size, tile = 1)

        scales = scales.to(torch.float32)
        all_weight[i, :, :] = q_weight

        all_scales[i, :, :] = scales
      hidden4 = torch.zeros_like(hidden3)

      act = act_fn(out)
      lib_quant.warp_specialized_gemv_down_host(all_weight,  act,  hidden4, 
                                                topk_weight,  topk_ids , 
                                                all_scales, group_size, kernel_type)
      

      final_hidden_states_ = torch.zeros_like(hidden)
      moe_gemm.moe_gemv_down(act,
                final_hidden_states_,
                down_weight,
                topk_weight,
                topk_ids,
                32, 
                4 )
      
      print(final_hidden_states_)
      print(hidden4)

      torch.testing.assert_close(hidden4.to(torch.float16), final_hidden_states_.to(torch.float16), rtol=1e2, atol=10)



      # exit()

  torch.cuda.synchronize()
  with torch.cuda.stream(torch.cuda.Stream()):

    if args.quant == 1:
       ms = do_bench(lambda:  lib_quant.warp_specialized_gemv_down_host(all_weight,  act,  hidden4, 
                                                topk_weight,  topk_ids , 
                                                all_scales, group_size, kernel_type))

    else:
      if args.naive == 1:
        ms = do_bench(lambda: moe_gemm.moe_gemv_down(act, final_hidden_states_, down_weight,
                  topk_weight,
                  topk_ids, 32, 4 ))
        
      if args.new == 1:
        ms = do_bench(lambda:  lib_down.warp_specialized_gemv_down_host( down_weight,  
                                                                        act,  hidden3, topk_weight,  
                                                                        topk_ids, kernel_type))


  

  intermediate_size = out_dim * 2
  gb = ( intermediate_size / 2 * topk # down 读入的tensor
        + intermediate_size / 2 * k * topk # down 读入的weight
        + k # down 写出的tensor
        ) * 2 / 1024 / 1024 / 1024
  bandwidth_gb_s = (gb) / ms * 1000
  print(f"{bandwidth_gb_s=} GB/s, {gb} GB, {ms} ms")


  # intermediate_size = out_dim * 2
  # gb = (intermediate_size * k * topk  # 读入的专家
  #       + intermediate_size * topk  # 写出的tensor
  #       + k # 读入的tensor
  #       + intermediate_size * topk  + intermediate_size / 2 * topk  # act 读入和写出的tensor
  #       + intermediate_size / 2 * topk # down 读入的tensor
  #       + intermediate_size / 2 * k * topk # down 读入的weight
  #       + k # down 写出的tensor
  #       ) * 2 / 1024 / 1024 / 1024
  # bandwidth_gb_s = (gb) / ms * 1000
  # print(f"{bandwidth_gb_s=} GB/s, {gb} GB, {ms} ms")