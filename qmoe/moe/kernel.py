

from qmoe.common.common import  import_code
from jitcu import load_cuda_ops
import os
import torch

from pathlib import Path

# 获取当前文件的绝对路径
current_dir_pathlib = Path(__file__).parent.resolve()
print(f"当前文件所在目录: {current_dir_pathlib}")
current_file_abs_path = os.path.join(current_dir_pathlib,"moe")
def gate_gate_lib(dtype):
  
  code_gate_name = "moe_gemv_gate.cu"
  if dtype == torch.bfloat16:
    code_gate_name = "generated/bf16_" + code_gate_name


  
  code_gate = import_code(os.path.join(current_file_abs_path,code_gate_name))

  include_path = os.path.join(current_file_abs_path,"include")
  lib_gate = load_cuda_ops(
    name="gate",
    sources=code_gate,
    func_names=["warp_specialized_gemv_gate_host"],
    func_params=["t_t_t_t"],
    arches=[ "89", "90a"],
    extra_include_paths=[include_path],
    build_directory="build",
  )
  return lib_gate

def gate_down_lib(dtype, quant):
  
  code_down_name = "moe_gemv_down.cu"
  if dtype == torch.bfloat16:
    code_down_name = "generated/bf16_" + code_down_name

  if quant == 1:
      code_down_name = "generated/i4_moe_gemv_down.cu"
  include_path = os.path.join(current_file_abs_path,"include")

  code_down_name = os.path.join(current_file_abs_path, code_down_name)
  code_down = import_code(code_down_name)


  lib_down = load_cuda_ops(
        name="down",
        sources=code_down,
        func_names=["warp_specialized_gemv_down_host"],
        func_params=["t_t_t_t"],
        arches=[ "89", "90a"],
        extra_include_paths=[include_path],
        build_directory="build",
    )

  return lib_down

