import torch
import time
import numpy as np
from tabulate import tabulate

# 确保程序在H100上运行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("本程序需要在NVIDIA GPU上运行，且推荐使用H100")

# 检查是否是H100
gpu_name = torch.cuda.get_device_name(device)
if "H100" not in gpu_name:
    print(f"警告: 检测到的GPU是 {gpu_name}，不是H100，测试结果可能不准确")

# 定义要测试的矩阵形状 (M, N, K)
# 包含各种大小，从较小的矩阵到较大的矩阵
matrix_shapes = [
    (512, 8192, 8192),
    (1024, 8192, 8192),
    (2048, 8192, 8192),
    (4096, 8192, 8192),
    (8192, 8192, 8192),

]

# 预热GPU，确保准确测量
def warmup():
    a = torch.randn(1024, 1024,  device=device).to(torch.float8_e4m3fn)
    b = torch.randn(1024, 1024,  device=device).to(torch.float8_e4m3fn)
    s_a = torch.randn(1024, 1,  device=device).to(torch.float32)
    s_b = torch.randn(1, 1024,  device=device).to(torch.float32)
    for _ in range(10):
        torch._scaled_mm(a, b.T, scale_a=s_a, scale_b=s_b, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()

# 测试单次GEMM的时间
def test_gemm_time(M, N, K, dtype=torch.float8_e4m3fn, num_iterations=10):
    # 创建随机矩阵
    a = torch.randn(M, K,  device=device).to(dtype)
    b = torch.randn(K, N, device=device).to(dtype)
    s_a = torch.randn(M, 1,  device=device).to(torch.float32)
    s_b = torch.randn(1, N,  device=device).to(torch.float32)    
    # 预热
    torch._scaled_mm(a, b.T, scale_a=s_a, scale_b=s_b, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    
    # 计时
    start_time = time.time()
    for _ in range(num_iterations):
        c = torch._scaled_mm(a, b.T, scale_a=s_a, scale_b=s_b, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()  # 等待所有GPU操作完成
    end_time = time.time()
    
    # 计算平均时间(秒)
    avg_time = (end_time - start_time) / num_iterations
    
    # 计算FLOPS (2*M*N*K 操作)
    flops = 2 * M * N * K
    flops_per_sec = flops / avg_time
    
    return avg_time, flops_per_sec, c

def main():
    print(f"测试GPU: {gpu_name}")
    print(f"测试精度: FP8 (e4m3fn)")
    print(f"开始预热GPU...")
    warmup()
    print("预热完成，开始测试...\n")
    
    results = []
    
    for (M, N, K) in matrix_shapes:
        print(f"测试形状: ({M}, {N}, {K}) ... ", end="", flush=True)
        try:
            avg_time, flops_per_sec, _ = test_gemm_time(M, N, K)
            
            # 单位转换为更易读的格式
            if flops_per_sec < 1e9:
                flops_str = f"{flops_per_sec / 1e6:.2f} MFLOPS"
            elif flops_per_sec < 1e12:
                flops_str = f"{flops_per_sec / 1e9:.2f} GFLOPS"
            else:
                flops_str = f"{flops_per_sec / 1e12:.2f} TFLOPS"
            
            results.append([
                f"({M}, {N}, {K})",
                f"{avg_time:.6f} s",
                flops_str,
                f"{2*M*N*K / 1e9:.2f} G"
            ])
            print("完成")
        except Exception as e:
            print(f"失败: {str(e)}")
    
    # 打印结果表格
    print("\n测试结果汇总:")
    print(tabulate(results, headers=["矩阵形状 (M, N, K)", "平均时间", "性能", "操作数"], tablefmt="grid"))

if __name__ == "__main__":
    main()

