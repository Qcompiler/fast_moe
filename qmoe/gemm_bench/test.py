import torch
import time
import numpy as np

def measure_gemm_tflops(matrix_size, dtype=torch.float16, iterations=100):
    """
    测量FP16 GEMM的TFLOPS
    
    参数:
        matrix_size: 矩阵维度 (M, N, K)
        dtype: 数据类型
        iterations: 迭代次数，取平均值以获得更稳定的结果
        
    返回:
        tflops: 计算得到的TFLOPS
    """
    M, N, K = matrix_size
    
    # 在H100上创建FP16矩阵
    a = torch.randn(M, K, dtype=dtype, device='cuda')
    b = torch.randn(N, K, dtype=dtype, device='cuda')
    
    # 预热运行，确保CUDA内核已加载
    c = torch.matmul(a, b.T)
    torch.cuda.synchronize()  # 等待计算完成
    
    # 开始计时
    start_time = time.time()
    
    # 多次迭代以获得更准确的测量
    for _ in range(iterations):
        c = torch.matmul(a, b.T)
    
    # 等待所有计算完成
    torch.cuda.synchronize()
    end_time = time.time()
    
    # 计算总时间（秒）
    total_time = end_time - start_time
    
    # 计算总操作数：每个GEMM操作需要 2*M*N*K 次浮点运算
    total_ops = 2 * M * N * K * iterations
    
    # 转换为TFLOPS (1e12 次操作/秒)
    tflops = total_ops / (total_time * 1e12)
    
    return tflops

if __name__ == "__main__":
    # 检查是否有可用的CUDA设备
    if not torch.cuda.is_available():
        print("没有可用的CUDA设备")
        exit(1)
    
    # 检查是否是H100 GPU
    gpu_name = torch.cuda.get_device_name(0)
    print(f"使用的GPU: {gpu_name}")
    
    if "H100" not in gpu_name:
        print("警告: 此程序设计用于NVIDIA H100 GPU，可能无法在其他GPU上获得最佳结果")
    
    # 矩阵大小设置 - 可以根据需要调整
    # 较大的矩阵通常能获得更高的TFLOPS
    matrix_sizes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (16384, 16384, 16384),
    ]
    
    print(f"数据类型: float16")
    print(f"{'矩阵大小':<20} {'TFLOPS':<10}")
    print("-" * 30)
    
    for size in matrix_sizes:
        try:
            tflops = measure_gemm_tflops(size, dtype=torch.float16)
            print(f"{str(size):<20} {tflops:.2f}")
        except Exception as e:
            print(f"{str(size):<20} 失败: {str(e)}")
