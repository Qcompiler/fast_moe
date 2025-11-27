import torch
import triton
import triton.language as tl
from triton.testing import do_bench_cudagraph,do_bench

# 解量化函数

@triton.jit
def dequanti(b):
    x1, x2 = tl.inline_asm_elementwise(
        asm="""
            {
            .reg .b32 	r<16>;
	        .reg .b64 	rd<2>;
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
            mov.u32 	$0, r9;
            mov.u32 	$1, r12;
            }
        """,
        constraints=(
            "=r,=r,r"
        ),
        args=[b], #输入
        dtype=(tl.int32, tl.int32), #输出
        is_pure=False,
        pack=1,
    )

    
    return x1, x2

@triton.jit
def dequant_kernel_combined(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载输入数据
    b = tl.load(input_ptr + offsets, mask=mask)
    
    # 调用解量化函数
    result_part0, result_part1 = dequanti(b)
    b = b >> 8
    result_part2, result_part3 = dequanti(b)
    
    # 将两个结果交错存储到一个输出张量中
    output_offsets = offsets * 4
    tl.store(output_ptr + output_offsets, result_part0, mask=mask)
    tl.store(output_ptr + output_offsets + 1, result_part1, mask=mask)
    tl.store(output_ptr + output_offsets + 2, result_part2, mask=mask)
    tl.store(output_ptr + output_offsets + 3, result_part3, mask=mask)

from common.common import generate_randint, gen_quant4, gen_quant4_my


def int32_to_two_fp16(int32_tensor):
    """
    将每个 int32 转换为两个 fp16
    input: int32 tensor of shape [1, n]  
    output: fp16 tensor of shape [1, 2n]
    """
    # 确保是 int32 类型
    int32_tensor = int32_tensor.to(torch.int32)
    n = int32_tensor.shape[1]
    
    # 方法1: 使用位操作分离高16位和低16位
    high_16 = (int32_tensor >> 16).to(torch.int16)      # 高16位
    low_16 = (int32_tensor & 0xFFFF).to(torch.int16)    # 低16位
    
    # 将 int16 重新解释为 fp16
    high_fp16 = high_16.view(torch.float16)
    low_fp16 = low_16.view(torch.float16)
    
    # 交错拼接: [high1, low1, high2, low2, ...]
    result = torch.stack([low_fp16, high_fp16, ], dim=-1).view(1, 2 * n)
    
    return result

def test_dequant_combined():
    # 设置输入数据
    n_elements = 4096 * 4096
    input_tensor = torch.randint(-8, 8, (1, n_elements), dtype=torch.int32, device='cuda').to(torch.float16)
    
    q_weight, scales  = gen_quant4_my(1, n_elements, torch.clone(input_tensor),   groupsize = -1, tile = 1)
    # 准备输出张量（两倍大小）

    print(q_weight.shape)

    q_weight = q_weight.reshape(-1, 1)
    # exit()
    K = n_elements // 8
    output = torch.zeros(K * 4, dtype=torch.int32, device='cuda')
    
    # 计算网格大小
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(K, BLOCK_SIZE),)
    
    # 启动kernel
    dequant_kernel_combined[grid](
        q_weight,
        output,
        K,
        BLOCK_SIZE,
    )
    
    # 同步等待kernel完成
    torch.cuda.synchronize()

    # ms = do_bench_cudagraph(lambda: dequant_kernel_combined[grid](
    #     q_weight,
    #     output,
    #     K,
    #     BLOCK_SIZE,
    # ))

    # print(ms)

    
    print("输入张量形状:", input_tensor.shape)
    print("输出张量形状:", output.shape)
    print(input_tensor.cpu().numpy()[0, 0:16])
    # print("输出前20个元素:", output[:20].cpu().numpy())
    
    output = output.reshape((1, -1))
    fp16 = int32_to_two_fp16(output)
    print(fp16.cpu().numpy()[0, 0:16])
    return output
# 如果希望将两个输出合并为一个张量的版本


if __name__ == "__main__":
    print("=== 测试分离输出版本 ===")
    out1 = test_dequant_combined()
    

    
