from triton.testing import do_bench
import triton.language as tl
import torch
import triton



@triton.jit
def save_i32(ptr):
    return tl.inline_asm_elementwise(
        asm="""
        { 
            .reg .b32 	r<2>;
            mov.u32 	r1, 152;
            st.global.u32 [$1], r1;
        }
            
        """,
        constraints=("=l,l"),
        args=[ptr],
        dtype=(tl.int32),
        is_pure=False,
        pack=1
    )

# @triton.jit
# def save_data_to_global(global_ptr):
#     tl.inline_asm_elementwise(
#         asm="""
#             {
#                 .reg .b32 	r<2>;
#                 .reg .b64 	yd<3>;                
#                 cvta.to.global.u64 	yd2, $1;
#                 mov.u32 	r1, 152;
#                 st.global.u32 	[yd2], r1; 

#             }
#         """,
#         constraints=(
#             "=r,r"
#         ),
#         args=[global_ptr], #输入 #参数 1
#         is_pure=False,
#         dtype=(tl.float16), #参数 0
#         pack=1,
#     )
#     return 

@triton.jit
def test_save_kernel(
    output_ptr,
    m, n,
    stride_am, stride_an,
    BLOCK_SIZE: tl.constexpr
):
    # 每个程序处理多行，每个线程处理一行
    pid = tl.program_id(axis=0)
    
    # 计算行ID范围
    row_start = pid * BLOCK_SIZE
    
    # 每个线程处理一行
    row_idx = row_start + tl.arange(0, BLOCK_SIZE)

    # 计算存储地址
    offsets = row_idx * stride_am
    output_ptrs = output_ptr + offsets
    # 调用内联汇编函数
    save_i32(output_ptrs)

# 完整的测试函数
def test_save_global():
    # 设置测试参数
    M = 1024  # 行数
    N = 1     # 列数（因为我们只保存一个值）
    BLOCK_SIZE = 256
    
    # 创建输出张量
    output = torch.zeros((M, N), dtype=torch.int32, device='cuda')
    
    # 计算网格大小
    grid = (triton.cdiv(M, BLOCK_SIZE),)
    
    # 启动内核
    kernel = test_save_kernel[grid](
        output_ptr=output,
        m=M,
        n=N,
        stride_am=output.stride(0),
        stride_an=output.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # print(kernel.asm["ptx"])
    # f = open("save_global.ptx", "w")
    # f.write(kernel.asm["ptx"])
    # f.close()
    # return 
    # 同步以确保计算完成
    torch.cuda.synchronize()
    
    # 验证结果 - 内联汇编应该写入值152
    expected_value = 152
    actual_values = output.flatten().cpu().numpy()
    
    # 检查前几个值
    print(f"前10个输出值: {actual_values[:10]}")
    
    # 验证所有值是否都是152
    all_correct = all(val == expected_value for val in actual_values[:M])
    print(f"所有值都正确写入152: {all_correct}")
    


    return output

# 运行测试
if __name__ == "__main__":
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print("CUDA可用，开始测试...")
        result = test_save_global()
        print("测试完成！")
    else:
        print("CUDA不可用，跳过测试")