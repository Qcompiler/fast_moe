# 参考cuda kernel 代码
```
https://github.com/NVIDIA/cutlass/blob/49bd6bf1ba80abd588a56a3f3af2f1bcd41d215f/include/cutlass/detail/collective/mixed_input_utils.hpp#L223
```

# 关键步骤

找到 dequant kernel， 编写 cu文件。 使用 nvcc 编译成为 ptx， 尝试插入到 triton 里面。（建议参考int4 ，先试试 int8）


# 可以参考的文件


（1）gen_ptx/i4_kernel.cu
（2) gen_ptx/i4_kernel.ptx
 (3)test_gemv_int4_triton_inline_ptx.py