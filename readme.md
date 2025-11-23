# 代码结构说明

# 3rd cutlass
这个是一个nvidia开源的矩阵乘法的算子库，对于编写算子非常有帮助 

# jitcu
这个提供了just in time 的从cuda到python的动态编译的方法

# marlin
这个是一个int4的矩阵乘法的库，我们在比较gemv int4的时候把他作为baseline


# sglang
这个代码可以作为参考，如何修改推理引擎，把我们的算子加入进去


# qmoe 
这个是我们的主要代码路径，学习路径：

学习gemv

```srun -N 1  --pty --gres=gpu:H100:1  python  test_gemv.py```

学习 int 4 gemv

srun -N 1  --pty --gres=gpu:H100:1  python   test_gemv_int4.py

学习 int 4 gemv group

srun -N 1  --pty --gres=gpu:H100:1  python   test_gemv_int4_group.py

学习 int4 gemv mix

srun -N 1  --pty --gres=gpu:H100:1  python   test_gemv_mix.py

学习 int4 gemv mix group

srun -N 1  --pty --gres=gpu:H100:1  python   test_gemv_int4_group_mix.py

学习 moe 

moe_gemv_gate.py
moe_gemv_down.py


学习 int4 moe

i4_moe_gemv_gate.py
i4_moe_gemv_down.py





