nvcc  -std=c++17 -I 3rd/cutlass/include -gencode=arch=compute_86,code=sm_86  -gencode=arch=compute_89,code=sm_89 \
  -gencode=arch=compute_90a,code=sm_90a  -gencode=arch=compute_120a,code=sm_120a   bandwidth_test.cu


nvcc  -std=c++17 -I 3rd/cutlass/include -gencode=arch=compute_86,code=sm_86  -gencode=arch=compute_89,code=sm_89 \
  -gencode=arch=compute_90a,code=sm_90a  -gencode=arch=compute_120a,code=sm_120a   -DUSE_CUDA_KERNEL  bandwidth_test_half.cu -o band_test


nvcc  -diag-suppress=1444 -std=c++17 -I 3rd/cutlass/include -gencode=arch=compute_86,code=sm_86  -gencode=arch=compute_89,code=sm_89 \
  -gencode=arch=compute_90a,code=sm_90a  -gencode=arch=compute_120a,code=sm_120a  -DUSE_CUDA_KERNEL gemv_test_half.cu --expt-relaxed-constexpr


nvcc  -diag-suppress=1444 -std=c++17 -I 3rd/cutlass/include -gencode=arch=compute_86,code=sm_86  \
  -gencode=arch=compute_89,code=sm_89   -gencode=arch=compute_90a,code=sm_90a   \
   -gencode=arch=compute_120a,code=sm_120a   bandwidth_test_sum.cu  \
    --expt-relaxed-constexpr -DUSE_CUDA_KERNEL    -o  sum.out

nvcc  -ptx -std=c++17 -I 3rd/cutlass/include \
 -gencode=arch=compute_90a,code=sm_90a  -DUSE_CUDA_KERNEL bandwidth_test_half.cu  -o slow.ptx
nvcc  -ptx -std=c++17 -I 3rd/cutlass/include \
 -gencode=arch=compute_90a,code=sm_90a  -DUSE_CUTE_KERNEL bandwidth_test_half.cu  -o fast.ptx

 
nvcc  -diag-suppress=1444 -std=c++17 -I 3rd/cutlass/include -gencode=arch=compute_86,code=sm_86  -gencode=arch=compute_89,code=sm_89   -gencode=arch=compute_90a,code=sm_90a  -gencode=arch=compute_120a,code=sm_120a   gemv_test_half.cu --expt-relaxed-constexpr -DUSE_CUDA_KERNEL  -DQUANT_TEST1 -o  test1
nvcc  -diag-suppress=1444 -std=c++17 -I 3rd/cutlass/include -gencode=arch=compute_86,code=sm_86  -gencode=arch=compute_89,code=sm_89   -gencode=arch=compute_90a,code=sm_90a  -gencode=arch=compute_120a,code=sm_120a   gemv_test_half.cu --expt-relaxed-constexpr -DUSE_CUDA_KERNEL  -DQUANT_TEST2 -o  test2
nvcc  -diag-suppress=1444 -std=c++17 -I 3rd/cutlass/include -gencode=arch=compute_86,code=sm_86  -gencode=arch=compute_89,code=sm_89   -gencode=arch=compute_90a,code=sm_90a  -gencode=arch=compute_120a,code=sm_120a   gemv_test_half.cu --expt-relaxed-constexpr -DUSE_CUDA_KERNEL   -o  a.out

nvcc  -diag-suppress=1444 -std=c++17 -I 3rd/cutlass/include  -gencode=arch=compute_90a,code=sm_90a  \
  -gencode=arch=compute_120a,code=sm_120a   gemv_test_half.cu --expt-relaxed-constexpr -DUSE_CLUSTER_KERNEL\
   -o  cluster.out

nvcc  -diag-suppress=1444 -std=c++17 -I 3rd/cutlass/include \
 -gencode=arch=compute_86,code=sm_86  -gencode=arch=compute_89,code=sm_89  \
  -gencode=arch=compute_90a,code=sm_90a  -gencode=arch=compute_120a,code=sm_120a \
  bandwidth_test_gemv.cu --expt-relaxed-constexpr -DUSE_CUDA_KERNEL   -o  a.out