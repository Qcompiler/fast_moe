nvcc  -diag-suppress=1444 -std=c++17 -I 3rd/cutlass/include  \
 -gencode=arch=compute_86,code=sm_86  -gencode=arch=compute_89,code=sm_89 \
    -gencode=arch=compute_90a,code=sm_90a  -gencode=arch=compute_120a,code=sm_120a  \
 bandwidth_test_gemv_quant.cu --expt-relaxed-constexpr \
  -DUSE_CUDA_KERNEL -DQUANT_TEST1  -o  gemv_cuda_quant.out

nvcc  -diag-suppress=1444 -std=c++17 -I 3rd/cutlass/include  \
 -gencode=arch=compute_86,code=sm_86  -gencode=arch=compute_89,code=sm_89 \
    -gencode=arch=compute_90a,code=sm_90a  -gencode=arch=compute_120a,code=sm_120a  \
 bandwidth_test_gemv_quant.cu --expt-relaxed-constexpr \
  -DUSE_CUDA_KERNEL -DQUANT_TEST2  -o  gemv_cuda_quant2.out

nvcc  -diag-suppress=1444 -std=c++17 -I 3rd/cutlass/include  \
 -gencode=arch=compute_86,code=sm_86  -gencode=arch=compute_89,code=sm_89 \
    -gencode=arch=compute_90a,code=sm_90a  -gencode=arch=compute_120a,code=sm_120a  \
 bandwidth_test_gemv_quant.cu --expt-relaxed-constexpr \
  -DUSE_CUDA_KERNEL -DQUANT_TEST3  -o  gemv_cuda_quant3.out

nvcc  -diag-suppress=1444 -std=c++17 -I 3rd/cutlass/include  \
 -gencode=arch=compute_86,code=sm_86  -gencode=arch=compute_89,code=sm_89 \
    -gencode=arch=compute_90a,code=sm_90a  -gencode=arch=compute_120a,code=sm_120a  \
 bandwidth_test_gemv_quant.cu --expt-relaxed-constexpr \
  -DUSE_CUDA_KERNEL -DQUANT_TEST4  -o  gemv_cuda_quant4.out

nvcc  -diag-suppress=1444 -std=c++17 -I 3rd/cutlass/include  \
 -gencode=arch=compute_86,code=sm_86  -gencode=arch=compute_89,code=sm_89 \
    -gencode=arch=compute_90a,code=sm_90a  -gencode=arch=compute_120a,code=sm_120a  \
 bandwidth_test_gemv.cu --expt-relaxed-constexpr \
  -DUSE_CUDA_KERNEL  -o  gemv_cuda.out


nvcc  -diag-suppress=1444 -std=c++17 -I 3rd/cutlass/include  \
 -gencode=arch=compute_86,code=sm_86  -gencode=arch=compute_89,code=sm_89 \
    -gencode=arch=compute_90a,code=sm_90a  -gencode=arch=compute_120a,code=sm_120a  \
 bandwidth_test_gemv.cu --expt-relaxed-constexpr \
  -DUSE_WARP_SPECIAL  -o  gemv_cuda_shared.out


nvcc  -diag-suppress=1444 -std=c++17 -I 3rd/cutlass/include  \
 -gencode=arch=compute_86,code=sm_86  -gencode=arch=compute_89,code=sm_89 \
    -gencode=arch=compute_90a,code=sm_90a  -gencode=arch=compute_120a,code=sm_120a  \
 bandwidth_test_gemv_quant.cu --expt-relaxed-constexpr \
  -DUSE_WARP_SPECIAL  -DQUANT1  -o  gemv_cuda_shared_quant1.out


nvcc  -diag-suppress=1444 -std=c++17 -I 3rd/cutlass/include  \
 -gencode=arch=compute_86,code=sm_86  -gencode=arch=compute_89,code=sm_89 \
    -gencode=arch=compute_90a,code=sm_90a  -gencode=arch=compute_120a,code=sm_120a  \
 bandwidth_test_gemv_quant.cu --expt-relaxed-constexpr \
  -DUSE_WARP_SPECIAL   -DQUANT2  -o  gemv_cuda_shared_quant2.out