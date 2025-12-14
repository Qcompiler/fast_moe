nvcc  i4_kernel.cu -std=c++17 -w -Xcudafe --diag_suppress=177 --compiler-options -fPIC    -lcuda -gencode arch=compute_90a,code=sm_90a --ptx   --shared


nvcc  test_loadglobal.cu -std=c++17 -w -Xcudafe --diag_suppress=177 --compiler-options -fPIC    -lcuda -gencode arch=compute_90a,code=sm_90a --ptx   --shared