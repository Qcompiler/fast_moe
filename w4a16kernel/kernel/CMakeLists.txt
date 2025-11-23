



cmake_minimum_required(VERSION 3.8)
set(TARGET_NAME moekernel)
project(${TARGET_NAME})

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++20 -Xcompiler "-fPIC" 
        -gencode=arch=compute_89,code=sm_89 
        -gencode=arch=compute_90a,code=sm_90a 
        -O2 -DNDEBUG --expt-relaxed-constexpr
        -DENABLE_SCALED_MM_C3X=1
        -DENABLE_SCALED_MM_C2X=1
        -DCUTE_USE_PACKED_TUPLE=1
        -DCUTLASS_TEST_LEVEL=0
        -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1
        -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1
        -DCUTLASS_DEBUG_TRACE_LEVEL=0
        -Xcompiler=-fPIC
        -lineinfo
        -DENABLE_FP8
        --threads=1
        --ptxas-options=-v
        -D_GLIBCXX_USE_CXX11_ABI=0)
message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(csrc)
include_directories(cutlass/include)
include_directories(.)
cuda_add_library(
  ${TARGET_NAME} SHARED
  kernel.cu
  kerneli4.cu
 )


