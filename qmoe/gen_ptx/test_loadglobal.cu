



#pragma once 
 
#include <cuda_runtime.h>
#include <stddef.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cublasLt.h>

// nvcc -ptx -arch=compute_90a -arch=compute_90a kernel.cu -o kernel.ptx



extern "C" {

  __device__ __noinline__ int load_global_int(int* q) {


      int output = q[0];


      return output;
  }

  __device__ __noinline__ void save_global_int(int* q) {

      int  data = 152.0;
      q[0] = data;

    //   printf("111111");
      return ;
  }



  __global__ void test_dequant(int input, int *output){


      save_global_int(output);
  
  }

}