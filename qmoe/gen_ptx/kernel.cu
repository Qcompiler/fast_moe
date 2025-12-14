#pragma once 
 
#include <cuda_runtime.h>
#include <stddef.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cublasLt.h>

// nvcc -ptx -arch=compute_90a -arch=compute_90a kernel.cu -o kernel.ptx

template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile(
    "lop3.b32 %0, %1, %2, %3, %4;\n"
    : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
  );
  return res;
}
__noinline__  __device__ int64_t dequant(int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  int64_t output = 0;
  half2 *frag_b1 = reinterpret_cast<half2*>(&output);
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point directly into `SUB` and `ADD`.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;

  frag_b1[0] = __hsub2(
    *reinterpret_cast<half2*>(&lo),
    *reinterpret_cast<const half2*>(&SUB)
  );
  frag_b1[1] = __hfma2(
    *reinterpret_cast<half2*>(&hi),
    *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)
  );
  return output;
}



__noinline__  __device__ int64_t dequant_int64(uint64_t q) {

  int64_t output = 0;
  half *output_ = reinterpret_cast<half*>(&output);
  
  half *input = reinterpret_cast<half*>(&q);

  output_[0] = input[0];
  output_[1] = input[1];
  output_[2] = input[2];
  output_[3] = input[3];
  return output;
}


__global__ void test_dequant(int input, int64_t *output){


    int64_t output_ = dequant(input);
    
    int64_t output_int64 = dequant_int64(input);
    *output = output_;
    *output += output_int64;
}

int main(){

    int64_t output;
    test_dequant<<<1, 1>>>(1, &output);
    printf("output: %ld\n", output);
    cudaDeviceSynchronize();

}