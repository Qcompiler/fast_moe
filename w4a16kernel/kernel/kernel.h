#pragma once 
 
#include <cuda_runtime.h>
#include <stddef.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
#define WARP_SIZE 32
#define SHARED_MEM_MAX_ROWS 64
#define MAX_THREADS_PER_BLOCK 1024







template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) {
    return elems[i];
  }
};

using FragB = Vec<half2, 2>;


template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile(
    "lop3.b32 %0, %1, %2, %3, %4;\n"
    : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
  );
  return res;
}
__device__ inline FragB dequant(int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  FragB frag_b;
  // return frag_b;

  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point directly into `SUB` and `ADD`.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;

  frag_b[0] = __hsub2(
    *reinterpret_cast<half2*>(&lo),
    *reinterpret_cast<const half2*>(&SUB)
  );
  frag_b[1] = __hfma2(
    *reinterpret_cast<half2*>(&hi),
    *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)
  );
  return frag_b;
}






  

__device__ __forceinline__ float warpReduceSum(float sum,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
  return sum;

    // sum += __shfl_xor_sync(~0, sum, 16);
    // sum += __shfl_xor_sync(~0, sum, 8);
    // sum += __shfl_xor_sync(~0, sum, 4);
    // sum += __shfl_xor_sync(~0, sum, 2);
    // sum += __shfl_xor_sync(~0, sum, 1);
    // return sum;

}


// simple gemv
void gemv_cu(int m, int n, int k,  half * vec_data_,
                                  half * mat_data_, 
                                  half * result_data_,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y , 
                                   cudaStream_t stream); 

void gemv_moe_cu(const half * mat_data_, 
                const half * vec_data_,
                half * result_data_,
                int32_t *moe_index,
                int ntopx,
                int hidden,
                int num_expert,
                int output_dim,
                unsigned int block_dim_x,
                unsigned int block_dim_y, 
                cudaStream_t stream);

void gemv_moe_cu(const __nv_bfloat16 * mat_data_, 
                const __nv_bfloat16 * vec_data_,
                __nv_bfloat16 * result_data_,
                int32_t *moe_index,
                int ntopx,
                int hidden,
                int num_expert,
                int output_dim,
                unsigned int block_dim_x,
                unsigned int block_dim_y, 
                cudaStream_t stream);


void gemv_moe_down_cu(const half * mat_data_, 
                const half * vec_data_,
                half * result_data_,
                const float* topk_weight,
                int32_t *moe_index,
                int ntopx,
                int hidden,
                int num_expert,
                int output_dim,
                unsigned int block_dim_x,
                unsigned int block_dim_y, 
                cudaStream_t stream);
void gemv_moe_down_cu(const __nv_bfloat16 * mat_data_, 
                const __nv_bfloat16 * vec_data_,
                __nv_bfloat16 * result_data_,
                const float* topk_weight,
                int32_t *moe_index,
                int ntopx,
                int hidden,
                int num_expert,
                int output_dim,
                unsigned int block_dim_x,
                unsigned int block_dim_y, 
                cudaStream_t stream);                


void check(cudaError_t result, char const* const func, const char* const file,
           int const line);

void gemv_moe_cu_i4(const int32_t * mat_data_, 
                const half * vec_data_,
                half * result_data_,
                int32_t *moe_index,
                int ntopx,
                int hidden,
                int num_expert,
                int output_dim,
                unsigned int block_dim_x,
                unsigned int block_dim_y, 
                const half * scaling_data_,
                int groupsize,
                const half *outliers,
                const int32_t *ind,
                int n_outliers,
                cudaStream_t stream, int arch);


void gemv_moe_down_cu_i4(
                const int32_t * mat_data_, 
                const half * vec_data_,
                half * result_data_,
                const float * topk_weight,
                const int32_t *moe_index,
                int ntopx,
                int hidden,
                int num_expert,
                int output_dim,
                unsigned int block_dim_x,
                unsigned int block_dim_y, 
                const half * scaling_data_,
                int groupsize,
                const half *outliers,
                const int32_t *ind,
                int n_outliers,
                cudaStream_t stream, int arch);



__device__ __forceinline__ static half2 to_vec2(half v)
{
    return __half2half2(v);
}