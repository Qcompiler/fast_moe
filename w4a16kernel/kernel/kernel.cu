

#include "kernel.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include <assert.h>

void check(cudaError_t result, char const* const func, const char* const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error = %s at %s:%d '%s'\n",
            cudaGetErrorString(result), file, line, func);
    exit(1);
  }
}


 


__global__ void gemv_fp16(half* mat, half* vec, half* res, unsigned int n,
                          unsigned int num_per_thread) {
    float sum = 0;
    // each thread load num_per_thread elements from global
    unsigned int tid = threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int start_idx = threadIdx.x;
    float4* mat4 = reinterpret_cast<float4*>(mat);
    float4* vec4 = reinterpret_cast<float4*>(vec);

  #pragma unroll
    for (int iter = 0; iter < num_per_thread >> 3; iter++) {
      unsigned int j = start_idx + iter * blockDim.x;
      if (j < n >> 3) {
        float4 vec_val = vec4[j];
        float4 mat_val = mat4[row * (n >> 3) + j];
        half2* vec_h1 = (half2*)&vec_val.x;
        half2* vec_h2 = (half2*)&vec_val.y;
        half2* vec_h3 = (half2*)&vec_val.z;
        half2* vec_h4 = (half2*)&vec_val.w;
        half2* mat_h1 = (half2*)&mat_val.x;
        half2* mat_h2 = (half2*)&mat_val.y;
        half2* mat_h3 = (half2*)&mat_val.z;
        half2* mat_h4 = (half2*)&mat_val.w;
        sum += __half2float(vec_h1->x) * __half2float(mat_h1->x);
        sum += __half2float(vec_h1->y) * __half2float(mat_h1->y);
        sum += __half2float(vec_h2->x) * __half2float(mat_h2->x);
        sum += __half2float(vec_h2->y) * __half2float(mat_h2->y);
        sum += __half2float(vec_h3->x) * __half2float(mat_h3->x);
        sum += __half2float(vec_h3->y) * __half2float(mat_h3->y);
        sum += __half2float(vec_h4->x) * __half2float(mat_h4->x);
        sum += __half2float(vec_h4->y) * __half2float(mat_h4->y);
      }
    }

    sum = warpReduceSum(sum, blockDim.x);

    if (blockDim.x <= WARP_SIZE) {
      if (tid == 0) {
        res[row] = __float2half(sum);
      }
      return;
    }

    // Shared mem for partial sums (one per warp in the block)
    static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
    __syncthreads();
    // read from shared memory only if that warp existed
    sum = (threadIdx.x < blockDim.x / WARP_SIZE)
              ? warpLevelSums[threadIdx.y][laneId]
              : 0.0;
    // Final reduce using first warp
    if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
    if (tid == 0) {
      res[row] = __float2half(sum);
    }
}


// mat n expect * output_dim  *  hidden

template< int max_expect>
__global__ void gemv_moe_fp16(const half* mat, 
                            const half* vec, 
                            half* res, 
                            const int32_t *moe_index,
                            int ntopx,
                            unsigned int hidden,
                            unsigned int num_expert, 
                            unsigned int output_dim, 
                            unsigned int num_per_thread) 
                          {

    
    // each thread load num_per_thread elements from global
    unsigned int tid = threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int start_idx = threadIdx.x;

    float sum[max_expect];


    #pragma unroll
    for (int topx = 0; topx < max_expect; topx++ ){
        sum[topx] = 0.0;
    }


    #pragma unroll
    for (int topx = 0; topx < ntopx; topx++ ){

        int target_expect = moe_index[topx];

    

        const half * target_mat = mat +  target_expect * hidden * output_dim;
        const float4* mat4 = reinterpret_cast<const float4*>(target_mat);
        const float4* vec4 = reinterpret_cast<const float4*>(vec);

        #pragma unroll
          for (int iter = 0; iter < num_per_thread >> 3; iter++) {
            unsigned int j = start_idx + iter * blockDim.x;
            if (j < hidden >> 3) {
                float4 vec_val = vec4[j];
                float4 mat_val = mat4[row * (hidden >> 3) + j];
                half2* vec_h1 = (half2*)&vec_val.x;
                half2* vec_h2 = (half2*)&vec_val.y;
                half2* vec_h3 = (half2*)&vec_val.z;
                half2* vec_h4 = (half2*)&vec_val.w;
                half2* mat_h1 = (half2*)&mat_val.x;
                half2* mat_h2 = (half2*)&mat_val.y;
                half2* mat_h3 = (half2*)&mat_val.z;
                half2* mat_h4 = (half2*)&mat_val.w;
                sum[topx] += __half2float(vec_h1->x) * __half2float(mat_h1->x);
                sum[topx] += __half2float(vec_h1->y) * __half2float(mat_h1->y);
                sum[topx] += __half2float(vec_h2->x) * __half2float(mat_h2->x);
                sum[topx] += __half2float(vec_h2->y) * __half2float(mat_h2->y);
                sum[topx] += __half2float(vec_h3->x) * __half2float(mat_h3->x);
                sum[topx] += __half2float(vec_h3->y) * __half2float(mat_h3->y);
                sum[topx] += __half2float(vec_h4->x) * __half2float(mat_h4->x);
                sum[topx] += __half2float(vec_h4->y) * __half2float(mat_h4->y);
            }
          }

        sum[topx] = warpReduceSum(sum[topx], blockDim.x);


  }

   
   

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      // res ntopx * output_dim
       #pragma unroll
      for (int topx = 0; topx < ntopx; topx++ ){
          res[row + topx * output_dim] = __float2half(sum[topx]);
      }

    }
    return;
  }

          

  
}



template< int max_expect>
__global__ void gemv_moe_bf16(const __nv_bfloat16* mat, 
                            const __nv_bfloat16* vec, 
                            __nv_bfloat16* res, 
                            const int32_t *moe_index,
                            int ntopx,
                            unsigned int hidden,
                            unsigned int num_expert, 
                            unsigned int output_dim, 
                            unsigned int num_per_thread) 
                          {

    
    // each thread load num_per_thread elements from global
    unsigned int tid = threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int start_idx = threadIdx.x;

    float sum[max_expect];


    #pragma unroll
    for (int topx = 0; topx < max_expect; topx++ ){
        sum[topx] = 0.0;
    }


    #pragma unroll
    for (int topx = 0; topx < ntopx; topx++ ){

        int target_expect = moe_index[topx];

    

        const __nv_bfloat16 * target_mat = mat +  target_expect * hidden * output_dim;
        const float4* mat4 = reinterpret_cast<const float4*>(target_mat);
        const float4* vec4 = reinterpret_cast<const float4*>(vec);

        #pragma unroll
          for (int iter = 0; iter < num_per_thread >> 3; iter++) {
            unsigned int j = start_idx + iter * blockDim.x;
            if (j < hidden >> 3) {
                float4 vec_val = vec4[j];
                float4 mat_val = mat4[row * (hidden >> 3) + j];
                __nv_bfloat162* vec_h1 = (__nv_bfloat162*)&vec_val.x;
                __nv_bfloat162* vec_h2 = (__nv_bfloat162*)&vec_val.y;
                __nv_bfloat162* vec_h3 = (__nv_bfloat162*)&vec_val.z;
                __nv_bfloat162* vec_h4 = (__nv_bfloat162*)&vec_val.w;
                __nv_bfloat162* mat_h1 = (__nv_bfloat162*)&mat_val.x;
                __nv_bfloat162* mat_h2 = (__nv_bfloat162*)&mat_val.y;
                __nv_bfloat162* mat_h3 = (__nv_bfloat162*)&mat_val.z;
                __nv_bfloat162* mat_h4 = (__nv_bfloat162*)&mat_val.w;
                sum[topx] += __bfloat162float(vec_h1->x) * __bfloat162float(mat_h1->x);
                sum[topx] += __bfloat162float(vec_h1->y) * __bfloat162float(mat_h1->y);
                sum[topx] += __bfloat162float(vec_h2->x) * __bfloat162float(mat_h2->x);
                sum[topx] += __bfloat162float(vec_h2->y) * __bfloat162float(mat_h2->y);
                sum[topx] += __bfloat162float(vec_h3->x) * __bfloat162float(mat_h3->x);
                sum[topx] += __bfloat162float(vec_h3->y) * __bfloat162float(mat_h3->y);
                sum[topx] += __bfloat162float(vec_h4->x) * __bfloat162float(mat_h4->x);
                sum[topx] += __bfloat162float(vec_h4->y) * __bfloat162float(mat_h4->y);
            }
          }

        sum[topx] = warpReduceSum(sum[topx], blockDim.x);


  }

   
   

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      // res ntopx * output_dim
       #pragma unroll
      for (int topx = 0; topx < ntopx; topx++ ){
          res[row + topx * output_dim] = __float2bfloat16(sum[topx]);
      }

    }
    return;
  }

          

  
}



template< int max_expect>
__global__ void gemv_moe_down_fp16(const half* mat, 
                            const half* vec, 
                            half* res, 
                            const float* topk_weight,
                            const int32_t *moe_index,
                            int ntopx,
                            unsigned int hidden,
                            unsigned int num_expert, 
                            unsigned int output_dim, 
                            unsigned int num_per_thread) 
                          {

    
    // each thread load num_per_thread elements from global
    unsigned int tid = threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int start_idx = threadIdx.x;


    float sum[max_expect];


    #pragma unroll
    for (int topx = 0; topx < max_expect; topx++ ){
        sum[topx] = 0.0;
    }



    #pragma unroll
    for (int topx = 0; topx < ntopx; topx++ ){

        int target_expect = moe_index[topx];

    

        const half * target_mat = mat +  target_expect * hidden * output_dim;
        const half * target_vec = vec +  topx * hidden;
        
        const float4* mat4 = reinterpret_cast<const float4*>(target_mat);
        const float4* vec4 = reinterpret_cast<const float4*>(target_vec);

        #pragma unroll
          for (int iter = 0; iter < num_per_thread >> 3; iter++) {
            unsigned int j = start_idx + iter * blockDim.x;
            if (j < hidden >> 3) {
                float4 vec_val = vec4[j];
                float4 mat_val = mat4[row * (hidden >> 3) + j];
                half2* vec_h1 = (half2*)&vec_val.x;
                half2* vec_h2 = (half2*)&vec_val.y;
                half2* vec_h3 = (half2*)&vec_val.z;
                half2* vec_h4 = (half2*)&vec_val.w;
                half2* mat_h1 = (half2*)&mat_val.x;
                half2* mat_h2 = (half2*)&mat_val.y;
                half2* mat_h3 = (half2*)&mat_val.z;
                half2* mat_h4 = (half2*)&mat_val.w;
                sum[topx] += __half2float(vec_h1->x) * __half2float(mat_h1->x);
                sum[topx] += __half2float(vec_h1->y) * __half2float(mat_h1->y);
                sum[topx] += __half2float(vec_h2->x) * __half2float(mat_h2->x);
                sum[topx] += __half2float(vec_h2->y) * __half2float(mat_h2->y);
                sum[topx] += __half2float(vec_h3->x) * __half2float(mat_h3->x);
                sum[topx] += __half2float(vec_h3->y) * __half2float(mat_h3->y);
                sum[topx] += __half2float(vec_h4->x) * __half2float(mat_h4->x);
                sum[topx] += __half2float(vec_h4->y) * __half2float(mat_h4->y);
            }
          }
        if (num_per_thread % 8 ){
              int iter = num_per_thread >> 3;
              unsigned int j = start_idx + iter * blockDim.x;
              // 后面还有4个数
              if (j < hidden >> 3) {
                  const float4* vec4_now = vec4 +j;
                  float2 vec_val = reinterpret_cast<const float2*>(vec4_now)[0];
                  const float4* mat4_now = mat4 + (row * (hidden >> 3) + j);
                  float2 mat_val =  reinterpret_cast<const float2*>(mat4_now)[0]; 
                  half2* vec_h1 = (half2*)&vec_val.x;
                  half2* vec_h2 = (half2*)&vec_val.y;

                  half2* mat_h1 = (half2*)&mat_val.x;
                  half2* mat_h2 = (half2*)&mat_val.y;

                  sum[topx] += __half2float(vec_h1->x) * __half2float(mat_h1->x);
                  sum[topx] += __half2float(vec_h1->y) * __half2float(mat_h1->y);
                  sum[topx] += __half2float(vec_h2->x) * __half2float(mat_h2->x);
                  sum[topx] += __half2float(vec_h2->y) * __half2float(mat_h2->y);
                  
              }
            

        }

        sum[topx] = warpReduceSum(sum[topx], blockDim.x);


  }

   
   

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      // res ntopx * output_dim
      float sum_all = 0.0;
      #pragma unroll
      for (int topx = 0; topx < ntopx; topx++ ){
          sum_all += sum[topx] * (topk_weight[topx]);
      }
      res[row] = __float2half(sum_all);
      

    }
    return;
  }

}

void gemv_cu(int m, int n, int k,  half * vec_data_,
                                  half * mat_data_, 
                                  half * result_data_,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y , cudaStream_t stream){
                      
  
    assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
    assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
    unsigned int num_per_thread = k / block_dim_x; 
    assert(num_per_thread >= 8);
    assert(num_per_thread % 4 == 0);


    dim3 grid_dim(1, n / block_dim_y);
    dim3 block_dim(block_dim_x, block_dim_y);
    gemv_fp16<<<grid_dim, block_dim, 1024, stream>>>(mat_data_, vec_data_, result_data_,
                                      k, num_per_thread);
    checkCudaErrors(cudaPeekAtLastError());
    return  ;
}





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
                cudaStream_t stream){
    assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
    assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
    unsigned int num_per_thread = hidden / block_dim_x; 
    assert(num_per_thread >= 8);

    dim3 grid_dim(1, output_dim / block_dim_y);
    dim3 block_dim(block_dim_x, block_dim_y);
    // moe gate_up_data: [expert, output_dim, hidden_dim]
    // vec_data  :        [1 ,    hidden_dim]
    // output_data :      [ ntopx,  output_dim]

    assert ( ntopx <= 8);

    gemv_moe_fp16<8><<<grid_dim, block_dim, 1024, stream>>>
              (mat_data_, vec_data_, result_data_,
                moe_index, ntopx, hidden, 
                num_expert, output_dim, num_per_thread);

    
    
    checkCudaErrors(cudaPeekAtLastError());
    return  ;
}



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
                cudaStream_t stream){
    assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
    assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
    unsigned int num_per_thread = hidden / block_dim_x; 
    assert(num_per_thread >= 8);
    assert(num_per_thread % 4 == 0);

    dim3 grid_dim(1, output_dim / block_dim_y);
    dim3 block_dim(block_dim_x, block_dim_y);
    // moe gate_up_data: [expert, output_dim, hidden_dim]
    // vec_data  :        [1 ,    hidden_dim]
    // output_data :      [ ntopx,  output_dim]

    assert ( ntopx <= 8);

    gemv_moe_bf16<8><<<grid_dim, block_dim, 1024, stream>>>
              (mat_data_, vec_data_, result_data_,
                moe_index, ntopx, hidden, 
                num_expert, output_dim, num_per_thread);

    
    
    checkCudaErrors(cudaPeekAtLastError());
    return  ;
}


void gemv_moe_down_cu(const half * mat_data_, 
                const half * vec_data_,
                half * result_data_,
                const float * topk_weight,
                int32_t *moe_index,
                int ntopx,
                int hidden,
                int num_expert,
                int output_dim,
                unsigned int block_dim_x,
                unsigned int block_dim_y, 
                cudaStream_t stream){
    assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
    assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
    unsigned int num_per_thread = hidden / block_dim_x; 
    assert(num_per_thread >= 8);
    assert(num_per_thread % 4 == 0);

    dim3 grid_dim(1, output_dim / block_dim_y);
    dim3 block_dim(block_dim_x, block_dim_y);
    // moe down_data: [expert, output_dim, hidden_dim]
    // vec_data  :        [ ntopx ,    hidden_dim]  
    // // 不同的ntopx对应不同的expert，要加起来
    // output_data :      [ 1,  output_dim]


    assert ( ntopx <= 8);

    gemv_moe_down_fp16<8><<<grid_dim, block_dim, 1024, stream>>>
              (mat_data_, vec_data_, result_data_, topk_weight, 
                moe_index, ntopx, hidden, 
                num_expert, output_dim, num_per_thread);

    
    
    checkCudaErrors(cudaPeekAtLastError());
    return  ;
}



template< int max_expect>
__global__ void gemv_moe_down_bf16(const __nv_bfloat16* mat, 
                            const __nv_bfloat16* vec, 
                            __nv_bfloat16* res, 
                            const float* topk_weight,
                            const int32_t *moe_index,
                            int ntopx,
                            unsigned int hidden,
                            unsigned int num_expert, 
                            unsigned int output_dim, 
                            unsigned int num_per_thread) 
                          {

    
    // each thread load num_per_thread elements from global
    unsigned int tid = threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int start_idx = threadIdx.x;


    float sum[max_expect];


    #pragma unroll
    for (int topx = 0; topx < max_expect; topx++ ){
        sum[topx] = 0.0;
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif


    #pragma unroll
    for (int topx = 0; topx < ntopx; topx++ ){

        int target_expect = moe_index[topx];

    

        const __nv_bfloat16 * target_mat = mat +  target_expect * hidden * output_dim;
        const __nv_bfloat16 * target_vec = vec +  topx * hidden;
        
        const float4* mat4 = reinterpret_cast<const float4*>(target_mat);
        const float4* vec4 = reinterpret_cast<const float4*>(target_vec);


        bool noboundary = num_per_thread % 8 == 0;
        int loop = (num_per_thread >> 3);
        #pragma unroll
          for (int iter = 0; iter < loop; iter++) {
            unsigned int j = start_idx + iter * blockDim.x;
            if (j < hidden >> 3) {
                float4 vec_val = vec4[j];
                float4 mat_val = mat4[row * (hidden >> 3) + j];
                __nv_bfloat162* vec_h1 = (__nv_bfloat162*)&vec_val.x;
                __nv_bfloat162* vec_h2 = (__nv_bfloat162*)&vec_val.y;
                __nv_bfloat162* vec_h3 = (__nv_bfloat162*)&vec_val.z;
                __nv_bfloat162* vec_h4 = (__nv_bfloat162*)&vec_val.w;
                __nv_bfloat162* mat_h1 = (__nv_bfloat162*)&mat_val.x;
                __nv_bfloat162* mat_h2 = (__nv_bfloat162*)&mat_val.y;
                __nv_bfloat162* mat_h3 = (__nv_bfloat162*)&mat_val.z;
                __nv_bfloat162* mat_h4 = (__nv_bfloat162*)&mat_val.w;
                sum[topx] += __bfloat162float(vec_h1->x) * __bfloat162float(mat_h1->x);
                sum[topx] += __bfloat162float(vec_h1->y) * __bfloat162float(mat_h1->y);
                sum[topx] += __bfloat162float(vec_h2->x) * __bfloat162float(mat_h2->x);
                sum[topx] += __bfloat162float(vec_h2->y) * __bfloat162float(mat_h2->y);
                sum[topx] += __bfloat162float(vec_h3->x) * __bfloat162float(mat_h3->x);
                sum[topx] += __bfloat162float(vec_h3->y) * __bfloat162float(mat_h3->y);
                sum[topx] += __bfloat162float(vec_h4->x) * __bfloat162float(mat_h4->x);
                sum[topx] += __bfloat162float(vec_h4->y) * __bfloat162float(mat_h4->y);
            }
          }

        if (!noboundary ){
            unsigned int j = start_idx;
              const float4* vec4_now = vec4 + loop * blockDim.x;
              const float4 *mat4_now = mat4 + row * (hidden >> 3) + loop * blockDim.x;
              float2 vec_val =  reinterpret_cast<const float2*>(vec4_now)[j];
              float2 mat_val =  reinterpret_cast<const float2*>(mat4_now)[j];

              __nv_bfloat162* vec_h1 = (__nv_bfloat162*)&vec_val.x;
              __nv_bfloat162* vec_h2 = (__nv_bfloat162*)&vec_val.y;

              __nv_bfloat162* mat_h1 = (__nv_bfloat162*)&mat_val.x;
              __nv_bfloat162* mat_h2 = (__nv_bfloat162*)&mat_val.y;

              sum[topx] += __bfloat162float(vec_h1->x) * __bfloat162float(mat_h1->x);
              sum[topx] += __bfloat162float(vec_h1->y) * __bfloat162float(mat_h1->y);
              sum[topx] += __bfloat162float(vec_h2->x) * __bfloat162float(mat_h2->x);
              sum[topx] += __bfloat162float(vec_h2->y) * __bfloat162float(mat_h2->y);

            
            

        }
        sum[topx] = warpReduceSum(sum[topx], blockDim.x);


  }
   
  __syncthreads();

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      // res ntopx * output_dim
      float sum_all = 0.0;
      #pragma unroll
      for (int topx = 0; topx < ntopx; topx++ ){
          sum_all += sum[topx] * (topk_weight[topx]);
      }
      res[row] = __float2bfloat16(sum_all);
      

    }
   
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}



void gemv_moe_down_cu(const __nv_bfloat16 * mat_data_, 
                const __nv_bfloat16 * vec_data_,
                __nv_bfloat16 * result_data_,
                const float * topk_weight,
                int32_t *moe_index,
                int ntopx,
                int hidden,
                int num_expert,
                int output_dim,
                unsigned int block_dim_x,
                unsigned int block_dim_y, 
                cudaStream_t stream){
    assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
    assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
    unsigned int num_per_thread = hidden / block_dim_x; 
    assert(num_per_thread >= 8);

    dim3 grid_dim(1, output_dim / block_dim_y);
    dim3 block_dim(block_dim_x, block_dim_y);
    // moe down_data: [expert, output_dim, hidden_dim]
    // vec_data  :        [ ntopx ,    hidden_dim]  
    // // 不同的ntopx对应不同的expert，要加起来
    // output_data :      [ 1,  output_dim]


    assert ( ntopx <= 8);

    gemv_moe_down_bf16<8><<<grid_dim, block_dim, 1024, stream>>>
              (mat_data_, vec_data_, result_data_, topk_weight, 
                moe_index, ntopx, hidden, 
                num_expert, output_dim, num_per_thread);

    
    
    checkCudaErrors(cudaPeekAtLastError());
    return  ;
}