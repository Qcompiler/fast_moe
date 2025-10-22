#include "jitcu/all.h"
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h> 
#include <sys/time.h>
#include <stdint.h>
#include <assert.h>
#include "common.h"

template <int NUM_WARP, int each_warp_reduce_compute>
__global__ void warp_specialized_gemv_kernel_down_split_expert(
    const int32_t* __restrict__ a,
    const half* __restrict__ x,
    half* __restrict__ y,
    float *topk_weight,
    const int32_t *moe_index,
    int ntopx,
    float* scales, int group_size,  int M, int K
) {
    extern __shared__ half shmem_vector[];
    float *shem_float = (float *) shmem_vector;
    
    int warp_id = threadIdx.y;

    // 每个线程负责4个元素，一个warp覆盖128个元素
    int tx = threadIdx.x;         // 0~31
    int ty = threadIdx.y;         // 0~3
    int bx = blockIdx.x;          // 0~M/4
    int lane = tx % WARP_SIZE;    // 0~31
    int m = (blockDim.y * bx + ty) / each_warp_reduce_compute;
  
    const int TOTAL_WARPS = 4; // 每个block中的warp数量

    

    const int NUM_PER_THREAD = 2; int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + NUM_PER_THREAD - 1) / NUM_PER_THREAD;

    const int max_expect = 8;
    float sum[max_expect];
     #pragma unroll
    for (int topx = 0; topx < max_expect; topx++ ){
        sum[topx] = 0.0;
    }


  int stride =  (ty) % each_warp_reduce_compute;  
  if (m < M) {
    
    for (int topx = 0; topx < ntopx / each_warp_reduce_compute; topx++ ){
      // 这里修复了没有同步导致的计算结果错误的问题
      // 问题的表现为：虽然固定了随机数的种子，但是tensor的计算结果还是会出现随机的错误
      // 并且大部分的计算结果是正确的

      int index = topx * each_warp_reduce_compute + warp_id % each_warp_reduce_compute;

      int target_expect = moe_index[index];


      const int32_t * target_mat = a +  target_expect * M * K;
      // 要写的vector的长度变成了 each_warp_reduce_compute * K

      int total_iterations = (K  * each_warp_reduce_compute) / 8;

 			  total_iterations *=  ((sizeof(int32_t) / sizeof(int8_t)) * 2);      int iterations_per_warp = (total_iterations + TOTAL_WARPS - 1) / TOTAL_WARPS;
      int start = warp_id * iterations_per_warp;

      int end = min((warp_id + 1) * iterations_per_warp, total_iterations);

      for (int i = start + lane; i < end; i += WARP_SIZE) {
          *(((int4 *)shmem_vector) + i) = *(((int4 *)x) + (topx * each_warp_reduce_compute )* ((K * 8) / 8) + i);
      }

    //  if ((blockIdx.x == 0) && (warp_id < each_warp_reduce_compute) && (threadIdx.x == 0))
    //     {
    //       for (int i = 0 ; i < 2; ++i)
    //       printf("warp = %d, shmem_vector[%d] =%.4f  index = %d\n", warp_id, i, __half2float((shmem_vector + (warp_id % each_warp_reduce_compute) * (K * 8))[i]), index);

    //     }

      __syncthreads();


            #pragma unroll  4
              for (int w = 0; w < NUM_WARPS; ++w) {

 			 float tmp = 0.0;
 			 float* scales_ptr =  (float *) (  (int4 *) scales + target_expect * ( ( M * K * 8) / ( 128 * 4 ) ) +  m * ( (K  * 8) / ( 128 * 4 )) ) ;                int k = (w * WARP_SIZE + lane) * NUM_PER_THREAD;
                half2 reg_x_0[4];*(((int4 *)reg_x_0)) =  *(((int4 *)( shmem_vector +  (warp_id % each_warp_reduce_compute) * (K * 8))) +  k);
                half2 reg_x_1[4];*(((int4 *)reg_x_1)) =  *(((int4 *)(shmem_vector + (warp_id % each_warp_reduce_compute) * (K * 8))) +  k + 1);

                
                uint2 reg = ld_cs_u32_v2((uint2*)& target_mat[m * K + k + 0]);
                half2 reg_a_0[4]; int reg_a_0_int = *(reinterpret_cast<int *>(&reg)  ); dequant(reg_a_0_int, reg_a_0);  
                half2 reg_a_1[4]; int reg_a_1_int = *(reinterpret_cast<int *>(&reg ) + 1 ); dequant(reg_a_1_int, reg_a_1); 
								 for (int kk = 0; kk < 4; ++kk) 
                tmp += (float(reg_x_0[kk].x) * float(reg_a_0[kk].x) + float(reg_x_0[kk].y) * float(reg_a_0[kk].y) + float( reg_x_1[kk].x) * float(reg_a_1[kk].x) + float(reg_x_1[kk].y) * float(reg_a_1[kk].y));
									 tmp *=   scales_ptr[  (k * 8 ) / 128];   sum[topx] +=  tmp;

              }

      sum[topx] = warp_reduce_sum_f32<WARP_SIZE>(sum[topx]);
    //  if ((blockIdx.x == 0) && (warp_id < each_warp_reduce_compute) && (threadIdx.x == 0))
    //     {
    //       printf("warp = %d, sum[%d] =%.4f  index = %d\n", warp_id, topx, sum[topx], index);

    //     }


     __syncthreads();           
    }


    if (lane == 0)
      {
        float sum_all = 0.0;
        #pragma unroll
        for (int topx = 0; topx < ntopx / each_warp_reduce_compute ; topx++ ){
            int index = topx * each_warp_reduce_compute + warp_id % each_warp_reduce_compute;;
            sum_all += sum[topx] * (topk_weight[index]);
        }
        
        // 写回shared 然后加起来
        shem_float[warp_id] = sum_all; 
        
      }
    __syncthreads();
    int reduced_warp_id = warp_id % each_warp_reduce_compute;
    float all = 0.0;
    if ((lane == 0) && (reduced_warp_id == 0) ){
        for (int i = 0 ; i < each_warp_reduce_compute; ++i)
            all += shem_float[warp_id + i];

        y[m] = (__float2half)( all );
    }
  }
}

__global__ void warp_specialized_gemv_kernel_down(
    const int32_t* __restrict__ a,
    const half* __restrict__ x,
    half* __restrict__ y,
    float *topk_weight,
    const int32_t *moe_index,
    int ntopx,
    float* scales, int group_size,  int M, int K
) {
    extern __shared__ half shmem_vector[];
    
    int warp_id = threadIdx.y;

    // 每个线程负责4个元素，一个warp覆盖128个元素
    int tx = threadIdx.x;         // 0~31
    int ty = threadIdx.y;         // 0~3
    int bx = blockIdx.x;          // 0~M/4
    int lane = tx % WARP_SIZE;    // 0~31
    int m = ( blockDim.y * bx + ty ) ; // (0~M/4) * 4 + (0~3)
  
    const int TOTAL_WARPS = 4; // 每个block中的warp数量
    int total_iterations = K / 8;

 			  total_iterations *=  ((sizeof(int32_t) / sizeof(int8_t)) * 2);    // 每个warp需要处理的迭代次数
    int iterations_per_warp = (total_iterations + TOTAL_WARPS - 1) / TOTAL_WARPS;
    
    // 计算当前warp的起始迭代索引
    int start = warp_id * iterations_per_warp;
    // 计算当前warp的结束迭代索引（不包含）
    int end = min((warp_id + 1) * iterations_per_warp, total_iterations);
    

    const int NUM_PER_THREAD = 2; int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + NUM_PER_THREAD - 1) / NUM_PER_THREAD;

    const int max_expect = 8;
    float sum[max_expect];
     #pragma unroll
    for (int topx = 0; topx < max_expect; topx++ ){
        sum[topx] = 0.0;
    }


    
  if (m < M) {
    
    for (int topx = 0; topx < ntopx; topx++ ){
      int target_expect = moe_index[topx];
      const int32_t * target_mat = a +  target_expect * M * K;
      // if (topx)
      //   __syncthreads();
      for (int i = start + lane; i < end; i += WARP_SIZE) {
          *(((int4 *)shmem_vector) + i) = *(  ((int4 *)x + topx * ((K * 8) / 8)) + i);
      }
      __syncthreads();

      
      #pragma unroll  4
        for (int w = 0; w < NUM_WARPS; ++w) {

 			 float tmp = 0.0;
 			 float* scales_ptr =  (float *) (  (int4 *) scales + target_expect * ( ( M * K * 8) / ( 128 * 4 ) ) +  m * ( (K  * 8) / ( 128 * 4 )) ) ;          int k = (w * WARP_SIZE + lane) * NUM_PER_THREAD;

          half2 reg_x_0[4];*(((int4 *)reg_x_0)) =  *(((int4 *)shmem_vector) +  k);
          half2 reg_x_1[4];*(((int4 *)reg_x_1)) =  *(((int4 *)shmem_vector) +  k + 1);
          
          uint2 reg = ld_cs_u32_v2((uint2*)& target_mat[m * K + k + 0]);
          half2 reg_a_0[4]; int reg_a_0_int = *(reinterpret_cast<int *>(&reg)  ); dequant(reg_a_0_int, reg_a_0);  
          half2 reg_a_1[4]; int reg_a_1_int = *(reinterpret_cast<int *>(&reg ) + 1 ); dequant(reg_a_1_int, reg_a_1); 
								 for (int kk = 0; kk < 4; ++kk) 
          tmp += (float(reg_x_0[kk].x) * float(reg_a_0[kk].x) + float(reg_x_0[kk].y) * float(reg_a_0[kk].y) + float( reg_x_1[kk].x) * float(reg_a_1[kk].x) + float(reg_x_1[kk].y) * float(reg_a_1[kk].y));
									 tmp *=   scales_ptr[  (k * 8 ) / 128];   sum[topx] +=  tmp;

        }


       sum[topx] = warp_reduce_sum_f32<WARP_SIZE>(sum[topx]);
       __syncthreads();
    }
    if (lane == 0)
      {
        float sum_all = 0.0;
        #pragma unroll
        for (int topx = 0; topx < ntopx; topx++ ){
            sum_all += sum[topx] * (topk_weight[topx]);
        }
        y[m] = (__float2half)( sum_all );
      }
  }
}

const uint32_t table[16] = {
    1, 1, 2, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 16,
};
void warp_specialized_gemv_down( const int32_t* d_A, const  half* d_B,
     half* d_C, float *topk_weight, const int32_t *moe_index, 
     int ntopx, float* scales, int group_size,  int M, int K, cudaStream_t stream, int kernel_type) {

    const int NUM_WARP = 8;
    dim3 block(32, NUM_WARP);
 
    dim3 grid;
    if (kernel_type == 0){
       grid = dim3((M + NUM_WARP - 1) / NUM_WARP, 1);
      int sharedMemSize = K *  sizeof(half) ; // Shared memory for A

 sharedMemSize = sharedMemSize * 8;
 assert(group_size == 128);    

      warp_specialized_gemv_kernel_down<<<grid, block, sharedMemSize, stream>>>( d_A, d_B,  d_C, topk_weight, moe_index, ntopx,  scales, group_size, M, K);

    }
    if (kernel_type == 1)
    {
       const int each_warp_reduce_compute = 4;
       grid = dim3((M + NUM_WARP - 1) / NUM_WARP  * table[each_warp_reduce_compute], 1);

       int sharedMemSize = K *  sizeof(half) * each_warp_reduce_compute; // Shared memory for A

 sharedMemSize = sharedMemSize * 8;
 assert(group_size == 128);        warp_specialized_gemv_kernel_down_split_expert<NUM_WARP, each_warp_reduce_compute><<<grid, block,  sharedMemSize, stream>>>( d_A, d_B,  d_C,  topk_weight, moe_index, ntopx,  scales, group_size, M, K);

    }


    
}

extern "C" {


void warp_specialized_gemv_down_host(cudaStream_t stream, const jc::Tensor& down, 
          const jc::Tensor& input, jc::Tensor& output,
          const jc::Tensor& topk_weight, const jc::Tensor& topk_ids, const jc::Tensor& scales, int group_size, int kernel_type) {

  // input ntopx, intermediate , K

  int output_dim = down.size(1);
  int hidden_size = down.size(2); 
  int ntopx = input.size(0);

  // printf("ntopx = %d", ntopx);
  // printf("hidden = %d", hidden_size);
  // printf("output = %d", output_dim);

  warp_specialized_gemv_down( down.data_ptr<int32_t>(), 
  input.data_ptr<half>(), output.data_ptr<half>(), topk_weight.data_ptr<float>(), 
  topk_ids.data_ptr<int32_t>(), ntopx, scales.data_ptr<float>(), group_size, output_dim, hidden_size, stream, kernel_type);

  CUDA_CHECK_KERNEL_LAUNCH();
}

}