

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




template< int max_expect>
__global__ void gemv_moe_down_bf16_i4(const __nv_bfloat16* mat, 
                            const __nv_bfloat16* vec, 
                            __nv_bfloat16* res, 
                            const float* topk_weight,
                            const __nv_bfloat16 *scales,
                            int group_size,
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
    return;
  }

}

void gemv_moe_down_cu_int4(
                const __nv_bfloat16 * mat_data_, 
                const __nv_bfloat16 * vec_data_,
                __nv_bfloat16 * result_data_,
                const float * topk_weight,
                const __nv_bfloat16 * scales,
                int group_size,
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

    gemv_moe_down_bf16_i4<8><<<grid_dim, block_dim, 1024, stream>>>
              (mat_data_, vec_data_, result_data_, topk_weight, 
                scales, group_size,
                moe_index, ntopx, hidden, 
                num_expert, output_dim, num_per_thread);

    
    
    checkCudaErrors(cudaPeekAtLastError());
    return  ;
}




template< int max_expect>
__global__ void gemv_moe_fp16_i4(const int32_t* mat, 
                            const half* vec, 
                            half* res, 
                            const int32_t *moe_index,
                            int ntopx,
                            unsigned int hidden,
                            unsigned int num_expert, 
                            unsigned int output_dim, 
                            unsigned int num_per_thread,
                            const half * scaling_data_,
                            int groupsize,
                            const half *outliers,
                            const int32_t *ind,
                            int n_outliers,
                            int num_outliers_per_thread) 
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

    int scale_num_each_n =  hidden / groupsize;


#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif



    #pragma unroll
    for (int topx = 0; topx < ntopx; topx++ ){

        int target_expect = moe_index[topx];

    

        const int32_t * target_mat = mat +  target_expect * ( hidden * output_dim / 8 ) ;
        const half * real_scaling = scaling_data_ + target_expect * output_dim * scale_num_each_n;
        const half * target_outliers = outliers + target_expect * output_dim * n_outliers;
        const int32_t * target_ind = ind + target_expect * n_outliers;

        const int* mat4 = reinterpret_cast<const int*>(target_mat);
        // const float4* mat4 = reinterpret_cast<const float4*>(target_mat);
        const float4* vec4 = reinterpret_cast<const float4*>(vec);
        

        #pragma unroll
          for (int iter = 0; iter < num_per_thread >> 3; iter++) {
            unsigned int j = start_idx + iter * blockDim.x;
            if (j < hidden >> 3) {
                float4 vec_val = vec4[j];
                // float4 mat_val = mat4[row * (hidden >> 3) + j];
                int b_quant = __ldg(mat4 + row * (  (hidden >> 3)   ) + j   ); 
                int b_quant_shift = b_quant >> 8;
                FragB frag_b0 = dequant(b_quant);
                FragB frag_b1 = dequant(b_quant_shift);

                half2* vec_h1 = (half2*)&vec_val.x;
                half2* vec_h2 = (half2*)&vec_val.y;
                half2* vec_h3 = (half2*)&vec_val.z;
                half2* vec_h4 = (half2*)&vec_val.w;

                half2* mat_h1 = (half2*)&frag_b0[0];
                half2* mat_h2 = (half2*)&frag_b0[1];
                half2* mat_h3 = (half2*)&frag_b1[0];
                half2* mat_h4 = (half2*)&frag_b1[1];

                float scales = (__half2float)(real_scaling[row * scale_num_each_n  + (j * 8) / groupsize]);


                sum[topx] += __half2float(vec_h1->x) * __half2float(mat_h1->x) * scales;
                sum[topx] += __half2float(vec_h1->y) * __half2float(mat_h1->y) * scales;
                sum[topx] += __half2float(vec_h2->x) * __half2float(mat_h2->x) * scales;
                sum[topx] += __half2float(vec_h2->y) * __half2float(mat_h2->y) * scales;
                sum[topx] += __half2float(vec_h3->x) * __half2float(mat_h3->x) * scales;
                sum[topx] += __half2float(vec_h3->y) * __half2float(mat_h3->y) * scales;
                sum[topx] += __half2float(vec_h4->x) * __half2float(mat_h4->x) * scales;
                sum[topx] += __half2float(vec_h4->y) * __half2float(mat_h4->y) * scales;
            }
          }

        
        
        const float* mat2_weight_fp16 = reinterpret_cast<const float*>(target_outliers);
        #pragma unroll
        for (int iter = 0; iter < num_outliers_per_thread >> 1; iter++) {
          unsigned int j = start_idx + iter * blockDim.x;
          if (j < n_outliers >> 1) {
            // float4 vec_val = vec4[j];
            float mat_val = mat2_weight_fp16[row * (n_outliers >> 1) + j];
            int ind1 = target_ind[ 2 * j];
            int ind2 = target_ind[ 2 * j + 1];


            half vec_h1 = vec[ind1];
            half vec_h2 = vec[ind2];

              
            half2* mat_h1 = (half2*)&mat_val;
          
            
            sum[topx] += __half2float(vec_h1) * __half2float(mat_h1->x);
            sum[topx] += __half2float(vec_h2) * __half2float(mat_h1->y);
        
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

  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
          

  
}




template< int max_expect, int column, int column_outliers>
__global__ void gemv_moe_fp16_i4_sm90(const int32_t* mat, 
                            const half* vec, 
                            half* res, 
                            const int32_t *moe_index,
                            int ntopx,
                            unsigned int hidden,
                            unsigned int num_expert, 
                            unsigned int output_dim, 
                            unsigned int num_per_thread,
                            const half * scaling_data_,
                            int groupsize,
                            const half *outliers,
                            const int32_t *ind,
                            int n_outliers,
                            int num_outliers_per_thread) 
                          {

    
    // each thread load num_per_thread elements from global
    unsigned int tid = threadIdx.x;
    unsigned int row = ( blockIdx.y * blockDim.y + threadIdx.y) ;
    unsigned int start_idx = threadIdx.x;

    float sum[max_expect];


    #pragma unroll
    for (int topx = 0; topx < max_expect; topx++ ){
        sum[topx] = 0.0;
    }

    int scale_num_each_n =  hidden / groupsize;




    #pragma unroll
    for (int topx = 0; topx < ntopx; topx++ ){

        int target_expect = moe_index[topx];

    

        const int32_t * target_mat = mat +  target_expect * ( hidden * output_dim / 8 ) ;
        const half * real_scaling = scaling_data_ + target_expect * output_dim * scale_num_each_n;
        const half * target_outliers = outliers + target_expect * output_dim * n_outliers;
        const int32_t * target_ind = ind + target_expect * n_outliers;

        const int* mat4 = reinterpret_cast<const int*>(target_mat);
        // const float4* mat4 = reinterpret_cast<const float4*>(target_mat);
        const float4* vec4 = reinterpret_cast<const float4*>(vec);
        
 
        #pragma unroll
          for (int iter = 0; iter < ( num_per_thread >> 3) / column; iter++) {
            unsigned int j = (start_idx + iter * blockDim.x);

            int b_quant[column];
            int b_quant_shift[column];
            FragB frag_b0[column]; 
            FragB frag_b1[column]; 
            float scales[column]; 
            float4 vec_val [column];
            const int element_per_iter = 8;
            if (j < hidden >> 3) {
                

                // float4 mat_val = mat4[row * (hidden >> 3) + j];
                #pragma unroll
                for (int col = 0; col < column; ++col)
                {
                    vec_val[col] = vec4[j * column + col];
                    b_quant[col] = __ldg(mat4 + row * (  (hidden >> 3)   ) + j * column + col  ); 
                    b_quant_shift[col] = b_quant[col] >> 8;
                    scales[col] = (__half2float)(real_scaling[row * scale_num_each_n  + (j * 8 * column + col) / groupsize]);
                    frag_b0[col] = dequant(b_quant[col]);
                    frag_b1[col] = dequant(b_quant_shift[col]);

                }

                #pragma unroll
                for (int col = 0; col < column; ++col){
                    half2* mat_h1 = (half2*)&frag_b0[col][0];
                    half2* mat_h2 = (half2*)&frag_b0[col][1];
                    half2* mat_h3 = (half2*)&frag_b1[col][0];
                    half2* mat_h4 = (half2*)&frag_b1[col][1];
                    half2* vec_h1 = (half2*)&vec_val[col].x;
                    half2* vec_h2 = (half2*)&vec_val[col].y;
                    half2* vec_h3 = (half2*)&vec_val[col].z;
                    half2* vec_h4 = (half2*)&vec_val[col].w;

                    float scales_ =  scales[col];

                    sum[topx] += __half2float(vec_h1->x) * __half2float(mat_h1->x) * scales_;
                    sum[topx] += __half2float(vec_h1->y) * __half2float(mat_h1->y) * scales_;
                    sum[topx] += __half2float(vec_h2->x) * __half2float(mat_h2->x) * scales_;
                    sum[topx] += __half2float(vec_h2->y) * __half2float(mat_h2->y) * scales_;
                    sum[topx] += __half2float(vec_h3->x) * __half2float(mat_h3->x) * scales_;
                    sum[topx] += __half2float(vec_h3->y) * __half2float(mat_h3->y) * scales_;
                    sum[topx] += __half2float(vec_h4->x) * __half2float(mat_h4->x) * scales_;
                    sum[topx] += __half2float(vec_h4->y) * __half2float(mat_h4->y) * scales_;
                }

             
            }
          }

   
        
        const float* mat2_weight_fp16 = reinterpret_cast<const float*>(target_outliers);
        #pragma unroll
        for (int iter = 0; iter < 1; iter++) {
          unsigned int j = start_idx + iter * blockDim.x;

          float mat_val[column_outliers];
          half vec_h1[column_outliers];

          half vec_h2[column_outliers];

          if (j < n_outliers >> 1) {
            // float4 vec_val = vec4[j];
            #pragma unroll
            for (int col = 0; col < column_outliers; ++col){

                mat_val[col] = mat2_weight_fp16[row * (n_outliers >> 1) + j * column_outliers + col];

 
                vec_h1[col]  = vec[ target_ind[ 2 * (j * column_outliers + col)]];
                vec_h2[col]  = vec[target_ind[ 2 * (j * column_outliers + col) + 1]];
            }






            #pragma unroll
            for (int col = 0; col < column_outliers; ++col){   
                half2* mat_h1 = (half2*)&mat_val[col];
          
                sum[topx] += __half2float(vec_h1[col]) * __half2float(mat_h1->x);
                sum[topx] += __half2float(vec_h2[col]) * __half2float(mat_h1->y);
           }
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

  }





          

  
}

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
                cudaStream_t stream, int arch)
        {

    assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
    assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
    unsigned int num_per_thread = hidden / block_dim_x; 
    unsigned int num_outliers_per_thread = n_outliers / block_dim_x; 
    assert(num_per_thread >= 8);
    assert(num_per_thread % 4 == 0);

  
    // moe gate_up_data: [expert, output_dim, hidden_dim]
    // vec_data  :        [1 ,    hidden_dim]
    // output_data :      [ ntopx,  output_dim]

    assert ( ntopx <= 8);
    
    if (arch < 90){
    
          dim3 grid_dim(1, (output_dim )/ block_dim_y);
          dim3 block_dim(block_dim_x, block_dim_y);
    
          gemv_moe_fp16_i4<8><<<grid_dim, block_dim, 1024, stream>>>
              (mat_data_, vec_data_, result_data_,
                moe_index, ntopx, hidden, 
                num_expert, output_dim, num_per_thread, 
                scaling_data_, groupsize,
                outliers, ind, n_outliers, num_outliers_per_thread);
        }
        else
  {
          dim3 grid_dim(1, (output_dim )/ (block_dim_y));
          dim3 block_dim(block_dim_x, block_dim_y);
          const int column_outliers = 2;

          // outliers = 128
          // thread = 32
          // each thread compute 4 outliers
          
          gemv_moe_fp16_i4_sm90<8, 4, 2><<<grid_dim, block_dim, 1024, stream>>>
            (mat_data_, vec_data_, result_data_,
                moe_index, ntopx, hidden, 
                num_expert, output_dim, num_per_thread, 
                scaling_data_, groupsize,
                outliers, ind, n_outliers, num_outliers_per_thread);
 
          // switch (hidden){
          //   case 2048:

          //       gemv_moe_fp16_i4_sm90<8, 4><<<grid_dim, block_dim, 1024, stream>>>
          //         (mat_data_, vec_data_, result_data_,
          //             moe_index, ntopx, hidden, 
          //             num_expert, output_dim, num_per_thread, 
          //             scaling_data_, groupsize,
          //             outliers, ind, n_outliers, num_outliers_per_thread);
          //   break;
          //   case 1024:

          //       gemv_moe_fp16_i4_sm90<8, 4><<<grid_dim, block_dim, 1024, stream>>>
          //         (mat_data_, vec_data_, result_data_,
          //             moe_index, ntopx, hidden, 
          //             num_expert, output_dim, num_per_thread, 
          //             scaling_data_, groupsize,
          //             outliers, ind, n_outliers, num_outliers_per_thread);
          //   break;
                        
          //   default:
              
          //     throw std::invalid_argument("Not implement error in gemv_moe_cu_i4 for sm 90 with hidden = " + std::to_string(hidden));
          // }

    }

    
    
    checkCudaErrors(cudaPeekAtLastError());
    return  ;
}








template< int max_expect>
__global__ void gemv_moe_down_fp16_i4(const int32_t* mat, 
                            const half* vec, 
                            half* res, 
                            const float* topk_weight,
                            const int32_t *moe_index,
                            int ntopx,
                            unsigned int hidden,
                            unsigned int num_expert, 
                            unsigned int output_dim, 
                            unsigned int num_per_thread,
                            const half * scaling_data_,
                            int groupsize,
                            const half *outliers,
                            const int32_t *ind,
                            int n_outliers,
                            int num_outliers_per_thread) 
                          {

    
    // each thread load num_per_thread elements from global
    unsigned int tid = threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int start_idx = threadIdx.x;


    float sum[max_expect];
    float sum_outliers[max_expect];


    #pragma unroll
    for (int topx = 0; topx < max_expect; topx++ ){
        sum[topx] = 0.0;
        sum_outliers[topx] = 0.0;
    }


    int scale_num_each_n =  hidden / groupsize;
    #pragma unroll
    for (int topx = 0; topx < ntopx; topx++ ){

        int target_expect = moe_index[topx];

        const int32_t * target_mat = mat +  target_expect * ( hidden * output_dim / 8 ) ;
        const half * real_scaling = scaling_data_ + target_expect * output_dim * scale_num_each_n;
        const half * target_outliers = outliers + target_expect * output_dim * n_outliers;
        const int32_t * target_ind = ind + target_expect * n_outliers;

    
        const half * target_vec = vec +  topx * hidden;
        
        const int* mat4 = reinterpret_cast<const int*>(target_mat);
        const float4* vec4 = reinterpret_cast<const float4*>(target_vec);


        // bool noboundary = num_per_thread % 8 == 0;
        int loop = (num_per_thread >> 3);
        #pragma unroll
          for (int iter = 0; iter < loop; iter++) {
            unsigned int j = start_idx + iter * blockDim.x;
            if (j < hidden >> 3) {
                float4 vec_val = vec4[j];
                // float4 mat_val = mat4[row * (hidden >> 3) + j];
                int b_quant = __ldg(mat4 + row * (  (hidden >> 3)   ) + j   ); 

                int b_quant_shift = b_quant >> 8;
                FragB frag_b0 = dequant(b_quant);
                FragB frag_b1 = dequant(b_quant_shift);

                half2* vec_h1 = (half2*)&vec_val.x;
                half2* vec_h2 = (half2*)&vec_val.y;
                half2* vec_h3 = (half2*)&vec_val.z;
                half2* vec_h4 = (half2*)&vec_val.w;

                
                half2* mat_h1 = (half2*)&frag_b0[0];
                half2* mat_h2 = (half2*)&frag_b0[1];
                half2* mat_h3 = (half2*)&frag_b1[0];
                half2* mat_h4 = (half2*)&frag_b1[1];

                float scales = (__half2float)(real_scaling[row * scale_num_each_n  + (j * 8) / groupsize]);
                // scales = 1.0;
                sum[topx] += __half2float(vec_h1->x) * __half2float(mat_h1->x) * scales;
                sum[topx] += __half2float(vec_h1->y) * __half2float(mat_h1->y) * scales;
                sum[topx] += __half2float(vec_h2->x) * __half2float(mat_h2->x) * scales;
                sum[topx] += __half2float(vec_h2->y) * __half2float(mat_h2->y) * scales;
                sum[topx] += __half2float(vec_h3->x) * __half2float(mat_h3->x) * scales;
                sum[topx] += __half2float(vec_h3->y) * __half2float(mat_h3->y) * scales;
                sum[topx] += __half2float(vec_h4->x) * __half2float(mat_h4->x) * scales;
                sum[topx] += __half2float(vec_h4->y) * __half2float(mat_h4->y) * scales;
            }
          }

        // const float* mat2_weight_fp16 = reinterpret_cast<const float*>(target_outliers);
        // #pragma unroll
        // for (int iter = 0; iter < num_outliers_per_thread >> 1; iter++) {
        //   unsigned int j = start_idx + iter * blockDim.x;
        //   if (j < n_outliers >> 1) {
        //     // float4 vec_val = vec4[j];
        //     float mat_val = mat2_weight_fp16[row * (n_outliers >> 1) + j];
        //     int ind1 = target_ind[ 2 * j];
        //     int ind2 = target_ind[ 2 * j + 1];

        //     half vec_h1 = target_vec[ind1];
        //     half vec_h2 = target_vec[ind2];
             
        //     half2* mat_h1 = (half2*)&mat_val;
          
        //     sum_outliers[topx] += __half2float(vec_h1) * __half2float(mat_h1->x);
        //     sum_outliers[topx] += __half2float(vec_h2) * __half2float(mat_h1->y);
        
        //   }
        // }
        sum_outliers[topx] = warpReduceSum(sum_outliers[topx], blockDim.x);
        sum[topx] = warpReduceSum(sum[topx], blockDim.x);


  }
   
  __syncthreads();

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      // res ntopx * output_dim
      float sum_all = 0.0;
      #pragma unroll
      for (int topx = 0; topx < ntopx; topx++ ){
          sum_all += (sum[topx] + sum_outliers[topx]) * (topk_weight[topx]);

      }
      res[row] = __float2half(sum_all);
      

    }
    return;
  }

}






template< int max_expect, int column, int column_outliers>
__global__ void gemv_moe_down_fp16_i4_sm90(const int32_t* mat, 
                            const half* vec, 
                            half* res, 
                            const float* topk_weight,
                            const int32_t *moe_index,
                            int ntopx,
                            unsigned int hidden,
                            unsigned int num_expert, 
                            unsigned int output_dim, 
                            unsigned int num_per_thread,
                            const half * scaling_data_,
                            int groupsize,
                            const half *outliers,
                            const int32_t *ind,
                            int n_outliers,
                            int num_outliers_per_thread) 
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


    int scale_num_each_n =  hidden / groupsize;
    #pragma unroll
    for (int topx = 0; topx < ntopx; topx++ ){

        int target_expect = moe_index[topx];

        const int32_t * target_mat = mat +  target_expect * ( hidden * output_dim / 8 ) ;
        const half * real_scaling = scaling_data_ + target_expect * output_dim * scale_num_each_n;
        const half * target_outliers = outliers + target_expect * output_dim * n_outliers;
        const int32_t * target_ind = ind + target_expect * n_outliers;

    
        const half * target_vec = vec +  topx * hidden;
        
        const int* mat4 = reinterpret_cast<const int*>(target_mat);
        const float4* vec4 = reinterpret_cast<const float4*>(target_vec);

    
        int nn = ( num_per_thread >> 3) / column;

        #pragma unroll
          for (int iter = 0; iter < ( num_per_thread >> 3) / column; iter++) {
            unsigned int j = (start_idx + iter * blockDim.x);

            int b_quant[column];
            int b_quant_shift[column];
            FragB frag_b0[column]; 
            FragB frag_b1[column]; 
            float scales[column]; 
            float4 vec_val [column];
            
            {
                

                // float4 mat_val = mat4[row * (hidden >> 3) + j];
                #pragma unroll
                for (int col = 0; col < column; ++col)
                {
                    vec_val[col] = vec4[ j * column + col ];
                    b_quant[col] = mat4[ row * (  (hidden >> 3)   ) + j * column + col  ]; 
                    b_quant_shift[col] = b_quant[col] >> 8;
                    scales[col] = (__half2float)(real_scaling[row * scale_num_each_n  + (j * 8 * column + col) / groupsize]);
                    frag_b0[col] = dequant(b_quant[col]);
                    frag_b1[col] = dequant(b_quant_shift[col]);

                }

                
                #pragma unroll
                for (int col = 0; col < column; ++col){

                    float2 mat_h1  = __half22float2(( (half2*)&frag_b0[col]) [0]);
                    float2 mat_h2  = __half22float2(( (half2*)&frag_b0[col]) [1]);
                    float2 mat_h3  = __half22float2(( (half2*)&frag_b1[col]) [0]);
                    float2 mat_h4  = __half22float2(( (half2*)&frag_b1[col]) [1]);

                    
                    half2* vec_h1 = (half2*)&vec_val[col].x;
                    half2* vec_h2 = (half2*)&vec_val[col].y;
                    half2* vec_h3 = (half2*)&vec_val[col].z;
                    half2* vec_h4 = (half2*)&vec_val[col].w;

                    float scales_ =  scales[col];

                    sum[topx] += __half2float(vec_h1->x) * mat_h1.x * scales_;
                    sum[topx] += __half2float(vec_h1->y) * mat_h1.y * scales_;
                    sum[topx] += __half2float(vec_h2->x) * (mat_h2.x) * scales_;
                    sum[topx] += __half2float(vec_h2->y) * (mat_h2.y) * scales_;
                    sum[topx] += __half2float(vec_h3->x) * (mat_h3.x) * scales_;
                    sum[topx] += __half2float(vec_h3->y) * (mat_h3.y) * scales_;
                    sum[topx] += __half2float(vec_h4->x) * (mat_h4.x) * scales_;
                    sum[topx] += __half2float(vec_h4->y) * (mat_h4.y) * scales_;
                }

             
            }
          }


        // const float* mat2_weight_fp16 = reinterpret_cast<const float*>(target_outliers);
        // #pragma unroll
        // for (int iter = 0; iter < 1; iter++) {
        //   unsigned int j = start_idx + iter * blockDim.x;

        //   float mat_val[column_outliers];
        //   half vec_h1[column_outliers];

        //   half vec_h2[column_outliers];

        //   if (j < n_outliers >> 1) {
        //     // float4 vec_val = vec4[j];
        //     #pragma unroll
        //     for (int col = 0; col < column_outliers; ++col){

        //         mat_val[col] = mat2_weight_fp16[row * (n_outliers >> 1) + j * column_outliers + col];
        //         vec_h1[col]  = target_vec[ target_ind[ 2 * (j * column_outliers + col)]];
        //         vec_h2[col]  = target_vec[ target_ind[ 2 * (j * column_outliers + col) + 1]];
        //     }

        //     #pragma unroll
        //     for (int col = 0; col < column_outliers; ++col){   
        //         half2* mat_h1 = (half2*)&mat_val[col];
          
        //         sum[topx] += __half2float(vec_h1[col]) * __half2float(mat_h1->x);
        //         sum[topx] += __half2float(vec_h2[col]) * __half2float(mat_h1->y);
        //    }
        //   }
        // }
        sum[topx] = warpReduceSum(sum[topx], blockDim.x);


  }
   
  __syncthreads();

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      // res ntopx * output_dim
      float sum_all = 0.0;
      #pragma unroll
      for (int topx = 0; topx < ntopx; topx++ ){
          sum_all += (sum[topx] ) * (topk_weight[topx]);
      }
      res[row] = __float2half(sum_all);
      

    }
    return;
  }

}
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
                cudaStream_t stream, int arch){

    assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
    assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
    unsigned int num_per_thread = hidden / block_dim_x; 
    assert(num_per_thread >= 8);


    unsigned int num_outliers_per_thread = n_outliers / block_dim_x; 


    // moe down_data: [expert, output_dim, hidden_dim]
    // vec_data  :        [ ntopx ,    hidden_dim]  
    // // 不同的ntopx对应不同的expert，要加起来
    // output_data :      [ 1,  output_dim]


    assert ( ntopx <= 8);





    if (arch < 90) {
        dim3 grid_dim(1, output_dim / block_dim_y);
        dim3 block_dim(block_dim_x, block_dim_y);
        gemv_moe_down_fp16_i4<8><<<grid_dim, block_dim, 8192, stream>>>
            (mat_data_, vec_data_, result_data_, topk_weight, 
              moe_index, ntopx, hidden, 
              num_expert, output_dim, num_per_thread, 
              scaling_data_, groupsize,
              outliers, ind, n_outliers,  num_outliers_per_thread);
    }else{
        dim3 grid_dim(1, output_dim / (block_dim_y) );
        dim3 block_dim(block_dim_x, block_dim_y);
        gemv_moe_down_fp16_i4_sm90<8, 8, 2><<<grid_dim, block_dim, 8192, stream>>>
            (mat_data_, vec_data_, result_data_, topk_weight, 
              moe_index, ntopx, hidden, 
              num_expert, output_dim, num_per_thread, 
              scaling_data_, groupsize,
              outliers, ind, n_outliers,  num_outliers_per_thread);
    }
 


    
    
    checkCudaErrors(cudaPeekAtLastError());
    return  ;
}


