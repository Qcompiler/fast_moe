#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDABlas.h>
#include <c10/util/BFloat16.h>
#include "kernel.h"



void gemv(int m, int n, int k, const torch::Tensor  &x,
                                  const torch::Tensor &weight, 
                                  torch::Tensor _out,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y ){
                      

  half * vec_data_ = reinterpret_cast<half*>(x.data_ptr<at::Half>());                                  
  half * mat_data_ = reinterpret_cast<half*>(weight.data_ptr<at::Half>());
  
  half * result_data_ = reinterpret_cast<half*>(_out.data_ptr<at::Half>());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  gemv_cu(m, n, k, vec_data_, mat_data_, result_data_, block_dim_x, block_dim_y, stream);
  return  ;
}



void moe_gemv(const torch::Tensor  &x,
             torch::Tensor out,
              const torch::Tensor &gate_up_weight, 
              const torch::Tensor &down_weight, 
                  const torch::Tensor &moe_index,   
                    unsigned int block_dim_x,
                    unsigned int block_dim_y){

    

    
    int m = 1;
    int hidden = gate_up_weight.size(2);
    int num_expert = gate_up_weight.size(0);
    int itermediate_output_dim = gate_up_weight.size(1);
    assert ( down_weight.size(0) == num_expert );
    
    int ntopx = moe_index.size(1);
    int32_t * moe_index_ = reinterpret_cast<int32_t*>(moe_index.data_ptr<int32_t>() );


    if (out.dtype() == torch::kBFloat16){

        const __nv_bfloat16 * mat_data_ = reinterpret_cast<const __nv_bfloat16*>(gate_up_weight.data_ptr()); 
        const __nv_bfloat16 * vec_data_ = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
        __nv_bfloat16 * result_data_ = reinterpret_cast< __nv_bfloat16*>(out.data_ptr()); 
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                  
        gemv_moe_cu( mat_data_,  vec_data_,
                    result_data_,
                    moe_index_,
                    ntopx,
                    hidden,
                    num_expert,
                    itermediate_output_dim,
                    block_dim_x,
                    block_dim_y, 
                    stream);

    }
    else
    {
      const half * mat_data_ = reinterpret_cast<const half*>(gate_up_weight.data_ptr<at::Half>()); 
      const half * vec_data_ = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
      half * result_data_ = reinterpret_cast< half*>(out.data_ptr<at::Half>()); 
      cudaStream_t stream = at::cuda::getCurrentCUDAStream();                  
      gemv_moe_cu( mat_data_,  vec_data_,
                  result_data_,
                  moe_index_,
                  ntopx,
                  hidden,
                  num_expert,
                  itermediate_output_dim,
                  block_dim_x,
                  block_dim_y, 
                  stream);
    }
    return  ;

}


void moe_gemv_i4(int n, int k, const torch::Tensor  &x,
             torch::Tensor out,
              const torch::Tensor &gate_up_weight, 
              const torch::Tensor &down_weight, 
                  const torch::Tensor &moe_index,   
                    unsigned int block_dim_x,
                    unsigned int block_dim_y,
                    const torch::Tensor &scaling,
                    const torch::Tensor &outliers,
                    const torch::Tensor &ind,
                    int n_outliers,
                    int groupsize, int arch){

    

    
    int m = 1;
    int hidden = k;
    int num_expert = gate_up_weight.size(0);
    int itermediate_output_dim = n;
    assert ( down_weight.size(0) == num_expert );
    
    int ntopx = moe_index.size(1);
    int32_t * moe_index_ = reinterpret_cast<int32_t*>(moe_index.data_ptr<int32_t>() );

    
    const int32_t * mat_data_ = reinterpret_cast<const int32_t*>(gate_up_weight.data_ptr<int32_t>()); 
    const half * vec_data_ = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    half * result_data_ = reinterpret_cast< half*>(out.data_ptr<at::Half>()); 
    const half * scaling_data_ = reinterpret_cast<const half*>(scaling.data_ptr<at::Half>());
    const half * outliers_data_ = reinterpret_cast<const half*>(outliers.data_ptr<at::Half>());
    const int32_t * ind_data_ = reinterpret_cast<const int32_t*>(ind.data_ptr<int32_t>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();                  
    gemv_moe_cu_i4(mat_data_, 
                vec_data_,
                result_data_,
                moe_index_,
                ntopx,
                hidden,
                num_expert,
                itermediate_output_dim,
                block_dim_x,
                block_dim_y, 
                scaling_data_,
                groupsize,
                outliers_data_,
                ind_data_,
                n_outliers,
                stream, arch);
  
    return  ;

}





void moe_gemv_down(const torch::Tensor  &x,
             torch::Tensor out,
              const torch::Tensor &down_weight, 
                  const torch::Tensor &topk_weight, 
                  const torch::Tensor &moe_index,   
                    unsigned int block_dim_x,
                    unsigned int block_dim_y){

    
    int m = 1;
    int hidden = down_weight.size(2);
    int num_expert = down_weight.size(0);
    int output_dim = down_weight.size(1);
    

    
    int ntopx = moe_index.size(1);
    int32_t * moe_index_ = reinterpret_cast<int32_t*>(moe_index.data_ptr<int32_t>() );
    const float * topk_weight_data = reinterpret_cast<const float*>(topk_weight.data_ptr<float>());                  

    if (out.dtype() == torch::kBFloat16){

        const __nv_bfloat16 * mat_data_ = reinterpret_cast<const __nv_bfloat16*>(down_weight.data_ptr()); 
        const __nv_bfloat16 * vec_data_ = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
        
        __nv_bfloat16 * result_data_ = reinterpret_cast< __nv_bfloat16*>(out.data_ptr()); 
        
        
        

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                  
        gemv_moe_down_cu( mat_data_,  vec_data_,
                    result_data_,
                    topk_weight_data,
                    moe_index_,
                    ntopx,
                    hidden,
                    num_expert,
                    output_dim,
                    block_dim_x,
                    block_dim_y, 
                    stream);
    }
    else{

        const half * mat_data_ = reinterpret_cast<const half*>(down_weight.data_ptr<at::Half>()); 
        const half * vec_data_ = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
        half * result_data_ = reinterpret_cast< half*>(out.data_ptr<at::Half>()); 
        
        
        

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                  
        gemv_moe_down_cu( mat_data_,  vec_data_,
                    result_data_,
                    topk_weight_data,
                    moe_index_,
                    ntopx,
                    hidden,
                    num_expert,
                    output_dim,
                    block_dim_x,
                    block_dim_y, 
                    stream);
    }
    return  ;

}




void moe_gemv_down_i4(int n, int k, const torch::Tensor  &x,
             torch::Tensor out,
              const torch::Tensor &down_weight, 
                  const torch::Tensor &topk_weight, 
                  const torch::Tensor &moe_index,   
                    unsigned int block_dim_x,
                    unsigned int block_dim_y,
                    const torch::Tensor &scaling,
                    const torch::Tensor &outliers,
                    const torch::Tensor &ind,
                    int n_outliers,
                  int groupsize, int arch){

    
      int m = 1;
      int hidden = k;
      int num_expert = down_weight.size(0);
      int output_dim = n;
      

    
      int ntopx = moe_index.size(1);
      int32_t * moe_index_ = reinterpret_cast<int32_t*>(moe_index.data_ptr<int32_t>() );
      const float * topk_weight_data = reinterpret_cast<const float*>(topk_weight.data_ptr<float>());                  

      const int32_t * mat_data_ = reinterpret_cast<const int32_t*>(down_weight.data_ptr<int32_t>()); 
      const half * vec_data_ = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
      half * result_data_ = reinterpret_cast< half*>(out.data_ptr<at::Half>()); 
      const half * scaling_data_ = reinterpret_cast<const half*>(scaling.data_ptr<at::Half>());
      const half * outliers_data_ = reinterpret_cast<const half*>(outliers.data_ptr<at::Half>());
      const int32_t * ind_data_ = reinterpret_cast<const int32_t*>(ind.data_ptr<int32_t>());
      
      
      

      cudaStream_t stream = at::cuda::getCurrentCUDAStream();                  
      gemv_moe_down_cu_i4(
                mat_data_, 
                vec_data_,
                result_data_,
                topk_weight_data,
                moe_index_,
                ntopx,
                hidden,
                num_expert,
                output_dim,
                block_dim_x,
                block_dim_y, 
                scaling_data_,
                groupsize,
                outliers_data_,
                ind_data_,
                n_outliers,
                stream, arch);

   
           

    
    return  ;

}


