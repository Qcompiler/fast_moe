#pragma once

#include<vector>
#include <torch/extension.h>

void gemv(int m, int n, int k, const torch::Tensor  &x,
                                  const torch::Tensor &weight, 
                                  torch::Tensor _out,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y );

void moe_gemv(const torch::Tensor  &x,
             torch::Tensor _out,
              const torch::Tensor &gate_up_weight, 
              const torch::Tensor &down_weight, 
                  const torch::Tensor &moe_index,   
                    unsigned int block_dim_x,
                    unsigned int block_dim_y);

void moe_gemv_down(const torch::Tensor  &x,
             torch::Tensor out,
              const torch::Tensor &down_weight, 
                  const torch::Tensor &topk_weight, 
                  const torch::Tensor &moe_index,   
                    unsigned int block_dim_x,
                    unsigned int block_dim_y);

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
                    int groupsize, int arch);


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
                  int groupsize, int arch);