#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// 确保定义 USE_ROCM 以使用内联汇编
#define USE_ROCM 1



__device__ __noinline__ void dequant(int *x, float * a) {


    half2* res1 = reinterpret_cast<half2*>(a);

    res1[0] += res1[1];

    a[0] = 1.0;
    return ;


}

__global__ void main_(float4* output,  int* input) {
    float x1[8];  // half2 实际上是两个 half
    
    dequant(input, x1);
    
    // 将 half 转换为 float
    output[0].x = (x1[0]);
    output[0].y = (x1[1]);
    output[0].z = (x1[2]);
    output[0].w = (x1[3]);

    output[1].x = (x1[0 + 4]);
    output[1].y = (x1[1 + 4]);
    output[1].z = (x1[2 + 4]);
    output[1].w = (x1[3 + 4]);

}



 


int main(){

    float4* output = nullptr;
    int* input= nullptr ;
 

    main_<<<1, 1>>>(output , input);
    return 0;
}