#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// 确保定义 USE_ROCM 以使用内联汇编
#define USE_ROCM 1

__device__ __forceinline__ uint32_t bfi(const uint32_t S0, const uint32_t S1,
                                        const uint32_t S2) {
#if defined(USE_ROCM)
  uint32_t result;
  __asm__ volatile(
    "v_bfi_b32 %0, %1, %2, %3"
    : "=v"(result)
    : "v"(S0), "v"(S1), "v"(S2)
  );
  return result;
#else
  return (S0 & S1) | (~S0 & S2);
#endif
}

__device__ __noinline__ void dequant(int q, half* x1) {
    const int LO = 0x000f000f;
    const int HI = 0x00f000f0;
    const int EX = 0x64006400;
    const int SUB = 0x64006400;
    const int MUL = 0x2c002c00;
    const int ADD = 0xd480d480;

    int lo0 = bfi(LO, q, EX);
    int hi0 = bfi(HI, q, EX);
    int p = (q >> 8);
    int lo1 = bfi(LO, p, EX);
    int hi1 = bfi(HI, p, EX);

    half2* res1 = reinterpret_cast<half2*>(x1);


    res1[0] = __hsub2(*reinterpret_cast<half2*>(&lo0),
                     *reinterpret_cast<const half2*>(&SUB));
    res1[1] = __hfma2(*reinterpret_cast<half2*>(&hi0),
                     *reinterpret_cast<const half2*>(&MUL),
                     *reinterpret_cast<const half2*>(&ADD));
    res1[2] = __hsub2(*reinterpret_cast<half2*>(&lo1),
                     *reinterpret_cast<const half2*>(&SUB));
    res1[3] = __hfma2(*reinterpret_cast<half2*>(&hi1),
                     *reinterpret_cast<const half2*>(&MUL),
                     *reinterpret_cast<const half2*>(&ADD));

    // res1[0].x = res1[0].x + res1[0].y + res1[1].x + res1[1].y + res1[2].x + res1[2].y  + res1[3].x + res1[3].y ;


}

__global__ void main_(float4* output, const int* input) {
    half x1[8];  // half2 实际上是两个 half
    
    dequant(input[0], x1);
    
    // 将 half 转换为 float
    output[0].x = __half2float(x1[0]);
    output[0].y = __half2float(x1[1]);
    output[0].z = __half2float(x1[2]);
    output[0].w = __half2float(x1[3]);

    output[1].x = __half2float(x1[0 + 4]);
    output[1].y = __half2float(x1[1 + 4]);
    output[1].z = __half2float(x1[2 + 4]);
    output[1].w = __half2float(x1[3 + 4]);

}



 


int main(){

    float4* output = nullptr;
    int* input= nullptr ;
 

    main_<<<1, 1>>>(output , input);
    return 0;
}