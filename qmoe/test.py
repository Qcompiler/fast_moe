import torch
from jitcu import load_cuda_ops
from triton.testing import do_bench_cudagraph,do_bench

"export PYTHONPATH=/home/chenyidong/newstart/bandwidth/jitcu"
code = r"""
#include "jitcu/all.h"

#include "cute/tensor.hpp"

using namespace cute;

template<typename T, int NumElemPerThread>
__global__ void vec_add_kernel(T* c, const T* a, const T* b, int64_t num) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num / NumElemPerThread) {
    return;
  }

  Tensor gc = make_tensor(c, make_shape(num));
  Tensor ga = make_tensor(a, make_shape(num));
  Tensor gb = make_tensor(b, make_shape(num));

  Tensor tc = local_tile(gc, make_shape(Int<NumElemPerThread>{}), make_coord(idx));
  Tensor ta = local_tile(ga, make_shape(Int<NumElemPerThread>{}), make_coord(idx));
  Tensor tb = local_tile(gb, make_shape(Int<NumElemPerThread>{}), make_coord(idx));

  Tensor tc_r = make_tensor_like(tc);
  Tensor ta_r = make_tensor_like(ta);
  Tensor tb_r = make_tensor_like(tb);

  copy(ta, ta_r);
  copy(tb, tb_r);

  #pragma unroll
  for (size_t i = 0; i < size(tc_r); ++i) {
    tc_r(i) = ta_r(i) + tb_r(i);
  }

  copy(tc_r, tc);
}

extern "C" {

void vec_add(cudaStream_t stream, jc::Tensor& c, const jc::Tensor& a, const jc::Tensor& b) {
  int64_t num = a.size(0);
  int64_t block_size = 1024;
  constexpr int NumElemPerThread = 4;
  assert(num % NumElemPerThread == 0);
  int64_t grid_size = ((num / NumElemPerThread) + block_size - 1) / block_size;

  using T = float;
  vec_add_kernel<T, NumElemPerThread><<<grid_size, block_size, 0, stream>>>(c.data_ptr<T>(), a.data_ptr<T>(), b.data_ptr<T>(), num);
  CUDA_CHECK_KERNEL_LAUNCH();
}

}

"""

lib = load_cuda_ops(
  name="vec_add",
  sources=code,
  func_names=["vec_add"],
  func_params=["t_t_t"],
  arches=["89", "90a", "120a"],
  extra_include_paths=["3rd/cutlass/include"],
  build_directory="./build",
)

device = torch.cuda.current_device()

# num = 1024 * 1024 * 1024
num = 1024 * 1024 * 16

for num in [2048*2048*1, 2048*2048*2, 2048*2048*4, 2048*2048*8, 2048*2048*16]:
  dtype = torch.float32

  a = torch.randn(num, dtype=dtype, device=device)
  b = torch.randn(num, dtype=dtype, device=device)
  c = torch.zeros_like(a)
  # print(c)
  lib.vec_add(c, a, b)
  # print(c)
  torch.cuda.synchronize()
  torch.testing.assert_close(c, a + b)


  torch.cuda.synchronize()
  with torch.cuda.stream(torch.cuda.Stream()):
    ms = do_bench(lambda: lib.vec_add(c, a, b))

  gb = sum(_.numel() * _.element_size() for _ in [a, b, c]) * 1e-9
  gb_s = gb / (ms * 1e-3)
  print(f"{gb_s=} GB/s, {gb} GB, {ms} ms")
