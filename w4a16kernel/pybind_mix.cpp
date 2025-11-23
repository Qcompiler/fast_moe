#include <pybind11/pybind11.h>

#include<vector>
#include <torch/extension.h>
#include "entry.h"



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("gemv", &gemv,
        "gemv");
  
    m.def("moe_gemv", &moe_gemv,
        "moe_gemv");
  
    m.def("moe_gemv_down", &moe_gemv_down,
        "moe_gemv_down");

    m.def("moe_gemv_i4", &moe_gemv_i4,
        "moe_gemv_i4");
    m.def("moe_gemv_down_i4", &moe_gemv_down_i4,
        "moe_gemv_down_i4");
        
        
}