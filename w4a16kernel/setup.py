import os
import torch
from pathlib import Path
from setuptools import setup, find_packages
from distutils.sysconfig import get_python_lib
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CUDAExtension

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

try:
    CUDA_VERSION = "".join(os.environ.get("CUDA_VERSION", torch.version.cuda).split("."))
except Exception as ex:
    raise RuntimeError("Your system must have an Nvidia GPU")

common_setup_kwargs = {
    "version": f"0.1.0+cu{CUDA_VERSION}",
    "name": "moe_gemm",
    "author": "Jidong Zhai; Yidong Chen; Tsinghua University",
    "license": "MIT",
    "python_requires": ">=3.8.0",
    "long_description_content_type": "text/markdown",
    "platforms": ["linux", "windows"],
    "classifiers": [
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: C++",
    ]
}

requirements = [

]

def get_include_dirs():
    include_dirs = []

    conda_cuda_include_dir = os.path.join(get_python_lib(), "nvidia/cuda_runtime/include")
    if os.path.isdir(conda_cuda_include_dir):
        include_dirs.append(conda_cuda_include_dir)

    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    include_dirs.append(this_dir)
    include_dirs.append(os.path.join(this_dir,"kernel"))


    return include_dirs

def get_generator_flag():
    generator_flag = []
    torch_dir = torch.__path__[0]
    if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
        generator_flag = ["-DOLD_GENERATOR_PATH"]
    
    return generator_flag

def check_dependencies():
    if CUDA_HOME is None:
        raise RuntimeError(
            f"Cannot find CUDA_HOME. CUDA must be available to build the package.")

def get_compute_capabilities():

    # figure out compute capability
    compute_capabilities = { 80, 89, 90 }

    capability_flags = []
    for cap in compute_capabilities:
        capability_flags += ["-gencode", f"arch=compute_{cap},code=sm_{cap}"]

    capability_flags += ["-gencode", f"arch=compute_90a,code=sm_90a"]
    return capability_flags


def get_library_dirs():
    include_dirs = []


    this_dir = os.path.dirname(os.path.abspath(__file__))
    include_dirs.append(this_dir)
    include_dirs.append(os.path.join(this_dir,"kernel/build"))  
    include_dirs.append(os.path.join(this_dir,"kernel/build/kernel"))   
    return include_dirs 

check_dependencies()
include_dirs = get_include_dirs()
generator_flags = get_generator_flag()
arch_flags = get_compute_capabilities()
library_dirs = get_library_dirs()
print("----library_dirs----")
print(library_dirs)

# 遇到问题，编译以后的cutlass的性能非常差~
extra_compile_args={
    "cxx": ["-g", "-O2","-std=c++20", "-fopenmp", "-lgomp", "-lcuda", "-DENABLE_BF16","-DNDEBUG"],
    "nvcc": [
        "-std=c++20",
        "-O2",
        "-DNDEBUG",
        "--expt-relaxed-constexpr",
        "-DENABLE_SCALED_MM_C3X=1",
        "-DENABLE_SCALED_MM_C2X=1",
        "-DCUTE_USE_PACKED_TUPLE=1",
        "-DCUTLASS_TEST_LEVEL=0",
        "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
        "-DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "-Xcompiler=-fPIC",
        "-lineinfo",
        "-DENABLE_FP8",
        "--threads=1",
        "-D_GLIBCXX_USE_CXX11_ABI=0"

    ] + arch_flags + generator_flags
}


cuda_extra = ['-lcudart', '-lcublas', '-lcurand', '-lcuda',   '-lmoekernel']
extensions = [
    CUDAExtension(
        "moe_gemm",
        [
            "pybind_mix.cpp",
            "entry.cu",
             
        ],   
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args = cuda_extra ,
    )
]



additional_setup_kwargs = {
    "ext_modules": extensions,
    "cmdclass": {'build_ext': BuildExtension}
}

common_setup_kwargs.update(additional_setup_kwargs)


setup(
    packages=find_packages(),
    install_requires=requirements,
    include_dirs=include_dirs,
    
    **common_setup_kwargs
)
