import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

CUDA_HOME = "/usr/local/cuda"
sources = ["src/dummy.cpp", "src/dummy_cuda.cu"]
this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="Dummy-Collectives",
    version="0.0.1",
    ext_modules=[
        CUDAExtension(
            name="dummy_collectives",
            sources=sources,
            include_dirs=[
                os.path.join(this_dir, "include"),
                os.path.join(CUDA_HOME, "include"),
            ],
            library_dirs=[
                os.path.join(CUDA_HOME, "lib64"),
            ],
            libraries=["cudart", "nccl"],
            extra_compile_args={
                "cxx": ["-O2", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=1"],
                "nvcc": [
                    "-O2",
                    "-std=c++17",
                    "-arch=sm_80",
                    "-D_GLIBCXX_USE_CXX11_ABI=1",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
