from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension



setup(
    # package name
    name='variable_gather',
    ext_modules=[
        CUDAExtension('variable_gather', [
            'gather_index.cpp',
            'gather_index_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension()
    }
)