import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy

sources = [
    os.path.join('cpp', 'matrix_metal.cpp'),
    os.path.join('cpp', 'matmul.mm'),
    '_matrix_metal.cpp',
]

extra_compile_args = ['-std=c++17']
if sys.platform == 'darwin':
    extra_link_args = ['-framework', 'Metal', '-framework', 'Foundation']
else:
    extra_link_args = []

class custom_build_ext(build_ext):
    def build_extensions(self):
        if sys.platform == 'darwin':
            for ext in self.extensions:
                for i, src in enumerate(ext.sources):
                    if src.endswith('.mm'):
                        # Use clang++ for Objective-C++
                        self.compiler.src_extensions.append('.mm')
                        # Set the compiler for .mm files
                        original_compile = self.compiler._compile
                        def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
                            if src.endswith('.mm'):
                                self.compiler.set_executable('compiler_so', 'clang++')
                            return original_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
                        self.compiler._compile = _compile
        super().build_extensions()

ext_modules = [
    Extension(
        'matrix_metal._matrix_metal',
        sources=[os.path.join('matrix_metal', s) for s in sources],
        include_dirs=[numpy.get_include(), os.path.join('matrix_metal', 'cpp')],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name='matrix_metal',
    version='0.1.0',
    description='Python library for Metal-accelerated matrix multiplication',
    author='Your Name',
    packages=['matrix_metal', 'matrix_metal.cpp'],
    # install_requires=['numpy'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': custom_build_ext},
)
