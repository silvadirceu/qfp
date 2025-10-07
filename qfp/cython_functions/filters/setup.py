from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

# Configuração das extensões
extensions = [
    Extension(
        "cython_filter",
        ["cython_filter.pyx"],
        include_dirs=[np.get_include()],  # Inclui headers do NumPy
        extra_compile_args=["-O3"],  # Otimizações
        language="c++"  # Use "c" se não precisar de C++
    )
]

setup(
    name='Cython Filter',
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'initializedcheck': False,
            'nonecheck': False
        },
    ),
    zip_safe=False
)
