from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "bilateral_median",
        ["bilateral_median.pyx"],
        extra_compile_args=["/Ox", "/openmp","/MD","/fp:fast"],
        #extra_link_args=['/openmp'],
    )
]
setup(
    ext_modules = cythonize(ext_modules), 
    include_dirs=[numpy.get_include()]
)