from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension("lsss._lsss", ["lsss/_lsss.pyx", "src/lsss.cpp"],
        include_dirs=['include'],
        extra_compile_args=['-std=c++11', '-pthread', '-O2'],
        language='c++')
]
setup(
    name="lsss",
    ext_modules=cythonize(extensions),
)
