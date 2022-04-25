#setup.py
from distutils.core import setup, Extension, DEBUG
import numpy

module = Extension('dag_completion_time', sources = ['cpplib.cpp'])

setup(name = 'dag_completion_time',
      version = '1.0',
      description = 'Python Package with super fast code C++ extension',
      ext_modules = [module],
      include_dirs = [numpy.get_include()],
)

# python3 build_script.py build_ext --inplace