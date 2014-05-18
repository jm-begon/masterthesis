# -*- coding: utf-8 -*-
"""
Created on Sun May 04 13:02:09 2014

@author: Jm Begon
"""

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

extensions = [Extension("*", "*.pyx")]


setup(
    include_dirs = [np.get_include()],
    ext_modules = cythonize(extensions)
)




