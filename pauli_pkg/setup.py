
from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy

setup(
    name="pauli-propagation",
    version="0.1.0",
    packages=find_packages(),                          
    ext_modules=cythonize(
        "pauli_propagation/su4_gate_cy.pyx",           
        language_level="3"
    ),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
