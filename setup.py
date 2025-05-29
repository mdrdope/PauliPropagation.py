# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# """
# Setup script for compiling Cython extensions.
# """

# from setuptools import setup, Extension, find_packages
# from Cython.Build import cythonize
# import numpy as np

# # Define extensions
# extensions = [
#     Extension(
#         "pauli_pkg.pauli_propagation.pauli_gates_cy",
#         ["pauli_pkg/pauli_propagation/pauli_gates_cy.pyx"],
#         include_dirs=[np.get_include()],
#         define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
#     )
# ]

# # Setup configuration
# setup(
#     name="pauli_pkg",
#     version="0.1.0",
#     description="Pauli propagation package with Cython optimizations",
#     packages=find_packages(),
#     ext_modules=cythonize(
#         extensions,
#         compiler_directives={
#             'language_level': "3",
#             'boundscheck': False,
#             'wraparound': False, 
#             'initializedcheck': False,
#             'cdivision': True,
#         }
#     ),
#     include_dirs=[np.get_include()],
# ) 