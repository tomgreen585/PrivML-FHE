# PySEAL Machine Learning Library Research

This repository explores the capabilities of the [`PySEAL`](https://github.com/Lab41/PySEAL) library — a privacy-preserving machine learning framework using Fully Homomorphic Encryption (FHE). The goal is to evaluate performance and feasibility of encrypted inference using traditional ML models.

## Objectives

- Understand the features and limitations of the `PySEAL` library.
- Implement homomorphic encryption examples

## Getting Started

### 1. Prerequisites

- Python 3.8 to 3.11

### 2. Installation for MacOS

```bash
# Clone PySEAL directory
git clone https://github.com/Lab41/PySEAL
cd PySEAL

# Fix configure script line endings
cd SEAL
sed -i '' $'s/\r$//' configure
./configure

# Path the GCC version check for macOS/Clang
nano seal/util/defines.h # Open the file

## Replace:
#if defined(__GNUC__) && (__GNUC__ < 5)
#error "SEAL requires __GNUC__ >= 5"
#endif

## With:
#if defined(__GNUC__) && !defined(__clang__)
#if (__GNUC__ < 5)
#error "SEAL requires __GNUC__ >= 5"
#endif
#endif

# Compile the SEAL static library which creates ../bin/libseal.a
make clean
make -j$(sysctl -n hw.logicalcpu)

# Edit SEALPython/setup.py to:
# from distutils.core import setup, Extension
# import pybind11
# import sysconfig

# cfg_vars = sysconfig.get_config_vars()
# for key, value in cfg_vars.items():
#     if isinstance(value, str):
#         cfg_vars[key] = value.replace('-Wstrict-prototypes', '')

# ext_modules = [
#     Extension(
#         'seal',
#         ['wrapper.cpp'],
#         include_dirs=[
#             pybind11.get_include(),
#             '../SEAL'  # <- NOT ../SEAL/seal
#         ],
#         extra_objects=[
#             '../bin/libseal.a' #location of the libseal.a 
#         ],
#         language='c++',
#         extra_compile_args=['-std=c++11']
#     )
# ]

# setup(
#     name='seal',
#     version='2.3',
#     author='Todd Stavish, Shashwat Kishore',
#     description='Python wrapper for SEAL',
#     ext_modules=ext_modules,
# )

# Install dependencies
pip install pybind11

# Install PySEAL
pip install . --no-build-isolation --no-cache-dir
```

### 3. Directory structure

- pyseal_ciphertext_implementation.ipynb
- pyseal_function_implementations.ipynb
- README.md

### 4. TODO

- [x] Observe functions provided by the PySEAL library and how they are used to deploy FHE
- [x] Document FHE-related constraints

### 5. Issues with Models

- Library is mainly used for deep FHE deployment and does not cross over well with ML deployment. Functions provided by the library do serve most FHE use cases and may be beneficial for final model. Not going to provide the 'exploratory' dynamic of this project, so have priotised development to use TenSEAL or Concrete-ML.
