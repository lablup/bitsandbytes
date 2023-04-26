# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import ctypes
import glob
import os
import subprocess
import sys

from setuptools import find_packages, setup

"""
CUDA_VERSION=117 make cuda11x
"""
def build_cuda_label(major: str, minor: str) -> str:
    postfix = ""
    if int(major) < 7 or (int(major) == 7 and int(minor) < 5):
        postfix = "_nomatmul"
    cuda_version = major + minor
    if len(cuda_version) < 3:
        return "cuda92" + postfix
    if cuda_version == "110":
        return "cuda110" + postfix
    if cuda_version.startswith("11"):
        return "cuda11x" + postfix
    return "cpu"


libcudart = ctypes.CDLL("libcudart.so")

version = ctypes.c_int()
err = libcudart.cudaRuntimeGetVersion(ctypes.byref(version))
print(f"cudart.cudaRuntimeGetVersion(): {version.value} (Error: {err})")

major = str(version.value // 1000)
minor = str((version.value % 1000) // 10)

cuda_label = build_cuda_label(major, minor)
cuda_version = major + minor
print(f"CUDA_VERSION={cuda_version} make {cuda_label}")
if subprocess.call(["make", cuda_label], env={**os.environ, "CUDA_VERSION": cuda_version}) != 0:
    sys.exit(-1)

libs = list(glob.glob("./bitsandbytes/libbitsandbytes*.so"))
libs = [os.path.basename(p) for p in libs]
print("libs:", libs)


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name=f"bitsandbytes",
    version=f"0.38.0",
    author="Tim Dettmers",
    author_email="dettmers@cs.washington.edu",
    description="8-bit optimizers and matrix multiplication routines.",
    license="MIT",
    keywords="gpu optimizers optimization 8-bit quantization compression",
    url="https://github.com/TimDettmers/bitsandbytes",
    packages=find_packages(),
    package_data={"": libs},
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
