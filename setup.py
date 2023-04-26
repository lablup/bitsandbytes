# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import glob
import os
import subprocess
import sys

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py


class MakeBuild(build_py):
    def run(self):
        cuda_version = self._cuda_version
        major, minor = cuda_version.split(".")
        cuda_label = self._build_cuda_label(*cuda_version.split("."))
        if subprocess.call(["make", cuda_label], env={**os.environ, "CUDA_VERSION": cuda_version}) != 0:
            sys.exit(-1)
        super().run()

    def _build_cuda_label(self, major: str, minor: str) -> str:
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

    @property
    def _cuda_version(self) -> str:
        import torch

        return torch.version.cuda.replace(".", "")  # 117


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
    cmdclass={
        "build_py": MakeBuild,
    },
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
