"""
AutoTritonOps - Triton-Ascend Operator Implementation

Setup script for building and distributing the package.
"""

from setuptools import setup, find_packages
import os

# Read version from pyproject.toml
version = "1.0.0"

# Read README for long description
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="autotritonops",
    version=version,
    description="High-performance operators implemented with Triton-Ascend for NPU acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AutoTritonOps Team",
    author_email="support@example.com",
    url="https://gitcode.com/aflyingto/AutoTritonOps",
    project_urls={
        "Bug Tracker": "https://gitcode.com/aflyingto/AutoTritonOps/issues",
        "Documentation": "https://gitcode.com/aflyingto/AutoTritonOps/blob/main/README.md",
        "Source Code": "https://gitcode.com/aflyingto/AutoTritonOps",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "bandit>=1.7.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.15.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "triton",
        "ascend",
        "npu",
        "vector-add",
        "softmax",
        "layer-norm",
        "flash-attention",
        "matmul",
        "deep-learning",
        "acceleration",
    ],
    license="Apache-2.0",
    include_package_data=True,
    zip_safe=False,
)
