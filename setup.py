#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "pyarrow",
    "dask[dataframe]",
    "dill",
    "pydantic",
    "polars",
    "sortedcontainers",
    "datatable>=1.1.0",
    "arviz",
    "pymc>=5.0",
    "icecream>=2.1.3",
    "loguru>=0.7.2",
    "matplotlib",
    "numpy",
    "pandas>=2.2.2",
    "scikit_learn",
    "scipy",
    "seaborn>=0.13.2",
    "setuptools>69.1.1",
    "statsmodels>=0.14.1",
    "torch",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Rohit Ghosh",
    author_email="rohit.ghosh@yale.edu",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Analysis of single-cell Massively Parallel Reporter Assay (MPRA) data",
    entry_points={
        "console_scripts": [
            "glmpy=glmpy.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    package_data={
        "glmpy": ["data/*.mtx"],
    },
    keywords="glmpy",
    name="glmpy",
    packages=find_packages(include=["glmpy", "glmpy.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/rghosh670/glmpy",
    version="0.1.0",
    zip_safe=False,
)
