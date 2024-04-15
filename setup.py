#!/usr/bin/env python
# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import io
import os

import setuptools


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return io.open(file_path, encoding="utf-8").read()


version = None
for line in read("anemoi/datasets/__init__.py").split("\n"):
    if line.startswith("__version__"):
        version = line.split("=")[-1].strip()[1:-1]


assert version


data_requires = [
    "anemoi-utils[provenance]",
    "zarr",
    "pyyaml",
    "numpy",
    "tqdm",
    "semantic-version",
]

remote_requires = [
    "boto3",
    "requests",
    "s3fs",  # prepml copy only
]


create_requires = [
    "zarr",
    "numpy",
    "tqdm",
    "climetlab",  # "earthkit-data"
    "earthkit-meteo",
    "pyproj",
    "ecmwflibs>=0.6.3",
]


all_requires = data_requires + create_requires + remote_requires
dev_requires = ["sphinx", "sphinx_rtd_theme", "nbsphinx", "pandoc"] + all_requires

setuptools.setup(
    name="anemoi-datasets",
    version=version,
    description="A package to hold various functions to support training of ML models on ECMWF data.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="European Centre for Medium-Range Weather Forecasts (ECMWF)",
    author_email="software.support@ecmwf.int",
    license="Apache License Version 2.0",
    url="https://github.com/ecmwf/anemoi-datasets",
    packages=setuptools.find_namespace_packages(include=["anemoi.*"]),
    include_package_data=True,
    install_requires=data_requires,
    extras_require={
        "data": [],
        "remote": data_requires + remote_requires,
        "create": create_requires,
        "dev": dev_requires,
        "all": all_requires,
    },
    zip_safe=True,
    keywords="tool",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Operating System :: OS Independent",
    ],
    entry_points={"console_scripts": ["anemoi-datasets=anemoi.datasets.__main__:main"]},
)
