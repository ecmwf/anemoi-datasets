# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os

from anemoi.utils.testing import get_test_data
from anemoi.utils.testing import skip_if_offline
from anemoi.utils.testing import skip_missing_packages
from anemoi.utils.testing import skip_slow_tests

from anemoi.datasets import open_dataset
from anemoi.datasets.create.testing import create_dataset


@skip_if_offline
def test_grib() -> None:
    """Test the creation of a dataset from GRIB files.

    This function tests the creation of a dataset using GRIB files from
    specific dates and verifies the shape of the resulting dataset.
    """
    data1 = get_test_data("anemoi-datasets/create/grib-20100101.grib")
    data2 = get_test_data("anemoi-datasets/create/grib-20100102.grib")
    assert os.path.dirname(data1) == os.path.dirname(data2)

    path = os.path.dirname(data1)

    config = {
        "dates": {
            "start": "2010-01-01T00:00:00",
            "end": "2010-01-02T18:00:00",
            "frequency": "6h",
        },
        "input": {
            "grib": {
                "path": os.path.join(path, "grib-{date:strftime(%Y%m%d)}.grib"),
            },
        },
    }

    created = create_dataset(config=config, output=None)
    ds = open_dataset(created)
    assert ds.shape == (8, 12, 1, 162)


@skip_if_offline
def test_netcdf() -> None:
    """Test for NetCDF files.

    This function tests the creation of a dataset from a NetCDF file.
    """
    data = get_test_data("anemoi-datasets/create/netcdf.nc")
    config = {
        "dates": {
            "start": "2023-01-01",
            "end": "2023-01-02",
            "frequency": "1d",
        },
        "input": {
            "netcdf": {"path": data},
        },
    }

    created = create_dataset(config=config, output=None)
    ds = open_dataset(created)
    assert ds.shape == (2, 2, 1, 162)


@skip_missing_packages("fstd", "rpnpy.librmn")
def test_eccs_fstd() -> None:
    """Test for 'fstd' files from ECCC."""
    # See https://github.com/neishm/fstd2nc

    data = get_test_data("anemoi-datasets/create/2025031000_000_TT.fstd", gzipped=True)
    config = {
        "dates": {
            "start": "2023-01-01",
            "end": "2023-01-02",
            "frequency": "1d",
        },
        "input": {
            "eccc_fstd": {"path": data},
        },
    }

    created = create_dataset(config=config, output=None)
    ds = open_dataset(created)
    assert ds.shape == (2, 2, 1, 162)


@skip_slow_tests
@skip_if_offline
@skip_missing_packages("kerchunk", "s3fs")
def test_kerchunk() -> None:
    """Test for Kerchunk JSON files.

    This function tests the creation of a dataset from a Kerchunk JSON file.

    """
    # Note: last version of kerchunk compatible with zarr 2 is 0.2.7

    data = get_test_data("anemoi-datasets/create/kerchunck.json", gzipped=True)

    config = {
        "dates": {
            "start": "2024-03-01T00:00:00",
            "end": "2024-03-01T18:00:00",
            "frequency": "6h",
        },
        "input": {
            "xarray-kerchunk": {
                "json": data,
                "param": ["T"],
                "level": [1000],
            },
        },
    }

    created = create_dataset(config=config, output=None)
    ds = open_dataset(created)
    assert ds.shape == (4, 1, 1, 1038240)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_kerchunk()
    exit()
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
