# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os

from anemoi.utils.testing import get_test_data

from anemoi.datasets import open_dataset
from anemoi.datasets.create.testing import create_dataset


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


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
