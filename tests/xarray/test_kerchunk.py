# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import xarray as xr

from anemoi.datasets.create.functions.sources.xarray import XarrayFieldList


def dont_test_kerchunk():

    ds = xr.open_dataset(
        "reference://",
        engine="zarr",
        backend_kwargs={
            "consolidated": False,
            "storage_options": {
                "fo": "combined.json",
                "remote_protocol": "s3",
                "remote_options": {"anon": True},
            },
        },
    )

    fs = XarrayFieldList.from_xarray(ds)

    print(fs[-1].metadata())

    assert len(fs) == 12432


if __name__ == "__main__":
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
