# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import xarray as xr

from anemoi.datasets.create.functions.sources.xarray import XarrayFieldList


def test_arco_era5():

    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
        chunks={"time": 48},
        consolidated=True,
    )

    fs = XarrayFieldList.from_xarray(ds)
    print(len(fs))

    print(fs[-1].metadata())
    print(fs[-1].to_numpy())

    assert len(fs) == 128677526


def test_weatherbench():
    ds = xr.open_zarr("gs://weatherbench2/datasets/pangu_hres_init/2020_0012_0p25.zarr")

    # https://weatherbench2.readthedocs.io/en/latest/init-vs-valid-time.html

    flavour = {
        "rules": {
            "latitude": {"name": "latitude"},
            "longitude": {"name": "longitude"},
            "step": {"name": "prediction_timedelta"},
            "date": {"name": "time"},
            "level": {"name": "level"},
        },
        "levtype": "pl",
    }

    fs = XarrayFieldList.from_xarray(ds, flavour)

    assert len(fs) == 2430240

    assert fs[0].metadata("valid_datetime") == "2020-01-01T00:00:00", fs[0].metadata("valid_datetime")


if __name__ == "__main__":
    test_weatherbench()
