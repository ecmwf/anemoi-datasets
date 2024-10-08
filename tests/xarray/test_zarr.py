# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import xarray as xr

from anemoi.datasets.create.functions.sources.xarray import XarrayFieldList
from anemoi.datasets.testing import assert_field_list


def test_arco_era5_1():

    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
        chunks={"time": 48},
        consolidated=True,
    )

    fs = XarrayFieldList.from_xarray(ds)
    assert_field_list(
        fs,
        128677526,
        "1959-01-01T00:00:00",
        "2021-12-31T23:00:00",
    )


def test_arco_era5_2():

    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr",
        chunks={"time": 48},
        consolidated=True,
    )

    fs = XarrayFieldList.from_xarray(ds)
    assert_field_list(
        fs,
        128677526,
        "1959-01-01T00:00:00",
        "2021-12-31T23:00:00",
    )


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

    assert_field_list(
        fs,
        2430240,
        "2020-01-01T06:00:00",
        "2021-01-10T12:00:00",
    )


def test_inca_one_date():
    url = "https://object-store.os-api.cci1.ecmwf.int/ml-tests/test-data/example-inca-one-date.zarr"

    ds = xr.open_zarr(url)
    fs = XarrayFieldList.from_xarray(ds)
    vars = ["DD_10M", "SP_10M", "TD_2M", "TOT_PREC", "T_2M"]

    for i, f in enumerate(fs):
        print(f)
        assert f.metadata("valid_datetime") == "2023-01-01T00:00:00"
        assert f.metadata("step") == 0
        assert f.metadata("number") == 0
        assert f.metadata("variable") == vars[i]

    print(fs[0].datetime())


if __name__ == "__main__":
    # test_arco_era5_2()
    # exit()
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
