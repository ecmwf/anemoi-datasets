# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import xarray as xr
from anemoi.utils.testing import skip_if_offline
from anemoi.utils.testing import skip_missing_packages

from anemoi.datasets.create.sources.xarray import XarrayFieldList
from anemoi.datasets.data.stores import name_to_zarr_store
from anemoi.datasets.testing import assert_field_list


@skip_if_offline
@skip_missing_packages("gcsfs")
def test_arco_era5_1() -> None:
    """Test loading and validating the arco_era5_1 dataset."""
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


@skip_if_offline
@skip_missing_packages("gcsfs")
def test_arco_era5_2() -> None:
    """Test loading and validating the arco_era5_2 dataset."""
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


@skip_if_offline
@skip_missing_packages("gcsfs")
def test_weatherbench() -> None:
    """Test loading and validating the weatherbench dataset."""
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

    fs = XarrayFieldList.from_xarray(ds, flavour=flavour)

    assert_field_list(
        fs,
        2430240,
        "2020-01-01T06:00:00",
        "2021-01-10T12:00:00",
    )


@skip_if_offline
@skip_missing_packages("aiohttp")
def test_inca_one_date() -> None:
    """Test loading and validating the inca_one_date dataset."""
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


@skip_if_offline
@skip_missing_packages("gcsfs")
def test_noaa_replay() -> None:
    """Test loading and validating the noaa_replay dataset."""
    ds = xr.open_zarr(
        "gs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/03h-freq/zarr/fv3.zarr",
        storage_options={"token": "anon"},
    )

    flavour = {
        "rules": {
            "latitude": {"name": "grid_yt"},
            "longitude": {"name": "grid_xt"},
            "time": {"name": "time"},
            "level": {"name": "pfull"},
        },
        "levtype": "pl",
    }

    fs = XarrayFieldList.from_xarray(ds, flavour=flavour)

    assert_field_list(
        fs,
        36972870,
        "1993-12-31T18:00:00",
        "1999-06-13T03:00:00",
    )


@skip_if_offline
@skip_missing_packages("planetary_computer", "adlfs")
def test_planetary_computer_conus404() -> None:
    """Test loading and validating the planetary_computer_conus404 dataset."""
    url = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/conus404"
    ds = xr.open_zarr(**name_to_zarr_store(url))

    flavour = {
        "rules": {
            "latitude": {"name": "lat"},
            "longitude": {"name": "lon"},
            "x": {"name": "west_east"},
            "y": {"name": "south_north"},
            "time": {"name": "time"},
        },
    }

    fs = XarrayFieldList.from_xarray(ds, flavour=flavour)

    assert_field_list(
        fs,
        74634912,
        "1979-10-01T00:00:00",
        "2022-09-30T23:00:00",
    )


if __name__ == "__main__":
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
