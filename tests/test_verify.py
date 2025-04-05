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

import numpy as np
import xarray as xr
from anemoi.utils.testing import skip_if_offline
from anemoi.utils.testing import skip_missing_packages

from anemoi.datasets.verify import verify_dataset

# https://github.com/google-research/arco-era5


class DummyDataset:

    def __init__(self, ds: xr.Dataset):
        self.ds = ds

        self._variables = {}

        for name in ds.data_vars:
            if "level" in ds[name].dims:
                for level in ds[name].level.values:
                    self._variables[f"{name}_{level}"] = ds[name].sel(level=level)
            else:
                self._variables[name] = ds[name]

        self._latitudes, self._longitudes = np.meshgrid(ds.latitude.values, ds.longitude.values)

    def __len__(self):
        # The length of the dataset is the number of dates
        return len(self.ds.time)

    @property
    def dates(self):
        # Return the dates in the dataset
        return self.ds.time.values

    @property
    def variables(self):
        # Return the variables in the dataset
        return list(self._variables.keys())

    @property
    def latitudes(self):
        # Return the latitudes in the dataset
        return self._latitudes

    @property
    def longitudes(self):
        # Return the longitudes in the dataset
        return self._longitudes

    @property
    def shape(self):
        return (len(self), len(self.variables), 1, len(self.latitudes))


def _open_dataset():

    cache = "anemoi-datasets-test-verify.nc"

    if os.path.exists(cache):
        return xr.open_dataset(cache)

    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
        consolidated=True,
        storage_options=dict(token="anon"),
    )

    ds = ds.isel(time=(slice(0, 24 * 7, 6)))  # Select a few dates

    ds = ds[
        [  # Single level
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
        ]
        + [  # Pressure levels
            "temperature",
            "geopotential",
        ]
    ]

    ds = ds.sel(level=[1000, 850, 500])

    ds.to_netcdf(cache, format="NETCDF4", mode="w")

    return ds


@skip_if_offline
@skip_missing_packages("gcsfs")
def test_validate() -> None:

    dummy = DummyDataset(_open_dataset())

    result = verify_dataset(dummy)
    assert result is None, "Dataset verification failed"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
