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
from functools import cached_property

import numpy as np
import xarray as xr
from anemoi.utils.testing import skip_if_offline
from anemoi.utils.testing import skip_missing_packages

from anemoi.datasets.verify import verify_dataset

LOG = logging.getLogger(__name__)

# https://github.com/google-research/arco-era5


class DemoAlternativeDataset:

    def __init__(self, ds: xr.Dataset):
        self.ds = ds

        self._xarray_variables = {}

        for name in ds.data_vars:
            if "level" in ds[name].dims:
                for level in ds[name].level.values:
                    self._xarray_variables[f"{name}_{level}"] = ds[name].sel(level=level)
            else:
                self._xarray_variables[name] = ds[name]

        self._latitudes, self._longitudes = np.meshgrid(ds.latitude.values, ds.longitude.values)
        assert len(self._latitudes) == len(self._longitudes)

        self._latitudes = self._latitudes.flatten()
        self._longitudes = self._longitudes.flatten()

        self._number_of_dates = len(ds.time)

        self._variables = list(self._xarray_variables)
        self._number_of_variables = len(self._variables)
        self._number_of_grid_points = len(self._latitudes)
        self._number_of_members = 1  # For now, we assume it is not an ensemble

        self._shape = (
            self._number_of_dates,
            self._number_of_variables,
            self._number_of_members,
            self._number_of_grid_points,
        )

    def __len__(self):
        return self._number_of_dates

    def __getitem__(self, index):

        if isinstance(index, int):
            result = np.concatenate(
                [self._xarray_variables[name].isel(time=index).values.flatten() for name in self._xarray_variables]
            )
            return result.reshape(-1, self._number_of_members, self._number_of_grid_points)

        if isinstance(index, slice):
            result = np.concatenate(
                [
                    self._xarray_variables[name]
                    .isel(time=index)
                    .values.reshape(-1, 1, self._number_of_members, self._number_of_grid_points)
                    for name in self._xarray_variables
                ],
                axis=1,
            )
            return result

        # First, handle Ellipsis
        if Ellipsis in index:
            assert index.count(Ellipsis) == 1, "Only one Ellipsis is allowed in the index"
            size = len(self._shape) - len(index)
            assert size >= 1, "Invalid index: Ellipsis must be at least one dimension"
            index = index[: index.index(Ellipsis)] + (slice(None),) * size + index[index.index(Ellipsis) + 1 :]

        # Complete the index with slices
        while len(index) < len(self._shape):
            index = index + (slice(None),)

        # Ignore the member index for now

        time_slice = index[0]
        if isinstance(index[1], int):
            variables = [self._variables[index[1]]]
        elif isinstance(index[1], slice):
            variables = self._variables[index[1]]
        else:
            variables = [self._variables[i] for i in index[1]]

        data_slices = index[3]

        result = np.concatenate(
            [
                self._xarray_variables[name]
                .isel(time=time_slice)
                .values.reshape(-1, 1, self._number_of_members, self._number_of_grid_points)
                for name in variables
            ],
            axis=1,
        )

        result = result[..., data_slices]

        # Squeeze non-slice dimensions

        for i in reversed(range(len(index))):
            if isinstance(index[i], int):
                result = result.squeeze(i)

        return result

    @property
    def variables(self):
        # Return the variables in the dataset
        return self._variables

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
        return self._shape

    @property
    def name_to_index(self):
        # Return the mapping from variable name to index
        return {name: i for i, name in enumerate(self._variables)}

    @cached_property
    def statistics(self):
        # Return the statistics of the dataset

        mean = np.array([self._xarray_variables[name].mean().values for name in self._variables])
        stdev = np.array([self._xarray_variables[name].std().values for name in self._variables])
        minimum = np.array([self._xarray_variables[name].min().values for name in self._variables])
        maximum = np.array([self._xarray_variables[name].max().values for name in self._variables])

        assert len(mean.shape) == 1, mean.shape

        return {
            "mean": mean,
            "stdev": stdev,
            "minimum": minimum,
            "maximum": maximum,
        }

    @property
    def missing(self):
        # Return the index of the missing dates in the dataset
        # This is a dummy implementation for the sake of the example
        return {1, 9}

    # Below are the methods that are used to add metadata in the checkpoint

    def metadata(self):
        # This will be stored in the model's checkpoint
        # to be used by `anemoi-inference`
        return {}

    def supporting_arrays(self):
        # This will be stored in the model's checkpoint
        # to be used by `anemoi-inference`
        return {
            "latitudes": self.latitudes,
            "longitudes": self.longitudes,
        }

    # Below are the methods that are not used during training

    @cached_property
    def dates(self):
        # Return the dates in the dataset
        return self.ds.time.values

    @property
    def start_date(self):
        # Return the start date of the dataset
        return self.dates[0]

    @property
    def end_date(self):
        # Return the end date of the dataset
        return self.dates[-1]

    @cached_property
    def frequency(self):
        frequency = np.diff(self.dates).astype("timedelta64[s]")
        assert np.all(frequency == frequency[0])
        return frequency[0]


def _open_dataset():

    cache = "anemoi-datasets-test-verify-cache-file.nc"

    if os.path.exists(cache):
        LOG.info("Loading dataset from %s", cache)
        return xr.open_dataset(cache)

    LOG.info("Loading dataset from the Internet")

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

    if int(os.environ.get("ANEMOI_DATASETS_TEST_VERIFY_CACHE_FILE", 0)):
        LOG.info("Caching dataset to %s", cache)
        ds.to_netcdf(cache, format="NETCDF4", mode="w")

    return ds


@skip_if_offline
@skip_missing_packages("gcsfs")
def test_validate() -> None:

    dummy = DemoAlternativeDataset(_open_dataset())

    result = verify_dataset(dummy, costly_checks=True, detailed=True)
    assert result is None, "Dataset verification failed"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
