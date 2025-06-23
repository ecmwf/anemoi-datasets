# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
import os

import numpy as np
from cachetools import LRUCache

LOG = logging.getLogger(__name__)


def normalise_key(k):
    return "".join([x.lower() if x.isalnum() else "-" for x in k])


class Backend:
    def __init__(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs

    def read(self, i, **kwargs):
        raise NotImplementedError("Must be implemented in subclass")

    def read_metadata(self):
        raise NotImplementedError("Must be implemented in subclass")

    def read_statistics(self):
        raise NotImplementedError("Must be implemented in subclass")

    def _check_data(self, data):
        for k in list(data.keys()):
            k = k.split(":")[-1]
            if k != normalise_key(k):
                raise ValueError(f"{k} must be alphanumerical and '-' only.")


class Npz1Backend(Backend):

    def __init__(self, *args, number_of_files_per_subdirectory=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.number_of_files_per_subdirectory = number_of_files_per_subdirectory
        self._cache = None

    def read(self, i, **kwargs):
        if self._cache is None:
            self._cache = LRUCache(maxsize=5)
        if i in self._cache:
            return self._cache[i]

        d = str(int(i / self.number_of_files_per_subdirectory))
        path = os.path.join(self.path, "data", d, f"{i}.npz")
        with open(path, "rb") as f:
            data = dict(np.load(f))
            self._cache[i] = data
            return data

    def read_metadata(self):
        with open(os.path.join(self.path, "metadata.json"), "r") as f:
            return json.load(f)

    def read_statistics(self):
        path = os.path.join(self.path, "statistics.npz")
        dic = {}
        for k, v in dict(np.load(path)).items():
            key, group = k.split(":")
            if group not in dic:
                dic[group] = {}
            dic[group][key] = v
        return dic


class Npz2Backend(Backend):
    def read(self, i, **kwargs):
        path = os.path.join(self.path, "data_", str(int(i / 10)), f"{i}_.npz")
        with open(path, "rb") as f:
            return dict(np.load(f))

    def read_metadata(self):
        with open(os.path.join(self.path, "metadata.json"), "r") as f:
            return json.load(f)

    def read_statistics(self):
        path = os.path.join(self.path, "statistics_.npz")
        dic = {}
        for k, v in dict(np.load(path)).items():
            key, group = k.split(":")
            if group not in dic:
                dic[group] = {}
            dic[group][key] = v
        return dic


class Nc1Backend(Backend):
    number_of_files_per_subdirectory = 100

    def read(self, i, **kwargs):
        d = str(int(i / self.number_of_files_per_subdirectory))
        path = os.path.join(self.path, "data", d, f"{i}.nc")
        import xarray as xr

        ds = xr.open_dataset(path)
        return {var: ds[var].values for var in ds.data_vars}

    def read_metadata(self):
        with open(os.path.join(self.path, "metadata.json"), "r") as f:
            return json.load(f)

    def read_statistics(self):
        path = os.path.join(self.path, "statistics.nc")
        import xarray as xr

        ds = xr.open_dataset(path)
        flatten = {var: ds[var].values for var in ds.data_vars}
        dic = {}
        for k, v in flatten.items():
            key, group = k.split(":")
            if group not in dic:
                dic[group] = {}
            dic[group][key] = v
        return dic


def backend_factory(name, *args, **kwargs):
    BACKENDS = dict(
        npz1=Npz1Backend,
        npz2=Npz2Backend,
        nc1=Nc1Backend,
    )
    cls = BACKENDS[name]
    return cls(*args, **kwargs)


class WriteBackend(Backend):
    def __init__(self, *, target, **kwargs):
        super().__init__(target, **kwargs)

    def write(self, i, data, **kwargs):
        raise NotImplementedError("Must be implemented in subclass")

    def write_metadata(self, metadata):
        raise NotImplementedError("Must be implemented in subclass")

    def write_statistics(self, statistics):
        raise NotImplementedError("Must be implemented in subclass")

    def _check_data(self, data):
        for k in list(data.keys()):
            k = k.split(":")[-1]
            if k != normalise_key(k):
                raise ValueError(f"{k} must be alphanumerical and '-' only.")

    def _dataframes_to_record(self, i, data, variables, **kwargs):

        assert isinstance(data, (dict)), type(data)
        if not data:
            LOG.warning(f"Empty data for index {i}.")
            return data
        first = data[list(data.keys())[0]]
        import pandas as pd

        if isinstance(first, pd.DataFrame):
            data = {name: self._dataframe_to_dict(name, df, **kwargs) for name, df in data.items()}
        else:
            assert False

        return data

    def _dataframe_to_dict(self, name, df, **kwargs):
        d = {}
        d["timedeltas:" + name] = df["timedeltas"]
        d["latitudes:" + name] = df["latitudes"]
        d["longitudes:" + name] = df["longitudes"]
        d["data:" + name] = df["data"]
        d["metadata:" + name] = df["metadata"]
        return d


class Npz1WriteBackend(WriteBackend):

    def write(self, i, data, number_of_files_per_subdirectory=100, **kwargs):
        self.number_of_files_per_subdirectory = number_of_files_per_subdirectory
        self._check_data(data)
        d = str(int(i / self.number_of_files_per_subdirectory))
        dir_path = os.path.join(self.path, "data", d)

        out_path = os.path.join(dir_path, f"{i}.npz")
        tmp_path = os.path.join(dir_path, f"{i}.tmp.npz")

        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        np.savez(tmp_path, **data)
        os.rename(tmp_path, out_path)

    def write_metadata(self, metadata):
        from anemoi.datasets.create import json_tidy

        os.makedirs(self.path, exist_ok=True)

        path = os.path.join(self.path, "metadata.json")
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(metadata, f, indent=2, default=json_tidy)
        os.rename(tmp_path, path)

    def write_statistics(self, statistics):
        os.makedirs(self.path, exist_ok=True)
        flatten = {}
        for name, d in statistics.items():
            assert isinstance(d, dict), f"Statistics for {name} must be a dict, got {type(d)}"
            assert "mean" in d, f"Statistics for {name} must contain 'mean' key but got {d.keys()}"
            for k, v in d.items():
                assert isinstance(
                    v, (int, float, np.ndarray)
                ), f"Statistics value for {k} in {name} must be int, float or ndarray, got {type(v)}"
                flatten[k + ":" + name] = v

        path = os.path.join(self.path, "statistics.npz")
        np.savez(path, **flatten)


class Nc1WriteBackend(WriteBackend):
    number_of_files_per_subdirectory = 100

    def write(self, i, data, **kwargs):
        self._check_data(data)
        d = str(int(i / self.number_of_files_per_subdirectory))
        path = os.path.join(self.path, "data", d)
        os.makedirs(path, exist_ok=True)
        out_path = os.path.join(path, f"{i}.nc")

        import xarray as xr

        ds = xr.Dataset(
            {key: ([f"dim_{key}" + str(i) for i in range(value.ndim)], value) for key, value in data.items()}
        )
        ds.to_netcdf(out_path)

    def write_metadata(self, metadata):
        from anemoi.datasets.create import json_tidy

        os.makedirs(self.path, exist_ok=True)
        with open(os.path.join(self.path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, default=json_tidy)

    def write_statistics(self, statistics):
        os.makedirs(self.path, exist_ok=True)
        flatten = {}
        for name, d in statistics.items():
            assert isinstance(d, dict), f"Statistics for {name} must be a dict, got {type(d)}"
            assert "mean" in d, f"Statistics for {name} must contain 'mean' key but got {d.keys()}"
            for k, v in d.items():
                assert isinstance(
                    v, (int, float, np.ndarray)
                ), f"Statistics value for {k} in {name} must be int, float or ndarray, got {type(v)}"
                flatten[k + ":" + name] = v

        path = os.path.join(self.path, "statistics.nc")

        import xarray as xr

        ds = xr.Dataset(
            {key: ([f"dim_{key}" + str(i) for i in range(value.ndim)], value) for key, value in flatten.items()}
        )
        ds.to_netcdf(path)
        np.savez(path, **flatten)


class Npz2WriteBackend(WriteBackend):
    def write(self, i, data, **kwargs):
        self._check_data(data)
        path = os.path.join(self.path, "data_", str(int(i / 10)))
        os.makedirs(path, exist_ok=True)
        out_path = os.path.join(path, f"{i}_.npz")
        np.savez(out_path, **data)

    def write_metadata(self, metadata):
        from anemoi.datasets.create import json_tidy

        os.makedirs(self.path, exist_ok=True)
        with open(os.path.join(self.path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, default=json_tidy)

    def write_statistics(self, statistics):
        flatten = {}
        for name, d in statistics.items():
            assert isinstance(d, dict), f"Statistics for {name} must be a dict, got {type(d)}"
            assert "mean" in d, f"Statistics for {name} must contain 'mean' key but got {d.keys()}"
            for k, v in d.items():
                assert isinstance(
                    v, (int, float, np.ndarray)
                ), f"Statistics value for {k} in {name} must be int, float or ndarray, got {type(v)}"
                flatten[k + ":" + name] = v

        os.makedirs(self.path, exist_ok=True)
        path = os.path.join(self.path, "statistics_.npz")
        np.savez(path, **flatten)


def writer_backend_factory(name, **kwargs):
    WRITE_BACKENDS = dict(
        npz1=Npz1WriteBackend,
        npz2=Npz2WriteBackend,
        nc1=Nc1WriteBackend,
    )
    return WRITE_BACKENDS[name](**kwargs)
