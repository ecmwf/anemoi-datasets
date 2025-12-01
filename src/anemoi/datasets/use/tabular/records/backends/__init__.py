# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import io
import json
import logging
import os

import numpy as np
from cachetools import LRUCache

LOG = logging.getLogger(__name__)


def normalise_key(k):
    return "".join([x.lower() if x.isalnum() else "_" for x in k])


class Backend:
    def __init__(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs

    def read(self, i, **kwargs):
        """Read the i-th record and return a dictionary of numpy arrays."""
        raise NotImplementedError("Must be implemented in subclass")

    def read_metadata(self):
        """Read the metadata of a record dataset. The metadata does not depend on the record index."""
        raise NotImplementedError("Must be implemented in subclass")

    def read_statistics(self):
        """Read the statistics of a record dataset. The statistics does not depend on the record index."""
        raise NotImplementedError("Must be implemented in subclass")

    def _check_data(self, data):
        # Check that all keys are normalised
        for k in list(data.keys()):
            k = k.split(":")[-1]
            if k != normalise_key(k):
                raise ValueError(f"{k} must be alphanumerical and '-' only.")


class Npz1Backend(Backend):

    def __init__(self, *args, number_of_files_per_subdirectory=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.number_of_files_per_subdirectory = number_of_files_per_subdirectory
        self._cache = LRUCache(maxsize=5)

    def read(self, i, **kwargs):
        if i in self._cache:
            return self._cache[i]

        d = str(int(i / self.number_of_files_per_subdirectory))
        path = os.path.join(self.path, "data", d, f"{i}.npz")
        raw = open(path, "rb").read()
        buffer = io.BytesIO(raw)
        self._cache[i] = dict(np.load(buffer))
        return self._cache[i]

    def read_metadata(self):
        with open(os.path.join(self.path, "metadata.json")) as f:
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


class NpyBackend(Backend):

    def __init__(self, *args, number_of_files_per_subdirectory=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.number_of_files_per_subdirectory = number_of_files_per_subdirectory
        self._cache = LRUCache(maxsize=5)
        self._metadata_cache = LRUCache(maxsize=500)

    def read(self, i, **kwargs):
        if i in self._cache:
            return self._cache[i]

        d = str(int(i / self.number_of_files_per_subdirectory))
        dir_path = os.path.join(self.path, "data", d)
        path = os.path.join(dir_path, f"{i}.npy")
        metadata_path = os.path.join(dir_path, f"{i}.json")

        if i not in self._metadata_cache:
            with open(metadata_path, "r") as f:
                self._metadata_cache[i] = json.load(f)
        metadata = self._metadata_cache[i]

        flattened = np.load(path)
        key_order = metadata["key_order"]
        data = {}
        offset = 0
        for k in key_order:
            arr_meta = metadata["arrays"][k]
            if k.startswith("metadata:"):
                data[k] = arr_meta["metadata"]
                continue

            shape = tuple(arr_meta["shape"])
            size = arr_meta["size"]
            arr_flat = flattened[offset : offset + size]
            offset += size
            array = arr_flat.reshape(shape)

            dtype = arr_meta["dtype"]
            if k.startswith("timedeltas:"):
                assert dtype == "timedelta64[s]", (k, dtype)
                array = array.astype(dtype)
            else:
                assert dtype == "float32", (k, dtype)

            assert array.dtype == np.dtype(dtype), (array.dtype, dtype)
            data[k] = array
        self._cache[i] = data

        return data

    def read_metadata(self):
        with open(os.path.join(self.path, "metadata.json")) as f:
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
        nc1=Nc1Backend,
        npy=NpyBackend,
    )
    cls = BACKENDS[name]
    return cls(*args, **kwargs)


class WriteBackend(Backend):
    # Write backend base class, not used for reading
    # provides implementation to write data
    def __init__(self, *, target, **kwargs):
        super().__init__(target, **kwargs)

    def write(self, i, data, **kwargs):
        # expects data to be a dict of numpy arrays
        raise NotImplementedError("Must be implemented in subclass")

    def write_metadata(self, metadata):
        # expects metadata to be a dict
        raise NotImplementedError("Must be implemented in subclass")

    def write_statistics(self, statistics):
        # expects statistics to be a dict of dicts with the right keys:
        # {group: {mean:..., std:..., min:..., max:...}}
        raise NotImplementedError("Must be implemented in subclass")

    def _check_data(self, data):
        for k in list(data.keys()):
            k = k.split(":")[-1]
            if k != normalise_key(k):
                raise ValueError(f"{k} must be alphanumerical and '_' only.")

    def _dataframes_to_record(self, i, data, variables, **kwargs):
        # Convert data from pandas DataFrames to a record format
        # will be used for writing, building obs datasets

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
        # will be used for writing, building obs datasets

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
        from anemoi.datasets.create.gridded.tasks import _json_tidy

        os.makedirs(self.path, exist_ok=True)

        path = os.path.join(self.path, "metadata.json")
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(metadata, f, indent=2, default=_json_tidy)
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


class NpyWriteBackend(WriteBackend):

    def write(self, i, data, number_of_files_per_subdirectory=100, **kwargs):
        self.number_of_files_per_subdirectory = number_of_files_per_subdirectory
        self._check_data(data)
        d = str(int(i / self.number_of_files_per_subdirectory))
        dir_path = os.path.join(self.path, "data", d)

        out_path = os.path.join(dir_path, f"{i}.npy")
        tmp_path = os.path.join(dir_path, f"{i}.tmp.npy")
        metadata_path = os.path.join(dir_path, f"{i}.json")
        metadata_tmp_path = os.path.join(dir_path, f"{i}.tmp.json")

        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)

        cast_data = {}
        metadata = {"arrays": {}}
        for k, v in data.items():
            metadata["arrays"][k] = {}
            if k.startswith("metadata:"):
                metadata["arrays"][k]["metadata"] = str(v)
                continue
            dtype = str(v.dtype)
            shape = v.shape

            new_v = v
            if k.startswith("timedeltas:"):
                assert dtype == "timedelta64[s]", f"Expected timedelta64[s], got {dtype}"
                new_v = v.astype(np.float32)
                if new_v.size:
                    roundtrip = new_v.astype(dtype)
                    assert roundtrip.shape == v.shape, (roundtrip.shape, shape)
                    assert roundtrip.dtype == v.dtype, (roundtrip.dtype, dtype)
                    assert roundtrip.size == v.size, (roundtrip.size, v.size)
                    error = np.abs(roundtrip.astype(np.int64) - v.astype(np.int64)).max()
                    if 2 >= error > 1e-6:
                        print("❌❌❌", k, "Ignoring rounding error:", error)

                    elif error > 2:
                        print("❌", k)
                        print(roundtrip.astype(np.int64))
                        print(v.astype(np.int64))
                        a = roundtrip[:]
                        b = v[:]
                        for i in range(a.shape[0]):
                            a_ = a[i]
                            b_ = b[i]
                            print(a_, b_)
                            if a_ != b_:
                                break
                        raise ValueError(k)

            metadata["arrays"][k]["dtype"] = dtype
            metadata["arrays"][k]["shape"] = v.shape
            metadata["arrays"][k]["size"] = v.size
            cast_data[k] = new_v

        metadata["key_order"] = list(cast_data.keys())
        cast_data = np.concatenate([a.ravel() for a in cast_data.values()])

        np.save(tmp_path, cast_data)
        json.dump(metadata, open(metadata_tmp_path, "w"), indent=2)
        os.rename(tmp_path, out_path)
        os.rename(metadata_tmp_path, metadata_path)

    def write_metadata(self, metadata):
        from anemoi.datasets.create.gridded.tasks import _json_tidy

        os.makedirs(self.path, exist_ok=True)

        path = os.path.join(self.path, "metadata.json")
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(metadata, f, indent=2, default=_json_tidy)
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
        from anemoi.datasets.create.gridded.tasks import _json_tidy

        os.makedirs(self.path, exist_ok=True)
        with open(os.path.join(self.path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, default=_json_tidy)

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


def writer_backend_factory(name, **kwargs):
    # choose the right backend for writing
    # this is intended to make benchmarking easier
    WRITE_BACKENDS = dict(
        npz1=Npz1WriteBackend,
        nc1=Nc1WriteBackend,
        npy=NpyWriteBackend,
    )
    return WRITE_BACKENDS[name](**kwargs)
