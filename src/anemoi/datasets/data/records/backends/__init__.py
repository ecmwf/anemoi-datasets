# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import os

import numpy as np


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


class Npz1Backend(Backend):
    def read(self, i, **kwargs):
        path = os.path.join(self.path, "data", str(int(i / 10)), f"{i}.npz")
        with open(path, "rb") as f:
            return dict(np.load(f))

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


class Npz2Backend(Backend):
    def read(self, i, **kwargs):
        path = os.path.join(self.path, "data_", str(int(i / 10)), f"{i}_.npz")
        with open(path, "rb") as f:
            return dict(np.load(f))

    def read_metadata(self):
        with open(os.path.join(self.path, "metadata.json")) as f:
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


def backend_factory(backend, *args, **kwargs):
    BACKENDS = dict(
        npz1=Npz1Backend,
        npz2=Npz2Backend,
    )
    return BACKENDS[backend](*args, **kwargs)


class WriteBackend(Backend):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)

    def write(self, i, data, **kwargs):
        raise NotImplementedError("Must be implemented in subclass")

    def write_metadata(self, metadata):
        raise NotImplementedError("Must be implemented in subclass")

    def write_statistics(self, statistics):
        raise NotImplementedError("Must be implemented in subclass")


class Npz1WriteBackend(WriteBackend):
    def write(self, i, data, **kwargs):
        path = os.path.join(self.path, "data", str(int(i / 10)))
        os.makedirs(path, exist_ok=True)
        out_path = os.path.join(path, f"{i}.npz")
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
            for k, v in d.items():
                assert isinstance(
                    v, (int, float, np.ndarray)
                ), f"Statistics value for {k} in {name} must be int, float or ndarray, got {type(v)}"
                flatten[k + ":" + name] = v

        path = os.path.join(self.path, "statistics.npz")
        np.savez(path, **flatten)


class Npz2WriteBackend(WriteBackend):
    def write(self, i, data, **kwargs):
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
            for k, v in d.items():
                assert isinstance(
                    v, (int, float, np.ndarray)
                ), f"Statistics value for {k} in {name} must be int, float or ndarray, got {type(v)}"
                flatten[k + ":" + name] = v

        os.makedirs(self.path, exist_ok=True)
        path = os.path.join(self.path, "statistics_.npz")
        np.savez(path, **flatten)


def writer_backend_factory(backend, *args, **kwargs):
    WRITE_BACKENDS = dict(
        npz1=Npz1WriteBackend,
        npz2=Npz2WriteBackend,
    )
    return WRITE_BACKENDS[backend](*args, **kwargs)
