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


def backend_factory(backend, *args, **kwargs):
    BACKENDS = dict(
        npz1=Npz1Backend,
    )
    return BACKENDS[backend](*args, **kwargs)
