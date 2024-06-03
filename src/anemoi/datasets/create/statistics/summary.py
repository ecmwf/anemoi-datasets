# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import json
from collections import defaultdict

import numpy as np

from ..check import StatisticsValueError
from ..check import check_data_values
from ..check import check_stats


class Summary(dict):
    """This class is used to store the summary statistics of a dataset. It can be saved and loaded from a json file. And does some basic checks on the data."""

    STATS_NAMES = [
        "minimum",
        "maximum",
        "mean",
        "stdev",
        "has_nans",
    ]  # order matter for __str__.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.check()

    @property
    def size(self):
        return len(self["variables_names"])

    def check(self):
        for k, v in self.items():
            if k == "variables_names":
                assert len(v) == self.size
                continue
            assert v.shape == (self.size,)
            if k == "count":
                assert (v >= 0).all(), (k, v)
                assert v.dtype == np.int64, (k, v)
                continue
            if k == "has_nans":
                assert v.dtype == np.bool_, (k, v)
                continue
            if k == "stdev":
                assert (v >= 0).all(), (k, v)
            assert v.dtype == np.float64, (k, v)

        for i, name in enumerate(self["variables_names"]):
            try:
                check_stats(**{k: v[i] for k, v in self.items()}, msg=f"{i} {name}")
                check_data_values(self["minimum"][i], name=name)
                check_data_values(self["maximum"][i], name=name)
                check_data_values(self["mean"][i], name=name)
            except StatisticsValueError as e:
                e.args += (i, name)
                raise

    def __str__(self):
        header = ["Variables"] + self.STATS_NAMES
        out = [" ".join(header)]

        out += [
            " ".join([v] + [f"{self[n][i]:.2f}" for n in self.STATS_NAMES])
            for i, v in enumerate(self["variables_names"])
        ]
        return "\n".join(out)

    def save(self, filename, **metadata):
        assert filename.endswith(".json"), filename
        dic = {}
        for k in self.STATS_NAMES:
            dic[k] = list(self[k])

        out = dict(data=defaultdict(dict))
        for i, name in enumerate(self["variables_names"]):
            for k in self.STATS_NAMES:
                out["data"][name][k] = dic[k][i]

        out["metadata"] = metadata

        with open(filename, "w") as f:
            json.dump(out, f, indent=2)

    def load(self, filename):
        assert filename.endswith(".json"), filename
        with open(filename) as f:
            dic = json.load(f)

        dic_ = {}
        for k, v in dic.items():
            if k == "count":
                dic_[k] = np.array(v, dtype=np.int64)
                continue
            if k == "variables":
                dic_[k] = v
                continue
            dic_[k] = np.array(v, dtype=np.float64)
        return Summary(dic_)
