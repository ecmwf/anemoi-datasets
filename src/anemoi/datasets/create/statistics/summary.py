# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
from collections import defaultdict
from typing import Any

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

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Summary object with given keyword arguments.

        Parameters
        ----------
        **kwargs : Any
            Arbitrary keyword arguments representing summary statistics.
        """
        super().__init__(**kwargs)
        self.check()

    @property
    def size(self) -> int:
        """Get the size of the summary, which is the number of variables."""
        return len(self["variables_names"])

    def check(self) -> None:
        """Perform checks on the summary statistics to ensure they are valid.

        Raises
        ------
        AssertionError
            If any of the checks fail.
        StatisticsValueError
            If any of the statistical checks fail.
        """
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

    def __str__(self) -> str:
        """Return a string representation of the summary statistics.

        Returns
        -------
        str
            A formatted string of the summary statistics.
        """
        header = ["Variables"] + self.STATS_NAMES
        out = [" ".join(header)]

        out += [
            " ".join([v] + [f"{self[n][i]:.2f}" for n in self.STATS_NAMES])
            for i, v in enumerate(self["variables_names"])
        ]
        return "\n".join(out)

    def save(self, filename: str, **metadata: Any) -> None:
        """Save the summary statistics to a JSON file.

        Parameters
        ----------
        filename : str
            The name of the file to save the summary statistics.
        **metadata : Any
            Additional metadata to include in the JSON file.
        """
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

    def load(self, filename: str) -> "Summary":
        """Load the summary statistics from a JSON file.

        Parameters
        ----------
        filename : str
            The name of the file to load the summary statistics from.

        Returns
        -------
        Summary
            The loaded Summary object.
        """
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
