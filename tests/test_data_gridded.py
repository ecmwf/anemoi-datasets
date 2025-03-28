# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
from functools import cache
from functools import wraps
from typing import Any
from typing import Callable
from typing import Optional
from typing import Type
from typing import Union
from unittest.mock import patch

import numpy as np
import zarr
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

from anemoi.datasets import open_dataset

VALUES = 20


def mockup_open_zarr(func: Callable) -> Callable:
    """Decorator to mock the open_zarr function.

    Parameters
    ----------
    func : Callable
        Function to wrap.

    Returns
    -------
    Callable
        Wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        with patch("zarr.convenience.open", zarr_from_str):
            with patch("anemoi.datasets.data.stores.zarr_lookup", lambda name: name):
                return func(*args, **kwargs)

    return wrapper


@cache
def _(date: datetime.datetime, var: str, k: int = 0, e: int = 0, values: int = VALUES) -> np.ndarray:
    """Create a simple array of values based on the date and variable name, ensemble, grid, and other parameters.

    Parameters
    ----------
    date : datetime.datetime
        Date.
    var : str
        Variable name.
    k : int, optional
        Grid index, by default 0.
    e : int, optional
        Ensemble index, by default 0.
    values : int, optional
        Number of values, by default VALUES.

    Returns
    -------
    np.ndarray
        Array of values.
    """
    d = date.year * 10000 + date.month * 100 + date.day
    v = ord(var) - ord("a") + 1
    assert 0 <= k <= 9
    assert 0 <= e <= 9

    return np.array([d * 100 + v + k / 10.0 + w / 100.0 + e / 1000.0 for w in range(values)])


def create_zarr(
    vars: str = "abcd",
    start: int = 2021,
    end: int = 2021,
    field_shape: tuple = [4, 5],
    frequency: datetime.timedelta = datetime.timedelta(hours=6),
    resolution: str = "o96",
    k: int = 0,
    ensemble: Optional[int] = None,
    grids: Optional[int] = None,
    missing: bool = False,
) -> zarr.Group:
    """Create a Zarr dataset.

    Parameters
    ----------
    vars : str, optional
        Variable names, by default "abcd".
    start : int, optional
        Start year, by default 2021.
    end : int, optional
        End year, by default 2021.
    field_shape : tuple, optional
        Field shape of dataset, by default [4, 5].
    frequency : datetime.timedelta, optional
        Frequency, by default datetime.timedelta(hours=6).
    resolution : str, optional
        Resolution, by default "o96".
    k : int, optional
        Grid index, by default 0.
    ensemble : Optional[int], optional
        Number of ensembles, by default None.
    grids : Optional[int], optional
        Number of grids, by default None.
    missing : bool, optional
        Whether to include missing dates, by default False.

    Returns
    -------
    zarr.Group
        Zarr dataset.
    """
    root = zarr.group()
    assert isinstance(frequency, datetime.timedelta)

    dates = []
    date = datetime.datetime(start, 1, 1)
    while date.year <= end:
        dates.append(date)
        date += frequency

    dates = np.array(dates, dtype="datetime64")

    ensembles = ensemble if ensemble is not None else 1
    values = grids if grids is not None else VALUES

    data = np.zeros(shape=(len(dates), len(vars), ensembles, values))

    for i, date in enumerate(dates):
        for j, var in enumerate(vars):
            for e in range(ensembles):
                data[i, j, e] = _(date.astype(object), var, k, e, values)

    root.create_dataset(
        "data",
        data=data,
        dtype=data.dtype,
        chunks=data.shape,
        compressor=None,
    )
    root.create_dataset(
        "dates",
        data=dates,
        compressor=None,
    )
    root.create_dataset(
        "latitudes",
        data=np.array([x + values for x in range(values)]),
        compressor=None,
    )
    root.create_dataset(
        "longitudes",
        data=np.array([x + values for x in range(values)]),
        compressor=None,
    )

    root.attrs["frequency"] = frequency_to_string(frequency)
    root.attrs["resolution"] = resolution
    root.attrs["name_to_index"] = {k: i for i, k in enumerate(vars)}

    root.attrs["data_request"] = {"grid": 1, "area": "g", "param_level": {}}
    root.attrs["variables_metadata"] = {v: {} for v in vars}

    if missing:
        missing_dates = []

        last = None
        for date in [d.astype(object) for d in dates]:
            yyyymmdd = date.strftime("%Y%m")
            if yyyymmdd != last:
                last = yyyymmdd
                missing_dates.append(date)

        root.attrs["missing_dates"] = [d.isoformat() for d in missing_dates]

    root.create_dataset(
        "mean",
        data=np.mean(data, axis=0),
        compressor=None,
    )
    root.create_dataset(
        "stdev",
        data=np.std(data, axis=0),
        compressor=None,
    )
    root.create_dataset(
        "maximum",
        data=np.max(data, axis=0),
        compressor=None,
    )
    root.create_dataset(
        "minimum",
        data=np.min(data, axis=0),
        compressor=None,
    )

    root.attrs["field_shape"] = field_shape

    return root


def zarr_from_str(name: str, mode: str) -> zarr.Group:
    """Create a Zarr dataset from a string.

    Parameters
    ----------
    name : str
        Dataset name.
    mode : str
        Mode.

    Returns
    -------
    zarr.Group
        Zarr dataset.
    """
    # Format: test-2021-2021-6h-o96-abcd-0

    args = dict(
        test="test",
        start=2021,
        end=2021,
        field_shape=[4, 5],
        frequency=6,
        resolution="o96",
        vars="abcd",
        k=0,
        ensemble=None,
        grids=None,
    )

    for name, bit in zip(args, name.split("-")):
        args[name] = bit

    args["field_shape"] = [int(i) for i in args["field_shape"].split(",")]

    print(args)

    return create_zarr(
        start=int(args["start"]),
        end=int(args["end"]),
        field_shape=args["field_shape"],
        frequency=frequency_to_timedelta(args["frequency"]),
        resolution=args["resolution"],
        vars=[x for x in args["vars"]],
        k=int(args["k"]),
        ensemble=int(args["ensemble"]) if args["ensemble"] is not None else None,
        grids=int(args["grids"]) if args["grids"] is not None else None,
        missing=args["test"] == "missing",
    )


class IndexTester:
    """Class to test indexing of datasets."""

    def __init__(self, ds: Any) -> None:
        """Initialize the IndexTester.

        Parameters
        ----------
        ds : Any
            Dataset.
        """
        self.ds = ds
        self.np = ds[:]  # Numpy array

        assert self.ds.shape == self.np.shape
        assert (self.ds == self.np).all()

    def __getitem__(self, index: Any) -> None:
        """Test indexing.

        Parameters
        ----------
        index : Any
            Index.
        """
        print("INDEX", type(self.ds), index)
        if self.ds[index] is None:
            assert False, (self.ds, index)

        if not (self.ds[index] == self.np[index]).all():
            # print("DS", self.ds[index])
            # print("NP", self.np[index])
            assert (self.ds[index] == self.np[index]).all()


def make_missing(x: Any) -> Any:
    """Mark data as missing.

    Parameters
    ----------
    x : Any
        Data.

    Returns
    -------
    Any
        Data with missing values.
    """
    if isinstance(x, tuple):
        return (make_missing(a) for a in x)
    if isinstance(x, list):
        return [make_missing(a) for a in x]
    if isinstance(x, dict):
        return {k: make_missing(v) for k, v in x.items()}
    if isinstance(x, str) and x.startswith("test-"):
        return x.replace("test-", "missing-")
    return x


class DatasetTester:
    """Class to test various dataset operations."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the DatasetTester.

        Parameters
        ----------
        *args : Any
            Arguments.
        **kwargs : Any
            Keyword arguments.
        """
        self.ds = open_dataset(*args, **kwargs)

        args, kwargs = make_missing((args, kwargs))

        print(f"ds={self.ds}")

    def run(
        self,
        *,
        expected_class: Type,
        expected_length: int,
        expected_shape: tuple,
        expected_variables: Union[str, list],
        expected_name_to_index: Union[str, dict],
        date_to_row: Callable,
        start_date: datetime.datetime,
        time_increment: datetime.timedelta,
        statistics_reference_dataset: Optional[Union[str, list]],
        statistics_reference_variables: Optional[Union[str, list]],
    ) -> None:
        """Run the dataset tests.

        Parameters
        ----------
        expected_class : Type
            Expected class.
        expected_length : int
            Expected length.
        expected_shape : tuple
            Expected shape.
        expected_variables : Union[str, list]
            Expected variables.
        expected_name_to_index : Union[str, dict]
            Expected name to index mapping.
        date_to_row : Callable
            Function to generate row data.
        start_date : datetime.datetime
            Start date.
        time_increment : datetime.timedelta
            Time increment.
        statistics_reference_dataset : Optional[Union[str, list]]
            Reference dataset for statistics.
        statistics_reference_variables : Optional[Union[str, list]]
            Reference variables for statistics.
        """
        if isinstance(expected_variables, str):
            expected_variables = [v for v in expected_variables]

        if isinstance(expected_name_to_index, str):
            expected_name_to_index = {v: i for i, v in enumerate(expected_name_to_index)}

        assert isinstance(self.ds, expected_class)
        assert len(self.ds) == expected_length
        assert len([row for row in self.ds]) == len(self.ds)
        assert self.ds.shape == expected_shape, (self.ds.shape, expected_shape)
        assert self.ds.variables == expected_variables

        assert set(self.ds.variables_metadata.keys()) == set(expected_variables)

        assert self.ds.name_to_index == expected_name_to_index
        assert self.ds.dates[0] == start_date
        assert self.ds.dates[1] - self.ds.dates[0] == time_increment

        dates = []
        date = start_date

        for row in self.ds:
            # print(f"{date=} {row.shape}")
            expect = date_to_row(date)
            assert (row == expect).all()
            dates.append(date)
            date += time_increment

        assert (self.ds.dates == np.array(dates, dtype="datetime64")).all()

        if statistics_reference_dataset is not None:
            self.same_stats(
                self.ds,
                open_dataset(statistics_reference_dataset),
                statistics_reference_variables,
            )

        self.indexing(self.ds)
        self.metadata(self.ds)

        self.ds.tree()

    def metadata(self, ds: Any) -> None:
        """Test metadata.

        Parameters
        ----------
        ds : Any
            Dataset.
        """
        metadata = ds.metadata()
        assert isinstance(metadata, dict)

    def same_stats(self, ds1: Any, ds2: Any, vars1: list, vars2: Optional[list] = None) -> None:
        """Compare statistics between two datasets.

        Parameters
        ----------
        ds1 : Any
            First dataset.
        ds2 : Any
            Second dataset.
        vars1 : list
            Variables in the first dataset.
        vars2 : Optional[list], optional
            Variables in the second dataset, by default None.
        """
        if vars2 is None:
            vars2 = vars1

        vars1 = [v for v in vars1]
        vars2 = [v for v in vars2]
        for v1, v2 in zip(vars1, vars2):
            idx1 = ds1.name_to_index[v1]
            idx2 = ds2.name_to_index[v2]
            assert (ds1.statistics["mean"][idx1] == ds2.statistics["mean"][idx2]).all()
            assert (ds1.statistics["stdev"][idx1] == ds2.statistics["stdev"][idx2]).all()
            assert (ds1.statistics["maximum"][idx1] == ds2.statistics["maximum"][idx2]).all()
            assert (ds1.statistics["minimum"][idx1] == ds2.statistics["minimum"][idx2]).all()

    def indexing(self, ds: Any) -> None:
        """Test indexing.

        Parameters
        ----------
        ds : Any
            Dataset.
        """
        t = IndexTester(ds)

        print("INDEXING", ds.shape)

        t[0:10, :, 0]
        t[:, 0:3, 0]
        # t[:, :, 0]
        t[0:10, 0:3, 0]
        t[:, :, :]

        if ds.shape[1] > 2:  # Variable dimension
            t[:, (1, 2), :]
            t[:, (1, 2)]

        t[0]
        t[0, :]
        t[0, 0, :]
        t[0, 0, 0, :]

        if ds.shape[2] > 1:  # Ensemble dimension
            t[0:10, :, (0, 1)]

        for i in range(3):
            t[i]
            start = 5 * i
            end = len(ds) - 5 * i
            step = len(ds) // 10

            t[start:end:step]
            t[start:end]
            t[start:]
            t[:end]
            t[::step]


@mockup_open_zarr
def test_trim_edge_simple() -> None:
    """Test trimming the edges of a dataset."""
    test = DatasetTester(
        "test-2021-2021-15,14-6h-o96-abcd",
        trim_edge=(2, 3, 4, 5),
    )

    expected_field_shape = (10, 5)
    assert test.ds.field_shape == expected_field_shape, test.ds.field_shape
    assert test.ds.shape == (365 * 4, 4, 1, np.prod(expected_field_shape)), test.ds.shape


@mockup_open_zarr
def test_trim_edge_zeros() -> None:
    """Test trimming the edges of a dataset when edges are 0"""
    for dim in range(2):
        trim_edge = [0, 0, 0, 0]
        trim_edge[dim] = 1
        test = DatasetTester(
            "test-2021-2021-15,14-6h-o96-abcd",
            trim_edge=trim_edge,
        )

        expected_field_shape = (14, 14)
        assert test.ds.field_shape == expected_field_shape, test.ds.field_shape
        assert test.ds.shape == (365 * 4, 4, 1, np.prod(expected_field_shape)), test.ds.shape

    for dim in range(2, 4):
        trim_edge = [0, 0, 0, 0]
        trim_edge[dim] = 1
        test = DatasetTester(
            "test-2021-2021-15,14-6h-o96-abcd",
            trim_edge=trim_edge,
        )

        expected_field_shape = (15, 13)
        assert test.ds.field_shape == expected_field_shape, test.ds.field_shape
        assert test.ds.shape == (365 * 4, 4, 1, np.prod(expected_field_shape)), test.ds.shape


if __name__ == "__main__":
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
