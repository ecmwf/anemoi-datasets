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
import pytest
import zarr
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

from anemoi.datasets import open_dataset
from anemoi.datasets.data.concat import Concat
from anemoi.datasets.data.ensemble import Ensemble
from anemoi.datasets.data.grids import GridsBase
from anemoi.datasets.data.join import Join
from anemoi.datasets.data.misc import as_first_date
from anemoi.datasets.data.misc import as_last_date
from anemoi.datasets.data.select import Rename
from anemoi.datasets.data.select import Select
from anemoi.datasets.data.statistics import Statistics
from anemoi.datasets.data.stores import Zarr
from anemoi.datasets.data.subset import Subset

VALUES = 10


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
    """Create a simple array of values based on the date, variable name, ensemble, grid, and other parameters.

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
        frequency=6,
        resolution="o96",
        vars="abcd",
        k=0,
        ensemble=None,
        grids=None,
    )

    for name, bit in zip(args, name.split("-")):
        args[name] = bit

    print(args)

    return create_zarr(
        start=int(args["start"]),
        end=int(args["end"]),
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
        """Initialise the IndexTester.

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


def make_row(*args: Any, ensemble: bool = False, grid: bool = False) -> np.ndarray:
    """Create a row of data.

    Parameters
    ----------
    *args : Any
        Additional arguments.
    ensemble : bool, optional
        Whether to include the ensemble dimension, by default False.
    grid : bool, optional
        Whether to include the grid dimension, by default False.

    Returns
    -------
    np.ndarray
        Row of data.
    """
    # assert not isinstance(args[0], (list, tuple))
    if grid:

        def _(x):
            return np.concatenate([np.array(p) for p in x])

        args = [_(a) for a in args]

    if ensemble:
        args = [np.array(a) for a in args]
    else:
        args = [np.array(a).reshape(1, -1) for a in args]  # Add ensemble dimension
    return np.array(args)


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
        """Initialise the DatasetTester.

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
            Expected name-to-index mapping.
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


def simple_row(date: datetime.datetime, vars: str) -> np.ndarray:
    """Create a simple row of data.

    Parameters
    ----------
    date : datetime.datetime
        Date.
    vars : str
        Variables.

    Returns
    -------
    np.ndarray
        Row of data.
    """
    values = [_(date, v) for v in vars]
    return make_row(*values)


@mockup_open_zarr
def test_simple() -> None:
    """Test a simple dataset."""
    test = DatasetTester("test-2021-2022-6h-o96-abcd")
    test.run(
        expected_class=Zarr,
        expected_length=365 * 2 * 4,
        date_to_row=lambda date: simple_row(date, "abcd"),
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=6),
        expected_shape=(365 * 2 * 4, 4, 1, VALUES),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        statistics_reference_dataset="test-2021-2022-6h-o96-abcd",
        statistics_reference_variables="abcd",
    )


@mockup_open_zarr
def test_concat() -> None:
    """Test concatenating datasets."""
    test = DatasetTester(
        "test-2021-2022-6h-o96-abcd",
        "test-2023-2023-6h-o96-abcd",
    )
    test.run(
        expected_class=Concat,
        expected_length=365 * 3 * 4,
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=6),
        expected_shape=(365 * 3 * 4, 4, 1, VALUES),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        date_to_row=lambda date: simple_row(date, "abcd"),
        statistics_reference_dataset="test-2021-2022-6h-o96-abcd",
        statistics_reference_variables="abcd",
    )


@mockup_open_zarr
def test_join_1() -> None:
    """Test joining datasets (case 1)."""
    test = DatasetTester("test-2021-2021-6h-o96-abcd", "test-2021-2021-6h-o96-efgh")
    test.run(
        expected_class=Join,
        expected_length=365 * 4,
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=6),
        expected_shape=(365 * 4, 8, 1, VALUES),
        expected_variables="abcdefgh",
        expected_name_to_index="abcdefgh",
        date_to_row=lambda date: simple_row(date, "abcdefgh"),
        # TODO: test second stats
        statistics_reference_dataset="test-2021-2021-6h-o96-abcd",
        statistics_reference_variables="abcd",
    )


@mockup_open_zarr
def test_join_2() -> None:
    """Test joining datasets (case 2)."""
    test = DatasetTester("test-2021-2021-6h-o96-abcd-1", "test-2021-2021-6h-o96-bdef-2")
    test.run(
        expected_class=Select,
        expected_length=365 * 4,
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=6),
        expected_shape=(365 * 4, 6, 1, VALUES),
        expected_variables="abcdef",
        expected_name_to_index="abcdef",
        date_to_row=lambda date: make_row(
            _(date, "a", 1),
            _(date, "b", 2),
            _(date, "c", 1),
            _(date, "d", 2),
            _(date, "e", 2),
            _(date, "f", 2),
        ),
        statistics_reference_dataset=[
            "test-2021-2021-6h-o96-ac-1",
            "test-2021-2021-6h-o96-bdef-2",
        ],
        statistics_reference_variables="abcdef",
    )


@mockup_open_zarr
def test_join_3() -> None:
    """Test joining datasets (case 3)."""
    test = DatasetTester("test-2021-2021-6h-o96-abcd-1", "test-2021-2021-6h-o96-abcd-2")

    # TODO: This should trigger a warning about occulted dataset

    test.run(
        expected_class=Select,
        expected_length=365 * 4,
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=6),
        expected_shape=(365 * 4, 4, 1, VALUES),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        date_to_row=lambda date: make_row(
            _(date, "a", 2),
            _(date, "b", 2),
            _(date, "c", 2),
            _(date, "d", 2),
        ),
        statistics_reference_dataset="test-2021-2021-6h-o96-abcd-2",
        statistics_reference_variables="abcd",
    )


@mockup_open_zarr
def test_subset_1() -> None:
    """Test subsetting a dataset (case 1)."""
    test = DatasetTester("test-2021-2023-1h-o96-abcd", frequency=12)
    test.run(
        expected_class=Subset,
        expected_length=365 * 3 * 2,
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=12),
        expected_shape=(365 * 3 * 2, 4, 1, VALUES),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        date_to_row=lambda date: simple_row(date, "abcd"),
        statistics_reference_dataset="test-2021-2023-1h-o96-abcd",
        statistics_reference_variables="abcd",
    )


@mockup_open_zarr
def test_subset_2() -> None:
    """Test subsetting a dataset (case 2)."""
    test = DatasetTester("test-2021-2023-1h-o96-abcd", start=2022, end=2022)
    test.run(
        expected_class=Subset,
        expected_length=365 * 24,
        expected_shape=(365 * 24, 4, 1, VALUES),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        date_to_row=lambda date: simple_row(date, "abcd"),
        start_date=datetime.datetime(2022, 1, 1),
        time_increment=datetime.timedelta(hours=1),
        statistics_reference_dataset="test-2021-2023-1h-o96-abcd",
        statistics_reference_variables="abcd",
    )


@mockup_open_zarr
def test_subset_3() -> None:
    """Test subsetting a dataset (case 3)."""
    test = DatasetTester("test-2021-2023-1h-o96-abcd", start=2022, end=2022, frequency=12)
    test.run(
        expected_class=Subset,
        expected_length=365 * 2,
        expected_shape=(365 * 2, 4, 1, VALUES),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        date_to_row=lambda date: simple_row(date, "abcd"),
        start_date=datetime.datetime(2022, 1, 1),
        time_increment=datetime.timedelta(hours=12),
        statistics_reference_dataset="test-2021-2023-1h-o96-abcd",
        statistics_reference_variables="abcd",
    )


@mockup_open_zarr
def test_subset_4() -> None:
    """Test subsetting a dataset (case 4)."""
    test = DatasetTester("test-2021-2023-1h-o96-abcd", start=202206, end=202208)
    test.run(
        expected_class=Subset,
        expected_length=(30 + 31 + 31) * 24,
        start_date=datetime.datetime(2022, 6, 1),
        time_increment=datetime.timedelta(hours=1),
        expected_shape=((30 + 31 + 31) * 24, 4, 1, VALUES),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        date_to_row=lambda date: simple_row(date, "abcd"),
        statistics_reference_dataset="test-2021-2023-1h-o96-abcd",
        statistics_reference_variables="abcd",
    )


@mockup_open_zarr
def test_subset_5() -> None:
    """Test subsetting a dataset (case 5)."""
    test = DatasetTester("test-2021-2023-1h-o96-abcd", start=20220601, end=20220831)
    test.run(
        expected_class=Subset,
        expected_length=(30 + 31 + 31) * 24,
        start_date=datetime.datetime(2022, 6, 1),
        time_increment=datetime.timedelta(hours=1),
        expected_shape=((30 + 31 + 31) * 24, 4, 1, VALUES),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        date_to_row=lambda date: simple_row(date, "abcd"),
        statistics_reference_dataset="test-2021-2023-1h-o96-abcd",
        statistics_reference_variables="abcd",
    )


@mockup_open_zarr
def test_subset_6() -> None:
    """Test subsetting a dataset (case 6)."""
    test = DatasetTester("test-2021-2023-1h-o96-abcd", start="2022-06-01", end="2022-08-31")
    test.run(
        expected_class=Subset,
        expected_length=(30 + 31 + 31) * 24,
        start_date=datetime.datetime(2022, 6, 1),
        time_increment=datetime.timedelta(hours=1),
        expected_shape=((30 + 31 + 31) * 24, 4, 1, VALUES),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        date_to_row=lambda date: simple_row(date, "abcd"),
        statistics_reference_dataset="test-2021-2023-1h-o96-abcd",
        statistics_reference_variables="abcd",
    )


@mockup_open_zarr
def test_subset_7() -> None:
    """Test subsetting a dataset (case 7)."""
    test = DatasetTester("test-2021-2023-1h-o96-abcd", start="2022-06", end="2022-08")
    test.run(
        expected_class=Subset,
        expected_length=(30 + 31 + 31) * 24,
        start_date=datetime.datetime(2022, 6, 1),
        time_increment=datetime.timedelta(hours=1),
        expected_shape=((30 + 31 + 31) * 24, 4, 1, VALUES),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        date_to_row=lambda date: simple_row(date, "abcd"),
        statistics_reference_dataset="test-2021-2023-1h-o96-abcd",
        statistics_reference_variables="abcd",
    )


@mockup_open_zarr
def test_subset_8() -> None:
    """Test subsetting a dataset (case 8)."""
    test = DatasetTester(
        "test-2021-2021-1h-o96-abcd",
        start="03:00",
        frequency="6h",
    )
    test.run(
        expected_class=Subset,
        expected_length=365 * 4,
        expected_shape=(365 * 4, 4, 1, VALUES),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        date_to_row=lambda date: simple_row(date, "abcd"),
        start_date=datetime.datetime(2021, 1, 1, 3, 0, 0),
        time_increment=datetime.timedelta(hours=6),
        statistics_reference_dataset="test-2021-2021-1h-o96-abcd",
        statistics_reference_variables="abcd",
    )


@mockup_open_zarr
def test_select_1() -> None:
    """Test selecting variables from a dataset (case 1)."""
    test = DatasetTester("test-2021-2021-6h-o96-abcd", select=["b", "d"])
    test.run(
        expected_class=Select,
        expected_length=365 * 4,
        expected_shape=(365 * 4, 2, 1, VALUES),
        expected_variables="bd",
        expected_name_to_index="bd",
        date_to_row=lambda date: simple_row(date, "bd"),
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=6),
        statistics_reference_dataset="test-2021-2021-6h-o96-abcd",
        statistics_reference_variables="bd",
    )


@mockup_open_zarr
def test_select_2() -> None:
    """Test selecting variables from a dataset (case 2)."""
    test = DatasetTester("test-2021-2021-6h-o96-abcd", select=["c", "a"])
    test.run(
        expected_class=Select,
        expected_length=365 * 4,
        expected_shape=(365 * 4, 2, 1, VALUES),
        expected_variables="ca",
        expected_name_to_index="ca",
        date_to_row=lambda date: simple_row(date, "ca"),
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=6),
        statistics_reference_dataset="test-2021-2021-6h-o96-abcd",
        statistics_reference_variables="ca",
    )


@mockup_open_zarr
def test_select_3() -> None:
    """Test selecting variables from a dataset (case 3)."""
    test = DatasetTester("test-2021-2021-6h-o96-abcd", select={"c", "a"})
    test.run(
        expected_class=Select,
        expected_length=365 * 4,
        expected_shape=(365 * 4, 2, 1, VALUES),
        expected_variables="ac",
        expected_name_to_index="ac",
        date_to_row=lambda date: simple_row(date, "ac"),
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=6),
        statistics_reference_dataset="test-2021-2021-6h-o96-abcd",
        statistics_reference_variables="ac",
    )


@mockup_open_zarr
def test_rename() -> None:
    """Test renaming variables in a dataset."""
    test = DatasetTester("test-2021-2021-6h-o96-abcd", rename={"a": "x", "c": "y"})
    test.run(
        expected_class=Rename,
        expected_length=365 * 4,
        expected_shape=(365 * 4, 4, 1, VALUES),
        expected_variables="xbyd",
        expected_name_to_index="xbyd",
        date_to_row=lambda date: simple_row(date, "abcd"),
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=6),
        statistics_reference_dataset=None,
        statistics_reference_variables=None,
    )

    test.same_stats(test.ds, open_dataset("test-2021-2021-6h-o96-abcd"), "xbyd", "abcd")


@mockup_open_zarr
def test_drop() -> None:
    """Test dropping variables from a dataset."""
    test = DatasetTester("test-2021-2021-6h-o96-abcd", drop="a")
    test.run(
        expected_class=Select,
        expected_length=365 * 4,
        expected_shape=(365 * 4, 3, 1, VALUES),
        expected_variables="bcd",
        expected_name_to_index="bcd",
        date_to_row=lambda date: simple_row(date, "bcd"),
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=6),
        statistics_reference_dataset="test-2021-2021-6h-o96-abcd",
        statistics_reference_variables="bcd",
    )


@mockup_open_zarr
def test_reorder_1() -> None:
    """Test reordering variables in a dataset (case 1)."""
    test = DatasetTester("test-2021-2021-6h-o96-abcd", reorder=["d", "c", "b", "a"])
    test.run(
        expected_class=Select,
        expected_length=365 * 4,
        expected_shape=(365 * 4, 4, 1, VALUES),
        expected_variables="dcba",
        expected_name_to_index="dcba",
        date_to_row=lambda date: simple_row(date, "dcba"),
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=6),
        statistics_reference_dataset="test-2021-2021-6h-o96-abcd",
        statistics_reference_variables="dcba",
    )


@mockup_open_zarr
def test_reorder_2() -> None:
    """Test reordering variables in a dataset (case 2)."""
    test = DatasetTester("test-2021-2021-6h-o96-abcd", reorder=dict(a=3, b=2, c=1, d=0))
    test.run(
        expected_class=Select,
        expected_length=365 * 4,
        expected_shape=(365 * 4, 4, 1, VALUES),
        expected_variables="dcba",
        expected_name_to_index="dcba",
        date_to_row=lambda date: simple_row(date, "dcba"),
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=6),
        statistics_reference_dataset="test-2021-2021-6h-o96-abcd",
        statistics_reference_variables="dcba",
    )


@mockup_open_zarr
def test_constructor_1() -> None:
    """Test dataset constructor (case 1)."""
    ds1 = open_dataset("test-2021-2021-6h-o96-abcd")

    ds2 = open_dataset("test-2022-2022-6h-o96-abcd")

    test = DatasetTester(ds1, ds2)
    test.run(
        expected_class=Concat,
        expected_length=365 * 2 * 4,
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=6),
        expected_shape=(365 * 2 * 4, 4, 1, VALUES),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        date_to_row=lambda date: simple_row(date, "abcd"),
        statistics_reference_dataset="test-2021-2021-6h-o96-abcd",
        statistics_reference_variables="abcd",
    )


@mockup_open_zarr
def test_constructor_2() -> None:
    """Test dataset constructor (case 2)."""
    test = DatasetTester(
        datasets=[
            "test-2021-2021-6h-o96-abcd",
            "test-2022-2022-6h-o96-abcd",
        ]
    )
    test.run(
        expected_class=Concat,
        expected_length=365 * 2 * 4,
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=6),
        expected_shape=(365 * 2 * 4, 4, 1, VALUES),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        date_to_row=lambda date: simple_row(date, "abcd"),
        statistics_reference_dataset="test-2021-2021-6h-o96-abcd",
        statistics_reference_variables="abcd",
    )


@mockup_open_zarr
def test_constructor_3() -> None:
    """Test dataset constructor (case 3)."""
    test = DatasetTester(
        {
            "datasets": [
                "test-2021-2021-6h-o96-abcd",
                "test-2022-2022-6h-o96-abcd",
            ]
        }
    )
    test.run(
        expected_class=Concat,
        expected_length=365 * 2 * 4,
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=6),
        expected_shape=(365 * 2 * 4, 4, 1, VALUES),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        date_to_row=lambda date: simple_row(date, "abcd"),
        statistics_reference_dataset="test-2021-2021-6h-o96-abcd",
        statistics_reference_variables="abcd",
    )


@mockup_open_zarr
def test_constructor_4() -> None:
    """Test dataset constructor (case 4)."""
    test = DatasetTester(
        "test-2021-2021-6h-o96-abcd",
        {
            "dataset": "test-2022-2022-1h-o96-abcd",
            "frequency": 6,
        },
    )
    test.run(
        expected_class=Concat,
        expected_length=365 * 2 * 4,
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=6),
        expected_shape=(365 * 2 * 4, 4, 1, VALUES),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        date_to_row=lambda date: simple_row(date, "abcd"),
        statistics_reference_dataset="test-2021-2021-6h-o96-abcd",
        statistics_reference_variables="abcd",
    )


@mockup_open_zarr
def test_constructor_5() -> None:
    """Test dataset constructor (case 5)."""
    test = DatasetTester(
        {"dataset": "test-2021-2021-6h-o96-abcd-1", "rename": {"a": "x", "c": "y"}},
        {"dataset": "test-2021-2021-6h-o96-abcd-2", "rename": {"c": "z", "d": "t"}},
    )
    test.run(
        expected_class=Select,
        expected_length=365 * 4,
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=6),
        expected_shape=(365 * 4, 7, 1, VALUES),
        expected_variables="xbydazt",
        expected_name_to_index="xbydazt",
        date_to_row=lambda date: make_row(
            _(date, "a", 1),
            _(date, "b", 2),
            _(date, "c", 1),
            _(date, "d", 1),
            _(date, "a", 2),
            _(date, "c", 2),
            _(date, "d", 2),
        ),
        statistics_reference_dataset=None,
        statistics_reference_variables=None,
    )

    test.same_stats(test.ds, open_dataset("test-2021-2021-6h-o96-abcd-1"), "xyd", "acd")
    test.same_stats(test.ds, open_dataset("test-2021-2021-6h-o96-abcd-2"), "abzt", "abcd")


@mockup_open_zarr
def test_dates() -> None:
    """Test date handling functions."""
    dates = None
    assert as_first_date(2021, dates) == np.datetime64("2021-01-01T00:00:00")
    assert as_last_date(2021, dates) == np.datetime64("2021-12-31T23:59:59")
    assert as_first_date("2021", dates) == np.datetime64("2021-01-01T00:00:00")
    assert as_last_date("2021", dates) == np.datetime64("2021-12-31T23:59:59")

    assert as_first_date(202106, dates) == np.datetime64("2021-06-01T00:00:00")
    assert as_last_date(202108, dates) == np.datetime64("2021-08-31T23:59:59")
    assert as_first_date("202106", dates) == np.datetime64("2021-06-01T00:00:00")
    assert as_last_date("202108", dates) == np.datetime64("2021-08-31T23:59:59")
    assert as_first_date("2021-06", dates) == np.datetime64("2021-06-01T00:00:00")
    assert as_last_date("2021-08", dates) == np.datetime64("2021-08-31T23:59:59")

    assert as_first_date(20210101, dates) == np.datetime64("2021-01-01T00:00:00")
    assert as_last_date(20210101, dates) == np.datetime64("2021-01-01T23:59:59")
    assert as_first_date("20210101", dates) == np.datetime64("2021-01-01T00:00:00")
    assert as_last_date("20210101", dates) == np.datetime64("2021-01-01T23:59:59")
    assert as_first_date("2021-01-01", dates) == np.datetime64("2021-01-01T00:00:00")
    assert as_last_date("2021-01-01", dates) == np.datetime64("2021-01-01T23:59:59")


@mockup_open_zarr
def test_dates_using_list() -> None:
    """Test date handling functions using a list of dates."""
    dates = [np.datetime64("2021-01-01T00:00:00") + i * np.timedelta64(6, "h") for i in range(3, 365 * 4 - 2)]
    assert dates[0] == np.datetime64("2021-01-01T18:00:00")
    assert dates[-1] == np.datetime64("2021-12-31T06:00:00")

    assert as_first_date(2021, dates) == np.datetime64("2021-01-01T18:00:00")
    assert as_last_date(2021, dates) == np.datetime64("2021-12-31T06:00:00")
    assert as_first_date("2021", dates) == np.datetime64("2021-01-01T18:00:00")
    assert as_last_date("2021", dates) == np.datetime64("2021-12-31T06:00:00")


@mockup_open_zarr
def test_dates_using_list_2() -> None:
    """Test date handling functions using a list of dates (case 2)."""
    dates = [np.datetime64("2021-01-01T00:00:00") + i * np.timedelta64(24, "h") for i in range(0, 10)]
    assert len(dates) == 10

    assert dates[0] == as_first_date("0%", dates)
    assert dates[0] == as_last_date("0%", dates)

    assert dates[0] == as_first_date("0.01%", dates)
    assert dates[0] == as_last_date("0.01%", dates)

    assert dates[0] == as_first_date("9.99%", dates)
    assert dates[0] == as_last_date("9.99%", dates)

    assert dates[0] == as_first_date("10%", dates)
    assert dates[0] == as_last_date("10%", dates)

    assert dates[1] == as_first_date("10.01%", dates)
    assert dates[0] == as_last_date("10.01%", dates)

    assert dates[1] == as_first_date("19.99%", dates)
    assert dates[0] == as_last_date("19.99%", dates)

    assert dates[1] == as_first_date("20%", dates)
    assert dates[1] == as_last_date("20%", dates)

    assert dates[2] == as_first_date("20.01%", dates)
    assert dates[1] == as_last_date("20.01%", dates)

    assert dates[-2] == as_first_date("89.99%", dates)
    assert dates[-3] == as_last_date("89.99%", dates)

    assert dates[-2] == as_first_date("90%", dates)
    assert dates[-2] == as_last_date("90%", dates)

    assert dates[-1] == as_first_date("90.01%", dates)
    assert dates[-2] == as_last_date("90.01%", dates)

    assert dates[-1] == as_first_date("99.99%", dates)
    assert dates[-2] == as_last_date("99.99%", dates)

    assert dates[-1] == as_first_date("100%", dates)
    assert dates[-1] == as_last_date("100%", dates)


@mockup_open_zarr
def test_slice_1() -> None:
    """Test slicing a dataset (case 1)."""
    test = DatasetTester("test-2021-2021-6h-o96-abcd")
    test.run(
        expected_class=Zarr,
        expected_length=365 * 1 * 4,
        expected_shape=(365 * 1 * 4, 4, 1, VALUES),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        date_to_row=lambda date: simple_row(date, "abcd"),
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=6),
        statistics_reference_dataset=None,
        statistics_reference_variables=None,
    )


@mockup_open_zarr
def test_slice_2() -> None:
    """Test slicing a dataset (case 2)."""
    test = DatasetTester([f"test-{year}-{year}-12h-o96-abcd" for year in range(1940, 2023)])
    test.run(
        expected_class=Concat,
        expected_length=60632,
        expected_shape=(60632, 4, 1, VALUES),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        date_to_row=lambda date: simple_row(date, "abcd"),
        start_date=datetime.datetime(1940, 1, 1),
        time_increment=datetime.timedelta(hours=12),
        statistics_reference_dataset=None,
        statistics_reference_variables=None,
    )


@mockup_open_zarr
def test_slice_3() -> None:
    """Test slicing a dataset (case 3)."""
    test = DatasetTester(
        [f"test-2020-2020-6h-o96-{vars}" for vars in ("abcd", "efgh", "ijkl", "mnop", "qrst", "uvwx", "yz")]
    )
    test.run(
        expected_class=Join,
        expected_length=366 * 4,
        date_to_row=lambda date: simple_row(date, "abcdefghijklmnopqrstuvwxyz"),
        start_date=datetime.datetime(2020, 1, 1),
        time_increment=datetime.timedelta(hours=6),
        expected_shape=(366 * 4, 26, 1, VALUES),
        expected_variables="abcdefghijklmnopqrstuvwxyz",
        expected_name_to_index="abcdefghijklmnopqrstuvwxyz",
        statistics_reference_dataset=None,
        statistics_reference_variables=None,
    )


@mockup_open_zarr
def test_slice_4() -> None:
    """Test slicing a dataset (case 4)."""
    test = DatasetTester([f"test-2020-2020-1h-o96-{vars}" for vars in ("abcd", "cd", "a", "c")])
    test.run(
        expected_class=Select,
        expected_length=8784,
        date_to_row=lambda date: simple_row(date, "abcd"),
        start_date=datetime.datetime(2020, 1, 1),
        time_increment=datetime.timedelta(hours=1),
        expected_shape=(8784, 4, 1, VALUES),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        statistics_reference_dataset=None,
        statistics_reference_variables=None,
    )


@mockup_open_zarr
def test_slice_5() -> None:
    """Test slicing a dataset (case 5)."""
    test = DatasetTester(
        [f"test-{year}-{year}-6h-o96-abcd" for year in range(2010, 2020)],
        frequency=18,
    )
    test.run(
        expected_class=Subset,
        expected_length=4870,
        date_to_row=lambda date: simple_row(date, "abcd"),
        start_date=datetime.datetime(2010, 1, 1),
        time_increment=datetime.timedelta(hours=18),
        expected_shape=(4870, 4, 1, VALUES),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        statistics_reference_dataset=None,
        statistics_reference_variables=None,
    )


@mockup_open_zarr
def test_ensemble_1() -> None:
    """Test ensemble datasets (case 1)."""
    test = DatasetTester(
        ensemble=[
            "test-2021-2021-6h-o96-abcd-1-10",
            "test-2021-2021-6h-o96-abcd-2-1",
        ]
    )
    test.run(
        expected_class=Ensemble,
        expected_length=365 * 1 * 4,
        expected_shape=(365 * 1 * 4, 4, 11, VALUES),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        date_to_row=lambda date: make_row(
            [_(date, "a", 1, i) for i in range(10)] + [_(date, "a", 2, 0)],
            [_(date, "b", 1, i) for i in range(10)] + [_(date, "b", 2, 0)],
            [_(date, "c", 1, i) for i in range(10)] + [_(date, "c", 2, 0)],
            [_(date, "d", 1, i) for i in range(10)] + [_(date, "d", 2, 0)],
            ensemble=True,
        ),
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=6),
        statistics_reference_dataset=None,
        statistics_reference_variables=None,
    )


@mockup_open_zarr
def test_ensemble_2() -> None:
    """Test ensemble datasets (case 2)."""
    test = DatasetTester(
        ensemble=[
            "test-2021-2021-6h-o96-abcd-1-10",
            "test-2021-2021-6h-o96-abcd-2-1",
            "test-2021-2021-6h-o96-abcd-3-5",
        ]
    )
    test.run(
        expected_class=Ensemble,
        expected_length=365 * 1 * 4,
        expected_shape=(365 * 1 * 4, 4, 16, VALUES),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        date_to_row=lambda date: make_row(
            [_(date, "a", 1, i) for i in range(10)] + [_(date, "a", 2, 0)] + [_(date, "a", 3, i) for i in range(5)],
            [_(date, "b", 1, i) for i in range(10)] + [_(date, "b", 2, 0)] + [_(date, "b", 3, i) for i in range(5)],
            [_(date, "c", 1, i) for i in range(10)] + [_(date, "c", 2, 0)] + [_(date, "c", 3, i) for i in range(5)],
            [_(date, "d", 1, i) for i in range(10)] + [_(date, "d", 2, 0)] + [_(date, "d", 3, i) for i in range(5)],
            ensemble=True,
        ),
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=6),
        statistics_reference_dataset=None,
        statistics_reference_variables=None,
    )


@mockup_open_zarr
def test_ensemble_3() -> None:
    """Test ensemble datasets (case 3)."""
    test = DatasetTester(
        ensemble=[
            {"dataset": "test-2021-2021-6h-o96-abcd-1-10", "frequency": 12},
            {"dataset": "test-2021-2021-6h-o96-abcd-2-1", "frequency": 12},
            {"dataset": "test-2021-2021-6h-o96-abcd-3-5", "frequency": 12},
        ]
    )
    test.run(
        expected_class=Ensemble,
        expected_length=365 * 1 * 2,
        expected_shape=(365 * 1 * 2, 4, 16, VALUES),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        date_to_row=lambda date: make_row(
            [_(date, "a", 1, i) for i in range(10)] + [_(date, "a", 2, 0)] + [_(date, "a", 3, i) for i in range(5)],
            [_(date, "b", 1, i) for i in range(10)] + [_(date, "b", 2, 0)] + [_(date, "b", 3, i) for i in range(5)],
            [_(date, "c", 1, i) for i in range(10)] + [_(date, "c", 2, 0)] + [_(date, "c", 3, i) for i in range(5)],
            [_(date, "d", 1, i) for i in range(10)] + [_(date, "d", 2, 0)] + [_(date, "d", 3, i) for i in range(5)],
            ensemble=True,
        ),
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=12),
        statistics_reference_dataset=None,
        statistics_reference_variables=None,
    )


@mockup_open_zarr
def test_grids() -> None:
    """Test datasets with different grids."""
    test = DatasetTester(
        grids=[
            "test-2021-2021-6h-o96-abcd-1-1",  # Default is 10 gridpoints
            "test-2021-2021-6h-o96-abcd-2-1-25",  # 25 gridpoints
        ]
    )
    test.run(
        expected_class=GridsBase,
        expected_length=365 * 1 * 4,
        expected_shape=(365 * 1 * 4, 4, 1, VALUES + 25),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        date_to_row=lambda date: make_row(
            [
                _(date, "a", 1),
                _(date, "a", 2, values=25),
            ],
            [
                _(date, "b", 1),
                _(date, "b", 2, values=25),
            ],
            [
                _(date, "c", 1),
                _(date, "c", 2, values=25),
            ],
            [
                _(date, "d", 1),
                _(date, "d", 2, values=25),
            ],
            grid=True,
        ),
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=6),
        statistics_reference_dataset="test-2021-2021-6h-o96-abcd-1-1",
        statistics_reference_variables="abcd",
    )

    ds1 = open_dataset("test-2021-2021-6h-o96-abcd-1-1")
    ds2 = open_dataset("test-2021-2021-6h-o96-abcd-2-1-25")

    assert (test.ds.longitudes == np.concatenate([ds1.longitudes, ds2.longitudes])).all()
    assert (test.ds.latitudes == np.concatenate([ds1.latitudes, ds2.latitudes])).all()


@mockup_open_zarr
def test_statistics() -> None:
    """Test datasets with statistics."""
    test = DatasetTester(
        "test-2021-2021-6h-o96-abcd",
        statistics="test-2000-2010-6h-o96-abcd",
    )
    test.run(
        expected_class=Statistics,
        expected_length=365 * 4,
        date_to_row=lambda date: simple_row(date, "abcd"),
        start_date=datetime.datetime(2021, 1, 1),
        time_increment=datetime.timedelta(hours=6),
        expected_shape=(365 * 4, 4, 1, VALUES),
        expected_variables="abcd",
        expected_name_to_index="abcd",
        statistics_reference_dataset="test-2000-2010-6h-o96-abcd",
        statistics_reference_variables="abcd",
    )


@mockup_open_zarr
def test_cropping() -> None:
    """Test cropping a dataset."""
    test = DatasetTester(
        "test-2021-2021-6h-o96-abcd",
        area=(18, 11, 11, 18),
    )
    assert test.ds.shape == (365 * 4, 4, 1, 8)


@mockup_open_zarr
def test_invalid_trim_edge() -> None:
    """Test that exception raised when attempting to trim a 1D dataset"""
    with pytest.raises(ValueError):
        DatasetTester(
            "test-2021-2021-6h-o96-abcd",
            trim_edge=(1, 2, 3, 4),
        )


if __name__ == "__main__":
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
