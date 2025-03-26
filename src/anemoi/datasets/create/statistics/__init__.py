# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import glob
import hashlib
import json
import logging
import os
import pickle
import shutil
import socket
from typing import Any
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import tqdm
from anemoi.utils.provenance import gather_provenance_info
from numpy.typing import NDArray

from ..check import check_data_values
from .summary import Summary

LOG = logging.getLogger(__name__)


def default_statistics_dates(dates: list[datetime.datetime]) -> tuple[datetime.datetime, datetime.datetime]:
    """Calculate default statistics dates based on the given list of dates.

    Parameters
    ----------
    dates : list of datetime.datetime
        List of datetime objects representing dates.

    Returns
    -------
    tuple of datetime.datetime
        A tuple containing the default start and end dates.
    """

    def to_datetime(d):
        if isinstance(d, np.datetime64):
            return d.tolist()
        assert isinstance(d, datetime.datetime), d
        return d

    first = dates[0]
    last = dates[-1]

    first = to_datetime(first)
    last = to_datetime(last)

    n_years = round((last - first).total_seconds() / (365.25 * 24 * 60 * 60))

    if n_years < 10:
        # leave out 20% of the data
        k = int(len(dates) * 0.8)
        end = dates[k - 1]
        LOG.info(f"Number of years {n_years} < 10, leaving out 20%. {end=}")
        return dates[0], end

    delta = 1
    if n_years >= 20:
        delta = 3
    LOG.info(f"Number of years {n_years}, leaving out {delta} years.")
    end_year = last.year - delta

    end = max(d for d in dates if to_datetime(d).year == end_year)
    return dates[0], end


def to_datetime(date: Union[str, datetime.datetime]) -> np.datetime64:
    """Convert a date to numpy datetime64 format.

    Parameters
    ----------
    date : str or datetime.datetime
        The date to convert.

    Returns
    -------
    numpy.datetime64
        The converted date.
    """
    if isinstance(date, str):
        return np.datetime64(date)
    if isinstance(date, datetime.datetime):
        return np.datetime64(date, "s")
    return date


def to_datetimes(dates: list[Union[str, datetime.datetime]]) -> list[np.datetime64]:
    """Convert a list of dates to numpy datetime64 format.

    Parameters
    ----------
    dates : list of str or datetime.datetime
        List of dates to convert.

    Returns
    -------
    list of numpy.datetime64
        List of converted dates.
    """
    return [to_datetime(d) for d in dates]


def fix_variance(x: float, name: str, count: NDArray[Any], sums: NDArray[Any], squares: NDArray[Any]) -> float:
    """Fix negative variance values due to numerical errors.

    Parameters
    ----------
    x : float
        The variance value.
    name : str
        The variable name.
    count : numpy.ndarray
        The count array.
    sums : numpy.ndarray
        The sums array.
    squares : numpy.ndarray
        The squares array.

    Returns
    -------
    float
        The fixed variance value.
    """
    assert count.shape == sums.shape == squares.shape
    assert isinstance(x, float)

    mean = sums / count
    assert mean.shape == count.shape

    if x >= 0:
        return x

    LOG.warning(f"Negative variance for {name=}, variance={x}")
    magnitude = np.sqrt((squares / count + mean * mean) / 2)
    LOG.warning(f"square / count - mean * mean =  {squares/count} - {mean*mean} = {squares/count - mean*mean}")
    LOG.warning(f"Variable span order of magnitude is {magnitude}.")
    LOG.warning(f"Count is {count}.")

    variances = squares / count - mean * mean
    assert variances.shape == squares.shape == mean.shape
    if np.all(variances >= 0):
        LOG.warning(f"All individual variances for {name} are positive, setting variance to 0.")
        return 0

    # if abs(x) < magnitude * 1e-6 and abs(x) < range * 1e-6:
    #     LOG.warning("Variance is negative but very small.")
    #     variances = squares / count - mean * mean
    #     return 0

    LOG.warning(f"ERROR at least one individual variance is negative ({np.nanmin(variances)}).")
    return 0


def check_variance(
    x: NDArray[Any],
    variables_names: list[str],
    minimum: NDArray[Any],
    maximum: NDArray[Any],
    mean: NDArray[Any],
    count: NDArray[Any],
    sums: NDArray[Any],
    squares: NDArray[Any],
) -> None:
    """Check for negative variance values and raise an error if found.

    Parameters
    ----------
    x : numpy.ndarray
        The variance array.
    variables_names : list of str
        List of variable names.
    minimum : numpy.ndarray
        The minimum values array.
    maximum : numpy.ndarray
        The maximum values array.
    mean : numpy.ndarray
        The mean values array.
    count : numpy.ndarray
        The count array.
    sums : numpy.ndarray
        The sums array.
    squares : numpy.ndarray
        The squares array.

    Raises
    ------
    ValueError
        If negative variance is found.
    """
    if (x >= 0).all():
        return
    print(x)
    print(variables_names)
    for i, (name, y) in enumerate(zip(variables_names, x)):
        if y >= 0:
            continue
        print("---")
        print(f"❗ Negative variance for {name=}, variance={y}")
        print(f" min={minimum[i]} max={maximum[i]} mean={mean[i]} count={count[i]} sums={sums[i]} squares={squares[i]}")
        print(f" -> sums: min={np.min(sums[i])}, max={np.max(sums[i])}, argmin={np.argmin(sums[i])}")
        print(f" -> squares: min={np.min(squares[i])}, max={np.max(squares[i])}, argmin={np.argmin(squares[i])}")
        print(f" -> count: min={np.min(count[i])}, max={np.max(count[i])}, argmin={np.argmin(count[i])}")
        print(
            f" squares / count - mean * mean =  {squares[i] / count[i]} - {mean[i] * mean[i]} = {squares[i] / count[i] - mean[i] * mean[i]}"
        )

    raise ValueError("Negative variance")


def compute_statistics(
    array: NDArray[Any], check_variables_names: Optional[List[str]] = None, allow_nans: bool = False
) -> dict[str, np.ndarray]:
    """Compute statistics for a given array, provides minimum, maximum, sum, squares, count and has_nans as a dictionary.

    Parameters
    ----------
    array : numpy.ndarray
        The array to compute statistics for.
    check_variables_names : list of str, optional
        List of variable names to check. Defaults to None.
    allow_nans : bool, optional
        Whether to allow NaN values. Defaults to False.

    Returns
    -------
    dict of str to numpy.ndarray
        A dictionary containing the computed statistics.
    """
    LOG.info(f"Computing statistics for {array.shape} array")
    nvars = array.shape[1]

    LOG.debug(f"Stats {nvars}, {array.shape}, {check_variables_names}")
    if check_variables_names:
        assert nvars == len(check_variables_names), (nvars, check_variables_names)
    stats_shape = (array.shape[0], nvars)

    count = np.zeros(stats_shape, dtype=np.int64)
    sums = np.zeros(stats_shape, dtype=np.float64)
    squares = np.zeros(stats_shape, dtype=np.float64)
    minimum = np.zeros(stats_shape, dtype=np.float64)
    maximum = np.zeros(stats_shape, dtype=np.float64)
    has_nans = np.zeros(stats_shape, dtype=np.bool_)

    for i, chunk in tqdm.tqdm(enumerate(array), delay=1, total=array.shape[0], desc="Computing statistics"):
        values = chunk.reshape((nvars, -1))

        for j, name in enumerate(check_variables_names):
            check_data_values(values[j, :], name=name, allow_nans=allow_nans)
            if np.isnan(values[j, :]).all():
                # LOG.warning(f"All NaN values for {name} ({j}) for date {i}")
                LOG.warning(f"All NaN values for {name} ({j}) for date {i}")

        # Ignore NaN values
        minimum[i] = np.nanmin(values, axis=1)
        maximum[i] = np.nanmax(values, axis=1)
        sums[i] = np.nansum(values, axis=1)
        squares[i] = np.nansum(values * values, axis=1)
        count[i] = np.sum(~np.isnan(values), axis=1)
        has_nans[i] = np.isnan(values).any()

    LOG.info(f"Statistics computed for {nvars} variables.")

    return {
        "minimum": minimum,
        "maximum": maximum,
        "sums": sums,
        "squares": squares,
        "count": count,
        "has_nans": has_nans,
    }


class TmpStatistics:
    """Temporary statistics storage class."""

    version = 3
    # Used in parrallel, during data loading,
    # to write statistics in pickled npz files.
    # can provide statistics for a subset of dates.

    def __init__(self, dirname: str, overwrite: bool = False) -> None:
        """Initialize TmpStatistics.

        Parameters
        ----------
        dirname : str
            Directory name for storing statistics.
        overwrite : bool, optional
            Whether to overwrite existing files. Defaults to False.
        """
        self.dirname = dirname
        self.overwrite = overwrite

    def add_provenance(self, **kwargs: dict) -> None:
        """Add provenance information.

        Parameters
        ----------
        **kwargs : dict
            Additional provenance information.
        """
        self.create(exist_ok=True)
        path = os.path.join(self.dirname, "provenance.json")
        if os.path.exists(path):
            return
        out = dict(provenance=gather_provenance_info(), **kwargs)
        with open(path, "w") as f:
            json.dump(out, f)

    def create(self, exist_ok: bool) -> None:
        """Create the directory for storing statistics.

        Parameters
        ----------
        exist_ok : bool
            Whether to ignore if the directory already exists.
        """
        os.makedirs(self.dirname, exist_ok=exist_ok)

    def delete(self) -> None:
        """Delete the directory for storing statistics."""
        try:
            shutil.rmtree(self.dirname)
        except FileNotFoundError:
            pass

    def write(self, key: str, data: any, dates: list[datetime.datetime]) -> None:
        """Write statistics data to a file.

        Parameters
        ----------
        key : str
            The key for the data.
        data : any
            The data to write.
        dates : list of datetime.datetime
            List of dates associated with the data.
        """
        self.create(exist_ok=True)
        h = hashlib.sha256(str(dates).encode("utf-8")).hexdigest()
        path = os.path.join(self.dirname, f"{h}.npz")

        if not self.overwrite:
            assert not os.path.exists(path), f"{path} already exists"

        tmp_path = path + f".tmp-{os.getpid()}-on-{socket.gethostname()}"
        with open(tmp_path, "wb") as f:
            pickle.dump((key, dates, data), f)
        shutil.move(tmp_path, path)

        LOG.debug(f"Written statistics data for {len(dates)} dates in {path} ({dates})")

    def _gather_data(self) -> tuple[str, list[datetime.datetime], dict]:
        """Gather data from stored files.

        Yields
        ------
        tuple of str, list of datetime.datetime, dict
            A tuple containing key, dates, and data.
        """
        # use glob to read all pickles
        files = glob.glob(self.dirname + "/*.npz")
        LOG.debug(f"Reading stats data, found {len(files)} files in {self.dirname}")
        assert len(files) > 0, f"No files found in {self.dirname}"
        for f in files:
            with open(f, "rb") as f:
                yield pickle.load(f)

    def get_aggregated(self, *args: Any, **kwargs: Any) -> Summary:
        """Get aggregated statistics.

        Parameters
        ----------
        *args : Any
            Additional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Summary
            The aggregated statistics summary.
        """
        aggregator = StatAggregator(self, *args, **kwargs)
        return aggregator.aggregate()

    def __str__(self) -> str:
        """String representation of TmpStatistics.

        Returns
        -------
        str
            The string representation.
        """
        return f"TmpStatistics({self.dirname})"


class StatAggregator:
    """Statistics aggregator class."""

    NAMES = ["minimum", "maximum", "sums", "squares", "count", "has_nans"]

    def __init__(
        self, owner: TmpStatistics, dates: list[datetime.datetime], variables_names: list[str], allow_nans: bool
    ) -> None:
        """Initialize StatAggregator.

        Parameters
        ----------
        owner : TmpStatistics
            The owner TmpStatistics instance.
        dates : list of datetime.datetime
            List of dates to aggregate.
        variables_names : list of str
            List of variable names.
        allow_nans : bool
            Whether to allow NaN values.
        """
        dates = sorted(dates)
        dates = to_datetimes(dates)
        assert dates, "No dates selected"
        self.owner = owner
        self.dates = dates
        self._number_of_dates = len(dates)
        self._set_of_dates = set(dates)
        self.variables_names = variables_names
        self.allow_nans = allow_nans

        self.shape = (self._number_of_dates, len(self.variables_names))
        LOG.debug(f"Aggregating statistics on shape={self.shape}. Variables : {self.variables_names}")

        self.minimum = np.full(self.shape, np.nan, dtype=np.float64)
        self.maximum = np.full(self.shape, np.nan, dtype=np.float64)
        self.sums = np.full(self.shape, np.nan, dtype=np.float64)
        self.squares = np.full(self.shape, np.nan, dtype=np.float64)
        self.count = np.full(self.shape, -1, dtype=np.int64)
        self.has_nans = np.full(self.shape, False, dtype=np.bool_)

        self._read()

    def _read(self) -> None:
        """Read and aggregate statistics data from files."""

        def check_type(a, b):
            if not isinstance(a, set):
                a = set(list(a))
            if not isinstance(b, set):
                b = set(list(b))
            a = next(iter(a)) if a else None
            b = next(iter(b)) if b else None
            assert type(a) is type(b), (type(a), type(b))

        found = set()
        offset = 0

        for _, _dates, stats in self.owner._gather_data():
            assert isinstance(stats, dict), stats
            assert stats["minimum"].shape[0] == len(_dates), (stats["minimum"].shape, len(_dates))
            assert stats["minimum"].shape[1] == len(self.variables_names), (
                stats["minimum"].shape,
                len(self.variables_names),
            )
            for n in self.NAMES:
                assert n in stats, (n, list(stats.keys()))
            _dates = to_datetimes(_dates)
            check_type(_dates, self._set_of_dates)
            if found:
                check_type(found, self._set_of_dates)
                assert found.isdisjoint(_dates), "Duplicate dates found in precomputed statistics"

            # filter dates
            dates = set(_dates) & self._set_of_dates

            if not dates:
                # dates have been completely filtered for this chunk
                continue

            # filter data
            bitmap = np.array([d in self._set_of_dates for d in _dates])
            for k in self.NAMES:
                stats[k] = stats[k][bitmap]

            assert stats["minimum"].shape[0] == len(dates), (stats["minimum"].shape, len(dates))

            # store data in self
            found |= set(dates)
            for name in self.NAMES:
                array = getattr(self, name)
                assert stats[name].shape[0] == len(dates), (stats[name].shape, len(dates))
                array[offset : offset + len(dates)] = stats[name]
            offset += len(dates)

        for d in self.dates:
            assert d in found, f"Statistics for date {d} not precomputed."
        assert self._number_of_dates == len(found), "Not all dates found in precomputed statistics"
        assert self._number_of_dates == offset, "Not all dates found in precomputed statistics."
        LOG.debug(f"Statistics for {len(found)} dates found.")

    def aggregate(self) -> Summary:
        """Aggregate the statistics data.

        Returns
        -------
        Summary
            The aggregated statistics summary.
        """
        minimum = np.nanmin(self.minimum, axis=0)
        maximum = np.nanmax(self.maximum, axis=0)

        sums = np.nansum(self.sums, axis=0)
        squares = np.nansum(self.squares, axis=0)
        count = np.nansum(self.count, axis=0)
        has_nans = np.any(self.has_nans, axis=0)
        assert sums.shape == count.shape == squares.shape == minimum.shape == maximum.shape

        mean = sums / count
        assert mean.shape == minimum.shape

        x = squares / count - mean * mean
        assert x.shape == minimum.shape

        for i, name in enumerate(self.variables_names):
            # remove negative variance due to numerical errors
            x[i] = fix_variance(x[i], name, self.count[i : i + 1], self.sums[i : i + 1], self.squares[i : i + 1])

        for i, name in enumerate(self.variables_names):
            check_variance(
                x[i : i + 1],
                [name],
                minimum[i : i + 1],
                maximum[i : i + 1],
                mean[i : i + 1],
                count[i : i + 1],
                sums[i : i + 1],
                squares[i : i + 1],
            )
            check_data_values(np.array([mean[i]]), name=name, allow_nans=False)

        stdev = np.sqrt(x)

        return Summary(
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            count=count,
            sums=sums,
            squares=squares,
            stdev=stdev,
            variables_names=self.variables_names,
            has_nans=has_nans,
        )
