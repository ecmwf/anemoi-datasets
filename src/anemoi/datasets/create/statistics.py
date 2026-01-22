# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import hashlib
import logging
import pickle
from collections.abc import Callable
from functools import reduce
from typing import Any

import numpy as np
import tqdm
from numpy.typing import NDArray

LOG = logging.getLogger(__name__)

STATISTICS = ("mean", "minimum", "maximum", "stdev")


def _(t):
    return str(t)[:50] if len(str(t)) < 50 else str(t)[:47] + "..."


class _State:
    def __init__(self, group: int, start: int, end: int, collector: "StatisticsCollector") -> None:
        self.group = group
        self.start = start
        self.end = end
        self.collector = collector
        self.missing_tendencies_count = collector.missing_tendencies_count()

    def __repr__(self):
        return f"_State(group={self.group}, start={self.start}, end={self.end}, missing_tendencies_count={self.missing_tendencies_count})"


class _Base:
    def __init__(self, column_names: list[str] | None = None) -> None:
        self._column_names = column_names

    def __repr__(self, extra: str = ""):
        return f"{self.__class__.__name__}(column_names={_(self._column_names)}" + (f", {extra}" if extra else "") + ")"


class _CollectorBase(_Base):
    def __init__(self, column_names: list[str]) -> None:
        num_columns = len(column_names)
        self._count = np.zeros(num_columns, dtype=np.int64)
        self._min = np.full(num_columns, np.inf, dtype=np.float64)
        self._max = np.full(num_columns, -np.inf, dtype=np.float64)
        self._column_names = column_names
        self._mean = np.zeros(num_columns, dtype=np.float64)
        self._m2 = np.zeros(num_columns, dtype=np.float64)

    def __repr__(self, extra: str = ""):
        return super().__repr__(
            f"count={_(self._count)}, min={_(self._min)}, max={_(self._max)}, mean={_(self._mean)}, m2={_(self._m2)}"
            + (f", {extra}" if extra else "")
        )

    @classmethod
    def _merge(cls, a: "_CollectorBase", b: "_CollectorBase", result: "_CollectorBase") -> "_CollectorBase":
        if a._column_names != b._column_names:
            raise ValueError("Cannot merge collectors with different column names")
        count = a._count + b._count
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(
                count > 0,
                (a._count * a._mean + b._count * b._mean) / count,
                0.0,
            )
            delta = b._mean - a._mean
            m2 = np.where(
                count > 0,
                a._m2 + b._m2 + delta**2 * a._count * b._count / count,
                0.0,
            )
        assert np.all(count >= 0), "Negative count in merge"
        assert np.all(np.isfinite(mean) | (count == 0)), "Non-finite mean in merge"
        assert np.all(np.isfinite(m2) | (count == 0)), "Non-finite m2 in merge"
        min_ = np.minimum(a._min, b._min)
        max_ = np.maximum(a._max, b._max)
        result._count = count
        result._mean = mean
        result._m2 = m2
        result._min = min_
        result._max = max_

    def merge(self, other: "_CollectorBase") -> "_CollectorBase":
        result = _CollectorBase(self._column_names)
        self._merge(self, other, result)
        return result

    def update(self, data: NDArray[np.float64]) -> None:
        # data shape: (n_samples, n_columns)
        data = data.astype(np.float64)

        # Create mask for valid data (not NaN)
        valid_mask = ~np.isnan(data)

        # Process each column
        for col_idx in range(data.shape[1]):
            col_data = data[:, col_idx]
            col_mask = valid_mask[:, col_idx]
            valid_data = col_data[col_mask]

            if valid_data.size == 0:
                continue

            n = len(valid_data)
            old_mean = self._mean[col_idx]
            old_count = self._count[col_idx]
            new_count = old_count + n

            # Batch mean
            batch_mean = np.mean(valid_data)

            # Update combined mean
            self._mean[col_idx] = (old_count * old_mean + n * batch_mean) / new_count

            # Update M2 using parallel algorithm formula
            batch_m2 = np.sum((valid_data - batch_mean) ** 2)
            delta = batch_mean - old_mean
            self._m2[col_idx] = self._m2[col_idx] + batch_m2 + (old_count * n * delta * delta) / new_count

            self._count[col_idx] = new_count

            # Update min/max
            self._min[col_idx] = min(self._min[col_idx], np.min(valid_data))
            self._max[col_idx] = max(self._max[col_idx], np.max(valid_data))

    def statistics(self) -> dict[str, NDArray[np.float64]]:
        # Returns arrays for each statistic across all columns
        result = {}

        for col_idx in range(len(self._count)):
            if self._count[col_idx] == 0:
                LOG.warning(f"Column {self._column_names[col_idx]}: no statistics collected")

        # Compute variance using Welford's M2
        # variance = M2 / count (except when count == 0, then variance is set to NaN)
        variance = np.divide(
            self._m2,
            self._count,
            out=np.full_like(self._m2, np.nan),
            where=self._count > 0,
        )

        # Check for negative variance (numerical errors)
        negative_mask = variance < 0
        if np.any(negative_mask):
            for col_idx in np.where(negative_mask)[0]:
                var = variance[col_idx]
                relative_error = abs(var) / max(abs(self._m2[col_idx]), abs(self._mean[col_idx] ** 2), 1e-100)
                if relative_error > 1e-6:
                    msg = f"Negative variance {var} for column {self._column_names[col_idx]}, {relative_error=}"
                    msg += f" m2={self._m2[col_idx]}, count={self._count[col_idx]}, mean={self._mean[col_idx]}"
                    msg += f", min={self._min[col_idx]}, max={self._max[col_idx]}"
                    LOG.error(msg)
                    raise ValueError(msg)
                variance[col_idx] = 0.0

        stdev = np.sqrt(variance)

        # Set NaN for columns with no data
        no_data_mask = self._count == 0
        result = {
            "mean": np.where(no_data_mask, np.nan, self._mean),
            "minimum": np.where(no_data_mask, np.nan, self._min),
            "maximum": np.where(no_data_mask, np.nan, self._max),
            "stdev": np.where(no_data_mask, np.nan, stdev),
        }

        return result


class _Collector(_CollectorBase):
    pass


class _TendencyCollector(_CollectorBase):
    def __init__(self, column_names: list[str], delta: int) -> None:
        super().__init__(column_names)
        self._delta = delta
        # Only keep a sliding window of the last 'delta' rows
        self._window = None
        self._last_window_date = None  # Store only last date for validation

        # Summary of dates seen (instead of storing full arrays)
        self._first_date = None
        self._last_date = None
        self._n_dates = 0

        # Offset tracking
        # _first_date_offset: offset from group start to first filtered date (in dataset indices)
        # _unfiltered_dates_seen: cumulative count of unfiltered dates (for computing offset across batches)
        self._first_date_offset = None
        self._unfiltered_dates_seen = 0

        # Number of dates at the start that couldn't have tendencies computed
        # A "missing" date d means we could not compute value(d) - value(d - delta)
        # because d - delta is before available data.
        # The first delta dates of the entire (filtered) dataset are permanently missing:
        # there's no data before them, so their tendencies can never be computed.
        # This is set once on the first update call and not changed afterwards.
        self._n_missing = None

    def __repr__(self, extra: str = ""):
        return super().__repr__(
            f"delta={_(self._delta)}, n_dates={self._n_dates}, n_missing={self._n_missing}, "
            f"first_date={self._first_date}, last_date={self._last_date}, first_date_offset={self._first_date_offset}"
            + (f", {extra}" if extra else "")
        )

    @classmethod
    def _merge(cls, a: "_TendencyCollector", b: "_TendencyCollector", result: "_TendencyCollector"):
        assert (
            a._window is None and b._window is None
        ), "Merging tendency collectors with non-empty windows is not supported"

        if a._delta != b._delta:
            raise ValueError("Cannot merge tendency collectors with different deltas")
        _CollectorBase._merge(a, b, result)
        result._delta = a._delta
        return result

    def merge(self, other: "_TendencyCollector") -> "_TendencyCollector":
        result = _TendencyCollector(self._column_names, self._delta)
        self._merge(self, other, result)
        return result

    def update(
        self,
        data: NDArray[np.float64],
        dates,
        unfiltered_batch_size: int | None = None,
        first_offset_in_batch: int | None = None,
    ) -> None:
        assert len(data) == len(dates), f"Data and dates length mismatch: {len(data)} vs {len(dates)}"
        assert np.all(np.array(dates[1:]) >= np.array(dates[:-1])), ("Dates must be sorted", dates)
        if self._last_window_date is not None and len(dates) > 0:
            assert dates[0] > self._last_window_date, (
                f"New dates must follow window dates chronologically: "
                f"window ends at {self._last_window_date}, new data starts at {dates[0]}"
            )

        # Concatenate window with new data for tendency computation
        combined = data if self._window is None else np.concatenate([self._window, data], axis=0)

        # Compute tendencies wherever we have enough history
        if len(combined) > self._delta:
            tendencies = combined[self._delta :] - combined[: -self._delta]
            _CollectorBase.update(self, tendencies)

        # Track date summary and offset (only on first call do we set n_missing and offset)
        if self._first_date is None and len(dates) > 0:
            self._first_date = dates[0]
            # On first call, the missing count is min(delta, number of dates in first batch)
            self._n_missing = min(self._delta, len(dates))
            # Compute offset from group start to first filtered date
            if first_offset_in_batch is not None:
                self._first_date_offset = self._unfiltered_dates_seen + first_offset_in_batch

        if len(dates) > 0:
            self._last_date = dates[-1]
        self._n_dates += len(dates)

        # Track unfiltered dates for offset computation across batches
        if unfiltered_batch_size is not None:
            self._unfiltered_dates_seen += unfiltered_batch_size

        # Update sliding window: keep only last 'delta' rows
        self._window = np.array(combined[-self._delta :], copy=True)
        if len(dates) > 0:
            self._last_window_date = dates[-1]

    def finalise(self) -> None:
        pass

    def adjust_partial_statistics(self, dataset, start, n_missing, filter_func) -> None:
        """Adjust statistics by computing tendencies for the missing dates at the start of this group.

        Uses data from the previous group (before start) as the window.

        Parameters
        ----------
        dataset : Dataset
            The dataset containing data and dates.
        start : int
            The start index of this group in the dataset.
        n_missing : int
            The number of missing dates at the start (after filtering).
        filter_func : Callable
            The filter function to apply to dates/data.
        """
        if n_missing == 0:
            return

        assert (
            self._first_date_offset is not None
        ), f"_first_date_offset not set for tendency collector with first_date={self._first_date}"
        first_date_idx = start + self._first_date_offset

        # Read n_missing dates starting from the first filtered date
        missing_dates = dataset.dates[first_date_idx : first_date_idx + n_missing]
        missing_data = dataset.data[first_date_idx : first_date_idx + n_missing]

        # Apply the filter with offset for tendency computation
        dates = filter_func(missing_dates, missing_dates, offset=-self._delta)
        data = filter_func(missing_data, missing_dates, offset=-self._delta)

        if len(dates) == 0:
            return

        self._window = dataset.data[first_date_idx - self._delta : first_date_idx]
        window_dates = dataset.dates[first_date_idx - self._delta : first_date_idx]
        self._last_window_date = window_dates[-1] if len(window_dates) > 0 else None
        self.update(data, dates)
        self._window = None
        self._last_window_date = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_window"] = None
        state["_last_window_date"] = None
        # _unfiltered_dates_seen is only needed during collection, reset after pickle
        state["_unfiltered_dates_seen"] = 0
        return state


class _ConstantsCollector(_Base):
    def __init__(self, index, column_names: list[str], name: str) -> None:
        super().__init__(column_names)
        self._index = index
        self._name = name
        self._is_constant = True
        self._first = None
        self._nans = None
        self._hash = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_hash"] = hashlib.sha256(self._first.tobytes()).hexdigest() if self._first is not None else None
        state["_first"] = None
        state["_nans"] = None
        return state

    def __repr__(self):
        return super().__repr__(
            f"index={self._index}, name={self._name}, is_constant={self._is_constant}, first={self._first}, nans={self._nans}"
        )

    @classmethod
    def _merge(
        cls, a: "_ConstantsCollector", b: "_ConstantsCollector", result: "_ConstantsCollector"
    ) -> "_ConstantsCollector":
        if a._index != b._index or a._name != b._name or a._column_names != b._column_names:
            raise ValueError("Cannot merge incompatible constants collectors")

        result._nans = a._nans.copy() if a._nans is not None else b._nans.copy() if b._nans is not None else None
        result._first = None
        result._is_constant = a._is_constant and b._is_constant and a._hash == b._hash
        result._hash = a._hash if a._hash == b._hash else None
        return result

    def merge(self, other: "_ConstantsCollector") -> "_ConstantsCollector":
        result = _ConstantsCollector(self._index, self._column_names, self._name)
        self._merge(self, other, result)
        return result

    def update(self, data: NDArray[np.float64]) -> None:

        if not self._is_constant:
            # No need to check further
            return

        data = data[:, self._index]

        if self._first is None:
            self._first = data[0].copy()
            self._nans = np.isnan(self._first)

        # Check for standard equality
        eq = data == self._first

        # Check where both are NaN
        both_nan = np.isnan(data) & self._nans

        # Combined check: All elements in all rows must satisfy one of the above

        if not np.all(eq | both_nan):
            LOG.debug(f"Variable {self._name} is not constant.")
            self._is_constant = False

    @property
    def is_constant(self) -> bool:
        return self._is_constant


def _all(array: np.array, dates: np.array) -> range:
    return array


class StatisticsCollector:
    """Main statistics collector interface.

    Collects statistics for multiple variables, including support for
    temporal tendency statistics. Handles NaN values and provides filtering
    capabilities for selective data collection.

    Parameters
    ----------
    variables_names : list[str] | None, optional
        Names of the variables to collect statistics for. If None, variables
        are numbered from 0.
    allow_nans : bool, optional
        Whether to allow NaN values in the data. Default is False.
    filter : Callable[[Any], range], optional
        Function to filter which dates to include. Default includes all dates.
    tendencies : list[int] | None, optional
        List of time deltas for tendency statistics. If None, no tendency
        statistics are collected.
    """

    def __init__(
        self,
        variables_names: list[str] | None = None,
        allow_nans: bool = False,
        filter: Callable[[Any], range] = _all,
        tendencies: list[int] | None = None,
        _collector: _Collector | None = None,
        _tendencies_collectors: dict[str, _TendencyCollector] | None = None,
        _constants_collectors: dict[str, _ConstantsCollector] | None = None,
    ) -> None:
        self._filter = filter

        self._collector = _collector
        self._variables_names = variables_names
        self._allow_nans = allow_nans
        self._tendencies = tendencies or {}
        self._tendencies_collectors = _tendencies_collectors or {}
        self._constants_collectors = _constants_collectors or {}

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(variables_names={self._variables_names}, "
            f"allow_nans={self._allow_nans}, tendencies={self._tendencies}, "
            f"collector={self._collector}, tendencies_collectors={self._tendencies_collectors}, "
            f"constants_collectors={self._constants_collectors})"
        )

    def merge(self, other: "StatisticsCollector") -> "StatisticsCollector":
        if not isinstance(other, StatisticsCollector):
            raise ValueError("Can only merge with another StatisticsCollector")
        if self._variables_names != other._variables_names:
            raise ValueError("Cannot merge StatisticsCollectors with different variable names")

        variables_names = self._variables_names

        if self._allow_nans != other._allow_nans:
            raise ValueError("Cannot merge StatisticsCollectors with different allow_nans settings")

        allow_nans = self._allow_nans

        if self._collector is None:
            return other
        elif other._collector is None:
            return self

        collector = self._collector.merge(other._collector)

        if set(self._tendencies.keys()) != set(other._tendencies.keys()):
            raise ValueError(
                f"Cannot merge StatisticsCollectors with different tendencies settings {self._tendencies.keys()} vs {other._tendencies.keys()}"
            )
        tendencies = self._tendencies

        if set(self._tendencies_collectors.keys()) != set(other._tendencies_collectors.keys()):
            raise ValueError(
                f"Cannot merge StatisticsCollectors with different tendencies collectors keys {self._tendencies_collectors.keys()} vs {other._tendencies_collectors.keys()}"
            )

        tendencies_collectors = {}
        for delta in self._tendencies_collectors.keys() | other._tendencies_collectors.keys():
            if delta not in self._tendencies_collectors or delta not in other._tendencies_collectors:
                LOG.error(f"Delta {delta} not in both collectors")
                LOG.error(f"Self tendencies: {self._tendencies_collectors}")
                LOG.error(f"Other tendencies: {other._tendencies_collectors}")
                LOG.error(f"Self tendencies keys: {list(self._tendencies_collectors.keys())}")
                LOG.error(f"Other tendencies keys: {list(other._tendencies_collectors.keys())}")
                raise ValueError(f"Cannot merge StatisticsCollectors with different tendencies settings, {delta=}")
            tendencies_collectors[delta] = self._tendencies_collectors[delta].merge(other._tendencies_collectors[delta])

        constants_collectors = {}
        for name in self._constants_collectors.keys() | other._constants_collectors.keys():
            if name not in self._constants_collectors or name not in other._constants_collectors:
                LOG.error(f"Constant {name} not in both collectors")
                LOG.error(f"Self constants: {self._constants_collectors}")
                LOG.error(f"Other constants: {other._constants_collectors}")
                LOG.error(f"Self constants keys: {list(self._constants_collectors.keys())}")
                LOG.error(f"Other constants keys: {list(other._constants_collectors.keys())}")
                raise ValueError(f"Cannot merge StatisticsCollectors with different constants settings, {name=}")
            constants_collectors[name] = self._constants_collectors[name].merge(other._constants_collectors[name])

        return StatisticsCollector(
            variables_names=variables_names,
            allow_nans=allow_nans,
            tendencies=tendencies,
            _tendencies_collectors=tendencies_collectors,
            _collector=collector,
            _constants_collectors=constants_collectors,
        )

    def missing_tendencies_count(self) -> dict[str, int]:
        """Return count of missing dates for each tendency collector."""
        return {name: c._n_missing or 0 for name, c in self._tendencies_collectors.items()}

    def collect(self, array: NDArray[np.float64], dates: Any) -> None:
        """Collect statistics from a batch of data.

        Initialises collectors on first call based on array shape and variable names.
        Updates both standard and tendency collectors with the new data.

        Parameters
        ----------
        array : NDArray[np.float64]
            Data array of shape (n_samples, n_columns).
        dates : Any
            Date information corresponding to the data samples.
        """
        unfiltered_batch_size = len(dates)
        filtered_array = self._filter(array, dates)
        filtered_dates = self._filter(dates, dates)

        # Compute offset of first filtered date within this batch (for tendency collectors)
        # This is O(log batch_size), not O(log 10B dataset)
        if len(filtered_dates) > 0:
            first_offset_in_batch = int(np.searchsorted(dates, filtered_dates[0]))
        else:
            first_offset_in_batch = None

        if len(filtered_array) == 0:
            # Still need to track unfiltered dates even if all filtered out
            for c in self._tendencies_collectors.values():
                c.update(
                    np.empty((0, 0)),
                    np.array([]),
                    unfiltered_batch_size=unfiltered_batch_size,
                    first_offset_in_batch=None,
                )
            return

        if self._collector is None:
            num_columns = filtered_array.shape[1]
            names = self._variables_names
            column_names = [str(i) if names is None else names[i] for i in range(num_columns)]

            # Single collector for all columns
            self._collector = _Collector(column_names)

            # Constant collectors
            for i, name in enumerate(names):
                self._constants_collectors[name] = _ConstantsCollector(i, column_names, name)

            # Tendency collectors
            for name, delta in self._tendencies.items():
                self._tendencies_collectors[name] = _TendencyCollector(column_names, delta)

        # Update all columns at once
        self._collector.update(filtered_array)

        for c in self._constants_collectors.values():
            c.update(filtered_array)

        # For tendencies, pass offset information for efficient index computation
        for c in self._tendencies_collectors.values():
            c.update(
                filtered_array,
                filtered_dates,
                unfiltered_batch_size=unfiltered_batch_size,
                first_offset_in_batch=first_offset_in_batch,
            )

    def statistics(self) -> dict[str, NDArray[np.float64]]:
        """Compute final statistics from all collected data.

        Retrieves statistics from the standard collector and all tendency collectors.
        Tendency statistics are prefixed with 'statistics_tendencies_{name}_'.

        Returns
        -------
        dict[str, NDArray[np.float64]]
            Dictionary containing statistics arrays. Keys include 'mean', 'minimum',
            'maximum', 'stdev', and tendency statistics with keys like
            'statistics_tendencies_{name}_mean'.
        """
        if self._collector is None:
            LOG.warning("No statistics collected")
            return {_: np.array([np.nan]) for _ in STATISTICS}

        result = self._collector.statistics()

        # Finalise tendency collectors before getting statistics
        for name, collector in self._tendencies_collectors.items():
            collector.finalise()
            tendencies = collector.statistics()

            for key in STATISTICS:
                result[f"statistics_tendencies_{name}_{key}"] = tendencies[key]

        return result

    def constant_variables(self) -> list[str]:
        """Get the list of variables that are constant over time.

        Returns
        -------
        list[str]
            List of variable names that are constant.
        """
        constants = []
        for name, collector in self._constants_collectors.items():
            if collector.is_constant:
                constants.append(name)
        return constants

    def add_to_dataset(self, dataset: Any) -> None:
        """Add collected statistics to the dataset.

        Parameters
        ----------
        dataset : Any
            The dataset object to which statistics will be added.
        """
        stats = self.statistics()
        for name, data in stats.items():
            assert data.dtype == np.float64, f"Expected float64 {name}, got {data.dtype}"
            dataset.add_array(name=name, data=data, dimensions=("variable",), overwrite=True)

        constants = self.constant_variables()

        variables_metadata = dataset.get_metadata("variables_metadata", {}).copy()
        for k in constants:
            if k in variables_metadata:
                variables_metadata[k]["constant_in_time"] = True

        dataset.update_metadata(constant_fields=constants, variables_metadata=variables_metadata)

    def adjust_partial_statistics(self, dataset, state) -> None:
        """Adjust statistics for a specific group and data range.

        For tendency collectors, this fills in the tendencies for dates at the
        start of the group that couldn't be computed during initial processing
        (because we needed data from the previous group)
        """

        for name, n_missing in state.missing_tendencies_count.items():
            tc = state.collector._tendencies_collectors[name]
            tc.adjust_partial_statistics(dataset, state.start, n_missing, state.collector._filter)

    def serialise(self, path, group, start, end) -> None:
        state = _State(group=group, start=start, end=end, collector=self)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load_precomputed(cls, dataset, precomputed):
        states = []
        for item in tqdm.tqdm(precomputed, desc="Loading precomputed statistics"):
            with open(item, "rb") as f:
                state = pickle.load(f)
                state.path = item
                states.append(state)

        states = sorted(states, key=lambda x: x.group)

        # Validate states
        offset = 0
        for i, state in enumerate(states):
            if state.group != i:
                raise ValueError(f"Missing statistics for group {i}")

            if state.start != offset:
                raise ValueError(f"Statistics for group {i} has start {state.start}, expected {offset}")

            offset = state.end

        if offset != len(dataset.data):
            raise ValueError(f"Statistics end {offset} does not match dataset length {len(dataset.data)}")

        # Adjust partial statistics
        for state in tqdm.tqdm(states, desc="Adjusting partial statistics", total=len(states)):
            state.collector.adjust_partial_statistics(dataset, state)

        # Merge all collectors
        return reduce(lambda a, b: a.merge(b), [s.collector for s in states])
