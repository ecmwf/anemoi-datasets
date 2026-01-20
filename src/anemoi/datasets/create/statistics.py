# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


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


class _State:
    def __init__(self, group: int, start: int, end: int, collector: "StatisticsCollector") -> None:
        self.group = group
        self.start = start
        self.end = end
        self.collector = collector


class _Base:
    def __init__(self, column_names: list[str] | None = None) -> None:
        self._column_names = column_names

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._column_names == other._column_names

    def __repr__(self, extra: str = ""):
        return f"{self.__class__.__name__}(column_names={self._column_names}" + (f", {extra}" if extra else "") + ")"


class _CollectorBase(_Base):
    def __init__(self, column_names: list[str]) -> None:
        num_columns = len(column_names)
        self._count = np.zeros(num_columns, dtype=np.int64)
        self._min = np.full(num_columns, np.inf, dtype=np.float64)
        self._max = np.full(num_columns, -np.inf, dtype=np.float64)
        self._column_names = column_names
        self._mean = np.zeros(num_columns, dtype=np.float64)
        self._m2 = np.zeros(num_columns, dtype=np.float64)

    def __eq__(self, other: Any) -> bool:

        if not isinstance(other, self.__class__):
            return False

        return (
            super().__eq__(other)
            and np.array_equal(self._count, other._count)
            and np.array_equal(self._min, other._min)
            and np.array_equal(self._max, other._max)
            and np.array_equal(self._mean, other._mean)
            and np.array_equal(self._m2, other._m2)
        )

    def __repr__(self, extra: str = ""):
        return super().__repr__(
            f"count={self._count}, min={self._min}, max={self._max}, mean={self._mean}, m2={self._m2}"
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

    def adjust_partial_statistics(self, dataset, start, end) -> None:
        """Adjust statistics for a specific group and data range."""
        # Nothing to do for the main collector
        pass


class _TendencyCollector(_CollectorBase):
    def __init__(self, column_names: list[str], delta: int) -> None:
        super().__init__(column_names)
        self._delta = delta
        # Only keep a sliding window of the last 'delta' rows
        self._window = None

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return (
            super().__eq__(other)
            and self._delta == other._delta
            and (
                (self._window is None and other._window is None)
                or (
                    self._window is not None
                    and other._window is not None
                    and np.array_equal(self._window, other._window)
                )
            )
        )

    def __repr__(self, extra: str = ""):
        return super().__repr__(f"delta={self._delta}, window={self._window}" + (f", {extra}" if extra else ""))

    @classmethod
    def _merge(cls, a: "_TendencyCollector", b: "_TendencyCollector", result: "_TendencyCollector"):
        if a._delta != b._delta:
            raise ValueError("Cannot merge tendency collectors with different deltas")
        _CollectorBase._merge(a, b, result)
        result._delta = a._delta
        assert False, "Merging tendency collectors with non-empty windows is not supported"
        return result

    def merge(self, other: "_TendencyCollector") -> "_TendencyCollector":
        result = _TendencyCollector(self._column_names, self._delta)
        self._merge(self, other, result)
        return result

    def update(self, data: NDArray[np.float64]) -> None:
        # Concatenate window with new data for tendency computation
        if self._window is None:
            combined = data
        else:
            combined = np.concatenate([self._window, data], axis=0)

        # Compute tendencies wherever we have enough history
        if len(combined) > self._delta:
            tendencies = combined[self._delta :] - combined[: -self._delta]
            super().update(tendencies)

        # Update sliding window: keep only last 'delta' rows
        self._window = combined[-self._delta :].copy()

    def finalise(self) -> None:
        pass

    def adjust_partial_statistics(self, dataset, start, end) -> None:
        """Adjust statistics for a specific group and data range."""
        if start < self._delta:
            # No adjustment needed for the first group
            return

        # For tendencies, we need to remove the influence of the first 'delta' rows
        # Get the delta rows from the dataset

        delta_rows = dataset.data[start - self._delta : start]  # noqa: F841

        # TODO: Implement the adjustment logic here

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_window", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._window = None


class _ConstantsCollector(_Base):
    def __init__(self, index, column_names: list[str], name: str) -> None:
        super().__init__(column_names)
        self._index = index
        self._name = name
        self._is_constant = True
        self._first = None
        self._nans = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_first", None)
        state.pop("_nans", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._first = None
        self._nans = None

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return (
            super().__eq__(other)
            and self._index == other._index
            and self._name == other._name
            and self._is_constant == other._is_constant
            and (
                (self._first is None and other._first is None)
                or (self._first is not None and other._first is not None and np.array_equal(self._first, other._first))
            )
            and (
                (self._nans is None and other._nans is None)
                or (self._nans is not None and other._nans is not None and np.array_equal(self._nans, other._nans))
            )
        )

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

        result._first = a._first.copy() if a._first is not None else b._first.copy() if b._first is not None else None
        result._nans = a._nans.copy() if a._nans is not None else b._nans.copy() if b._nans is not None else None

        if not a._is_constant or not b._is_constant:
            result._is_constant = False
        else:
            eq = a._first == b._first if a._first is not None and b._first is not None else True
            both_nan = (a._nans & np.isnan(b._first)) if a._nans is not None and b._first is not None else True
            result._is_constant = np.all(eq | both_nan)

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

    def adjust_partial_statistics(self, dataset, start, end) -> None:
        """Adjust statistics for a specific group and data range."""
        # Nothing to do for constant collector
        pass


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
    ) -> None:
        self._filter = filter

        self._collector = _collector
        self._variables_names = variables_names
        self._allow_nans = allow_nans
        self._tendencies = tendencies or {}
        self._tendencies_collectors = {}
        self._constants_collectors = {}

    @classmethod
    def combine_collectors(cls, dataset, collectors, filter) -> "StatisticsCollector":
        """Combine multiple Statistics
        Collectors into a single one by merging their statistics.
        """
        # TODO: implement merging logic
        return reduce(lambda a, b: a.merge(b), collectors)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return (
            self._variables_names == other._variables_names
            and self._allow_nans == other._allow_nans
            and self._tendencies == other._tendencies
            and self._collector == other._collector
            and self._tendencies_collectors == other._tendencies_collectors
            and self._constants_collectors == other._constants_collectors
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(variables_names={self._variables_names}, "
            f"allow_nans={self._allow_nans}, tendencies={self._tendencies}, "
            f"collector={self._collector}, tendencies_collectors={self._tendencies_collectors}, "
            f"constants_collectors={self._constants_collectors})"
        )

    def merge(self, other: "StatisticsCollector") -> "StatisticsCollector":
        if self._variables_names != other._variables_names:
            raise ValueError("Cannot merge StatisticsCollectors with different variable names")

        variables_names = self._variables_names

        if self._allow_nans != other._allow_nans:
            raise ValueError("Cannot merge StatisticsCollectors with different allow_nans settings")

        allow_nans = self._allow_nans

        collector = self._collector.merge(other._collector)

        tendencies_collectors = {}
        for delta in self._tendencies.keys() | other._tendencies.keys():
            if delta not in self._tendencies or delta not in other._tendencies:
                raise ValueError("Cannot merge StatisticsCollectors with different tendencies settings")
            tendencies_collectors[delta] = self._tendencies_collectors[delta].merge(other._tendencies_collectors[delta])

        LOG.warning("Merging StatisticsCollectors with tendencies is not fully supported yet, no filter")

        return StatisticsCollector(
            _collector=collector,
            variables_names=variables_names,
            allow_nans=allow_nans,
            tendencies=tendencies_collectors,
        )

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

        array = self._filter(array, dates)
        if len(array) == 0:
            return

        if self._collector is None:
            num_columns = array.shape[1]
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
        self._collector.update(array)

        for c in self._constants_collectors.values():
            c.update(array)

        # For tendencies, just buffer the data
        for c in self._tendencies_collectors.values():
            c.update(array)

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

    def adjust_partial_statistics(self, dataset, start, end) -> None:
        """Adjust statistics for a specific group and data range."""

        # Update all columns at once
        self._collector.adjust_partial_statistics(dataset, start, end)

        for c in self._constants_collectors.values():
            c.adjust_partial_statistics(dataset, start, end)

        # For tendencies, just buffer the data
        for c in self._tendencies_collectors.values():
            c.adjust_partial_statistics(dataset, start, end)

    def serialise(self, path, group, start, end) -> None:
        state = _State(group=group, start=start, end=end, collector=self)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load_precomputed(cls, dataset, precomputed, filter):
        states = []
        for item in tqdm.tqdm(precomputed, desc="Loading precomputed statistics"):
            with open(item, "rb") as f:
                state = pickle.load(f)
                states.append(state)

        states = sorted(states, key=lambda x: x.group)

        try:
            offset = 0
            for i, stat in enumerate(states):
                if stat[0].group != i:
                    raise ValueError(f"Missing statistics for group {i}")

                if stat[0].start != offset:
                    raise ValueError(f"Statistics for group {i} has start {stat[0].start}, expected {offset}")

                offset = stat[0].end

            if offset != len(dataset.data):
                raise ValueError(f"Statistics end {offset} does not match dataset length {len(dataset.data)}")
        except Exception as e:
            LOG.error("Error validating precomputed statistics: %s", e)
            # raise

        for state in tqdm.tqdm(states, desc="Adjusting partial statistics", total=len(states)):
            state.collector.adjust_partial_statistics(dataset, state.start, state.end)

        return cls.combine_collectors(dataset, [s.collector for s in states], filter)
