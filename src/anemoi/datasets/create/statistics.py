# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

LOG = logging.getLogger(__name__)

STATISTICS = ("mean", "minimum", "maximum", "stdev")


class _Base:
    def __init__(self, num_columns: int, column_names: list[str] | None = None) -> None:
        self._num_columns = num_columns
        self._column_names = column_names


class _CollectorBase(_Base):
    def __init__(self, num_columns, column_names: list[str] | None = None) -> None:
        super().__init__(num_columns, column_names)
        self._count = np.zeros(num_columns, dtype=np.int64)
        self._min = np.full(num_columns, np.inf, dtype=np.float64)
        self._max = np.full(num_columns, -np.inf, dtype=np.float64)
        self._column_names = column_names or [str(i) for i in range(num_columns)]
        self._mean = np.zeros(num_columns, dtype=np.float64)
        self._m2 = np.zeros(num_columns, dtype=np.float64)

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
    def __init__(self, num_columns: int, column_names: list[str], name: str, delta: int) -> None:
        super().__init__(num_columns, column_names)
        self._name = name
        self._delta = delta
        # Only keep a sliding window of the last 'delta' rows
        self._window = None

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


class _ConstantsCollector(_CollectorBase):
    def __init__(self, index, num_columns: int, column_names: list[str], name: str) -> None:
        super().__init__(num_columns, column_names)
        self._index = index
        self._name = name
        self._is_constant = True
        self._first = None
        self._nans = None

    def update(self, data: NDArray[np.float64]) -> None:

        if not self._is_constant:
            # No need to check further
            return

        data = data[:, self._index]

        if self._first is None:
            self._first = data[0].copy()
            self._nans = np.isnan(self._first)

        # print(f"First value for constant check in column {self._name}: {self._first}")
        # assert False, "Debug stop"

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
    ) -> None:
        self._filter = filter

        self._collector = None
        self._variables_names = variables_names
        self._allow_nans = allow_nans
        self._tendencies = tendencies or {}
        self._tendencies_collectors = {}
        self._constants_collectors = {}

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
            self._collector = _Collector(num_columns, column_names)

            # Constant collectors
            for i, name in enumerate(names):
                self._constants_collectors[name] = _ConstantsCollector(i, num_columns, column_names, name)

            # Tendency collectors
            for name, delta in self._tendencies.items():
                self._tendencies_collectors[name] = _TendencyCollector(num_columns, column_names, name, delta)

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
